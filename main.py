import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["HF_HOME"] = "./hf_cache"
import json
import tqdm
import torch
import shutil
import logging
import argparse
import warnings
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from metrics import MetricCalculator

# Assuming these are your custom modules
from dataset.LavalObjaverseDataset import EvalDataset

# Suppress noisy warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
os.environ['HF_HOME'] = './hf_cache'

logger = logging.getLogger(__name__)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def log_validation(dataloader, pipeline, args, metric_fn):
    device = get_device()
    output_dir = Path(args.output_dir).resolve()
    res_json_path = output_dir / f"{args.baseline}_{args.task}_results.json"
    
    # 🔹 Resume Logic (支援所有指標)
    if args.skip_exist and res_json_path.exists():
        with open(res_json_path, 'r') as f:
            evaluation_results = json.load(f)
        # 相容性處理：舊版 list → 新版 dict
        if isinstance(evaluation_results.get("data_pair"), list):
            evaluation_results["data_pair"] = {
                str(item["sample_idx"]): item for item in evaluation_results["data_pair"]
            }
        # 確保 average 鍵存在
        if "average" not in evaluation_results:
            evaluation_results["average"] = {}
        logger.info(f"📦 Resumed: {len(evaluation_results['data_pair'])} samples already processed.")
    else:
        evaluation_results = {"average": {}, "data_pair": {}}

    # Load Metadata
    with open(args.pair_info, 'r') as f:
        data_pairs = json.load(f)

    # 🔹 定義所有指標鍵（與 MetricCalculator 返回順序嚴格對應）
    METRIC_KEYS = [
        "psnr", "spsnr", "ssim", "lpips",                    # Unmasked (always computed)
        "psnr_mask", "spsnr_mask", "ssim_mask", "lpips_mask", # Masked (None if no mask)
        "depth_acc", "depth_mse", "mask_iou"                  # Depth + IoU (None if no depth/mask)
    ]

    bar = tqdm.tqdm(dataloader, desc="Evaluating")
    
    for batch in bar:
        if batch is None: 
            continue

        indices = batch['idx'].tolist()
        keep_indices = [i for i, idx in enumerate(indices) if str(idx) not in evaluation_results["data_pair"]]
        
        if not keep_indices: 
            continue
        
        # Filter batch for unprocessed samples only
        k_idx = torch.tensor(keep_indices, device=device)
        filtered_batch = {
            k: v[k_idx.to(v.device)] if isinstance(v, torch.Tensor) else v 
            for k, v in batch.items()
        }
        
        # Inference
        with torch.autocast("cuda"):
            outputs = pipeline(filtered_batch)  # Expected shape: (B, F, C, H, W) or tuple
            if isinstance(outputs, tuple):
                relit, depth_pred, mask_pred = outputs
            else:
                relit, depth_pred, mask_pred = outputs, None, None
            targets = filtered_batch['target_images']
            depth_gt = filtered_batch.get('target_depths')
            mask_gt = filtered_batch.get('target_mask')  # None if key not present

        # Calculate Metrics
        B_filtered = relit.shape[0]
        for b in range(B_filtered):
            sample_idx = filtered_batch['idx'][b].item()
            meta = data_pairs[sample_idx]
            
            # 🔹 安全獲取所有可選輸入（不存在則為 None）
            metric_vals = metric_fn(
                relit[b:b+1], 
                targets[b:b+1],
                mask_pred=mask_pred,   # None if not present
                mask_gt=mask_gt,                              # None if not present
                depth_pred=depth_pred,  # None if not present
                depth_gt=depth_gt,      # None if not present
                average=True  # ✅ Returns scalars or None
            )
            
            # Build {key: value} dict for easy access
            sample_metrics = dict(zip(METRIC_KEYS, metric_vals))
            
            current_eval = {
                "sample_idx": sample_idx,
                **sample_metrics,  # Expand all metrics (None values included)
                "pred_image": [],
                "pred_depth": []
            }

            # Save per view
            for v in range(relit.shape[1]):
                view_name = meta["view"][v].split('.')[0]
                view_dir = output_dir / str(sample_idx) / view_name
                view_dir.mkdir(parents=True, exist_ok=True)
                
                image_path = view_dir / "pred_relight.png"
                save_image(relit[b, v], image_path)
                current_eval["pred_image"].append(str(image_path))

                depth_path = view_dir / "pred_depths.png"
                save_image(relit[b, v], depth_path)
                current_eval["pred_depth"].append(str(depth_path))

                if args.save_gt:
                    save_image(filtered_batch["target_images"][b, v], view_dir / "gt.png")
                if args.save_ref:
                    save_image(filtered_batch["source_images"][b, v], view_dir / "ref.png")

            # Store result
            evaluation_results["data_pair"][str(sample_idx)] = current_eval

            # 🔹 Update averages for ALL metrics (ROBUST version)
            all_vals = list(evaluation_results["data_pair"].values())
            for k in METRIC_KEYS:
                # Collect valid numeric values only
                valid = []
                for x in all_vals:
                    if k in x and x[k] is not None:
                        try:
                            val = float(x[k])
                            if np.isfinite(val):
                                valid.append(val)
                        except (TypeError, ValueError):
                            continue
                
                if len(valid) >= 1:
                    mean_val = np.mean(valid)
                    if np.isfinite(mean_val):
                        evaluation_results["average"][k] = float(mean_val)

            # 🔹 Progress Bar: Dynamic display based on available metrics
            display_keys = ["psnr", "spsnr", "ssim", "lpips"]
            # Only add masked PSNR if it's actually computed and finite
            psnr_mask_val = evaluation_results["average"].get("psnr_mask")
            if psnr_mask_val is not None and np.isfinite(psnr_mask_val):
                display_keys.append("psnr_mask")
                
            postfix = {}
            for k in display_keys:
                val = evaluation_results["average"].get(k)
                if val is not None and np.isfinite(val):
                    key_display = k.upper().replace("_MASK", "M")
                    postfix[key_display] = f"{val:.2f}"
            bar.set_postfix(postfix)

            # Atomic JSON save (None → null in JSON)
            with open(res_json_path, 'w') as f:
                json.dump(evaluation_results, f, indent=4)

    # 🔹 Final output: Ensure all metrics are in average (even if some samples missing)
    all_vals = list(evaluation_results["data_pair"].values())
    for k in METRIC_KEYS:
        valid = [x[k] for x in all_vals if k in x and x[k] is not None]
        if valid and k not in evaluation_results["average"]:
            evaluation_results["average"][k] = float(np.mean(valid))
            
    return evaluation_results

def main(args):
    logging.basicConfig(level=logging.INFO)
    device = get_device()

    dataset = EvalDataset(args.dataset_path, args.pair_info, black_background=True)

    # Pipeline selection
    if args.baseline == "LightSwitch":
        from pipeline.LightSwitch import LightSwitchPipeline
        pipeline = LightSwitchPipeline()
    elif args.baseline == "DiffusionRenderer":
        from pipeline.DiffusionRenderer import Diffusion_RendererPipeline
        num_video_frames = int(args.pair_info.split('/')[-1].split('_')[0])
        pipeline = Diffusion_RendererPipeline(num_video_frames=num_video_frames)
        dataset = EvalDataset(args.dataset_path, args.pair_info, black_background=True)
    elif args.baseline == "NeuralGaffer":
        from pipeline.NeuralGaffer import NeuralGafferPipeline
        pipeline = NeuralGafferPipeline()
        dataset = EvalDataset(args.dataset_path, args.pair_info, black_background=False)
    elif args.baseline == "ReLi3D":
        from pipeline.Reli3D import Reli3DPipeline
        pipeline = Reli3DPipeline(
            device=str(device),
            dtype=torch.float32,
            reli3d_root=args.reli3d_root,
            config_path=args.reli3d_config,
            checkpoint_path=args.reli3d_checkpoint,
            blender_path=args.reli3d_blender_path,
            cache_dir=args.reli3d_cache_dir,
            texture_size=args.reli3d_texture_size,
            remesh=args.reli3d_remesh,
            vertex_count=args.reli3d_vertex_count,
            convert_source_view_cv_to_reli3d=(not args.reli3d_no_source_view_conversion),
            debug=args.reli3d_debug,
            mapper_dataset_is_repaired=bool(args.reli3d_mapper_dataset_repaired),
            export_case_inputs_dir=args.reli3d_export_case_inputs_dir,
            use_official_infer=bool(args.reli3d_use_official_infer),
            render_source_for_debug=args.reli3d_render_source_for_debug,
        )
        dataset = EvalDataset(args.dataset_path, args.pair_info, black_background=True)
    elif args.baseline == "Trained-NeuralGaffer":
        from pipeline.NeuralGaffer import NeuralGafferPipeline
        raise NotImplementedError(f"Baseline {args.baseline} not supported.")
        # pipeline = DiffusionRendererPipeline(resume_from_checkpoint)
    else:
        raise NotImplementedError(f"Baseline {args.baseline} not supported.")
    
    # Dataset Setup
    # dataset = EvalDataset(args.dataset_path, args.pair_info)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    metric_fn = MetricCalculator(device)

    log_validation(dataloader, pipeline, args, metric_fn)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_pretrained", type=str, default='models/2d_training/relight')
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default='./output/')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataset_path", type=str, default="/media/HDD1/hejun/LavalObjaverseDataset")
    parser.add_argument("--baseline", type=str, default="LightSwitch")
    parser.add_argument("--skip_exist", action='store_true')
    parser.add_argument("--pair_info", type=str, default='/media/HDD1/hejun/LavalObjaverseDataset/experimental_pair/1_to_1_mapping_pairs.json')
    parser.add_argument("--save_gt", action='store_true')
    parser.add_argument("--save_ref", action='store_true')
    parser.add_argument("--reli3d_root", type=str, default="ReLi3D")
    parser.add_argument("--reli3d_config", type=str, default=None)
    parser.add_argument("--reli3d_checkpoint", type=str, default=None)
    parser.add_argument("--reli3d_blender_path", type=str, default="blender")
    parser.add_argument("--reli3d_cache_dir", type=str, default="./output/reli3d_cache")
    parser.add_argument("--reli3d_texture_size", type=int, default=1024)
    parser.add_argument("--reli3d_remesh", type=str, default="none", choices=["none", "triangle", "quad"])
    parser.add_argument("--reli3d_vertex_count", type=int, default=-1)
    parser.add_argument("--reli3d_debug", action='store_true')
    parser.add_argument("--reli3d_no_source_view_conversion", action='store_true')
    parser.add_argument("--reli3d_mapper_dataset_repaired", type=int, default=1, choices=[0, 1])
    parser.add_argument("--reli3d_export_case_inputs_dir", type=str, default=None)
    parser.add_argument("--reli3d_use_official_infer", type=int, default=1, choices=[0, 1])
    parser.add_argument("--reli3d_render_source_for_debug", action='store_true')

    args = parser.parse_args()
    
    if args.baseline == "LightSwitch" and args.batch_size != 1:
        raise ValueError(f"Invalid batch size {args.batch_size} for Baseline f{args.baseline}")
    # Path preparation
    args.task = args.pair_info.split('/')[-1].split('.')[0]
    args.output_dir = os.path.join(args.output_dir, args.baseline, args.task)
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)

'''
dataset_path : /media/HDD1/hejun/LavalObjaverseDataset on 0823
dataset_path : /media/HDD2/hejun/LavalObjaverseDataset on 0321
'''
