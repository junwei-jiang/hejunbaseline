import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from pipeline import BaselinePipeline


class Reli3DPipeline(BaselinePipeline):
    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        reli3d_root: str = "ReLi3D",
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        blender_path: str = "blender",
        cache_dir: str = "./output/reli3d_cache",
        texture_size: int = 1024,
        remesh: str = "none",
        vertex_count: int = -1,
        convert_source_view_cv_to_reli3d: bool = True,
        debug: bool = False,
        mapper_dataset_is_repaired: bool = True,
        export_case_inputs_dir: Optional[str] = None,
        use_official_infer: bool = True,
        render_source_for_debug: bool = False,
        dump_camera_debug: bool = False,
        export_principal_mode: str = "dataset",
        export_fov_mode: str = "xy",
        export_coord_system: str = "ogl",
    ):
        super().__init__(device=device, dtype=dtype)
        self.device_obj = torch.device(
            "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        )

        self.reli3d_root = Path(reli3d_root).resolve()
        self.blender_path = blender_path
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.texture_size = int(texture_size)
        self.remesh = remesh
        self.vertex_count = int(vertex_count)
        self.convert_source_view_cv_to_reli3d = bool(convert_source_view_cv_to_reli3d)
        self.debug = bool(debug)
        self.mapper_dataset_is_repaired = bool(mapper_dataset_is_repaired)
        self.use_official_infer = bool(use_official_infer)
        self.render_source_for_debug = bool(render_source_for_debug)
        self.dump_camera_debug = bool(dump_camera_debug)
        self.export_principal_mode = str(export_principal_mode)
        self.export_fov_mode = str(export_fov_mode)
        self.export_coord_system = str(export_coord_system)
        self._printed_source_render_debug_hint = False
        self.export_case_inputs_dir = (
            Path(export_case_inputs_dir).resolve() if export_case_inputs_dir else None
        )
        if self.export_case_inputs_dir is not None:
            self.export_case_inputs_dir.mkdir(parents=True, exist_ok=True)
        self._printed_export_hint = False

        if not self.reli3d_root.exists():
            raise FileNotFoundError(f"ReLi3D root not found: {self.reli3d_root}")

        self._init_reli3d_imports()

        self.config_path = self._resolve_config_path(config_path)
        self.checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)

        self.model = None
        if not self.use_official_infer:
            self.model = self._load_model()

    def _init_reli3d_imports(self) -> None:
        if str(self.reli3d_root) not in sys.path:
            sys.path.insert(0, str(self.reli3d_root))

        from omegaconf import OmegaConf
        from src.constants import Names
        from src.data.reli3d_mapper import ReLi3DMapper
        from src.systems.feed_forward_system import FeedForwardSystem
        from src.utils.config import instantiate_config
        from src.utils.misc import load_module_weights

        self.OmegaConf = OmegaConf
        self.Names = Names
        self.ReLi3DMapper = ReLi3DMapper
        self.FeedForwardSystem = FeedForwardSystem
        self.instantiate_config = instantiate_config
        self.load_module_weights = load_module_weights

    def _resolve_config_path(self, config_path: Optional[str]) -> Path:
        if config_path is not None:
            p = Path(config_path).expanduser().resolve()
        else:
            p = (self.reli3d_root / "artifacts" / "model" / "config.yaml").resolve()
            if not p.exists():
                alt = (self.reli3d_root / "artifacts" / "model" / "raw.yaml").resolve()
                if alt.exists():
                    p = alt
        if not p.exists():
            raise FileNotFoundError(
                f"ReLi3D config not found: {p}. Please set --reli3d_config."
            )
        return p

    def _resolve_checkpoint_path(self, checkpoint_path: Optional[str]) -> Path:
        if checkpoint_path is not None:
            p = Path(checkpoint_path).expanduser().resolve()
        else:
            env_ckpt = os.environ.get("RELI3D_CHECKPOINT")
            if env_ckpt:
                p = Path(env_ckpt).expanduser().resolve()
            else:
                p = (
                    self.reli3d_root / "artifacts" / "model" / "reli3d_final.ckpt"
                ).resolve()
        if not p.exists():
            raise FileNotFoundError(
                f"ReLi3D checkpoint not found: {p}. Please set --reli3d_checkpoint."
            )
        return p

    def _load_system_cfg(self, config_path: Path):
        cfg = self.OmegaConf.load(config_path)
        if "system" in cfg:
            system_cfg = cfg.system
        elif "main_module" in cfg and "system" in cfg.main_module:
            system_cfg = cfg.main_module.system
        else:
            raise ValueError(
                f"Config at {config_path} must contain `system` or `main_module.system`."
            )
        return self.instantiate_config(
            self.FeedForwardSystem.Config,
            self.OmegaConf.to_container(system_cfg, resolve=True),
        )

    def _load_model(self):
        cfg = self._load_system_cfg(self.config_path)
        model = self.FeedForwardSystem(cfg)
        state_dict = self.load_module_weights(
            str(self.checkpoint_path), module_name="system", map_location="cpu"
        )[0]
        model.load_state_dict(state_dict)
        model.eval()
        model = model.to(self.device_obj)
        return model

    def _convert_source_view_for_reli3d(self, source_view: torch.Tensor) -> torch.Tensor:
        if not self.convert_source_view_cv_to_reli3d:
            return source_view
        cv_to_blender_cv = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=source_view.dtype,
            device=source_view.device,
        )
        return torch.matmul(source_view, cv_to_blender_cv)

    def _blender_to_ogl_c2w(self, c2w_blender: np.ndarray) -> np.ndarray:
        blender_to_gl = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return blender_to_gl @ c2w_blender

    def _export_case_inputs(
        self,
        sample_idx: int,
        source_images: torch.Tensor,
        source_mask: torch.Tensor,
        source_view: torch.Tensor,
        source_Ks: torch.Tensor,
    ) -> Optional[Path]:
        export_root = self.export_case_inputs_dir
        if export_root is None:
            export_root = self.cache_dir / "_official_inputs"
            export_root.mkdir(parents=True, exist_ok=True)

        case_name = f"sample_{int(sample_idx):06d}"
        case_dir = export_root / case_name
        rgba_dir = case_dir / "rgba"
        case_dir.mkdir(parents=True, exist_ok=True)
        rgba_dir.mkdir(parents=True, exist_ok=True)

        F, _, H, W = source_images.shape
        frames = []
        camera_debug = []
        for i in range(F):
            rgb = source_images[i].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            rgb_u8 = (rgb * 255.0).astype(np.uint8)

            mask = source_mask[i].detach().cpu().clamp(0, 1).numpy()
            if mask.ndim == 3:
                mask = mask[0]
            alpha_u8 = (mask * 255.0).astype(np.uint8)

            rgba = np.concatenate([rgb_u8, alpha_u8[..., None]], axis=-1)
            bgra = rgba[..., [2, 1, 0, 3]]
            fn = f"{i:04d}.png"
            cv2.imwrite(str(rgba_dir / fn), bgra)

            K = source_Ks[i].detach().cpu().float()
            fx = float(K[0, 0].item())
            fy = float(K[1, 1].item())
            cx = float(K[0, 2].item())
            cy = float(K[1, 2].item())
            fov_x = 2.0 * np.arctan(float(W) / (2.0 * max(fx, 1e-8)))
            fov_y = 2.0 * np.arctan(float(H) / (2.0 * max(fy, 1e-8)))

            if self.export_fov_mode == "scalar_x":
                camera_fov = [float(fov_x), float(fov_x)]
            else:
                camera_fov = [float(fov_x), float(fov_y)]

            if self.export_principal_mode == "center":
                principal = [float(W * 0.5), float(H * 0.5)]
            else:
                principal = [float(cx), float(cy)]

            c2w_blender = source_view[i].detach().cpu().float().numpy()
            if self.export_coord_system == "ogl":
                c2w_export = self._blender_to_ogl_c2w(c2w_blender)
            else:
                c2w_export = c2w_blender

            R = c2w_export[:3, :3]
            t = c2w_export[:3, 3]
            det_r = float(np.linalg.det(R))
            t_norm = float(np.linalg.norm(t))
            alpha_ratio = float((alpha_u8 > 0).mean())

            frames.append(
                {
                    "file_path": f"rgba/{fn}",
                    "transform_matrix": c2w_export.tolist(),
                    "camera_fov": camera_fov,
                    "camera_principal_point": principal,
                    "view_index": i,
                }
            )

            camera_debug.append(
                {
                    "view_index": i,
                    "fx": fx,
                    "fy": fy,
                    "cx": cx,
                    "cy": cy,
                    "export_principal": principal,
                    "fov_x": float(fov_x),
                    "fov_y": float(fov_y),
                    "export_fov": camera_fov,
                    "det_R": det_r,
                    "cam_radius": t_norm,
                    "alpha_ratio": alpha_ratio,
                }
            )

        transforms = {
            "object_uid": case_name,
            "coordinate_system": self.export_coord_system,
            "frames": frames,
        }
        with open(case_dir / "transforms.json", "w", encoding="utf-8") as f:
            json.dump(transforms, f, indent=2)

        if self.dump_camera_debug:
            summary = {
                "case": case_name,
                "width": int(W),
                "height": int(H),
                "export_principal_mode": self.export_principal_mode,
                "export_fov_mode": self.export_fov_mode,
                "export_coord_system": self.export_coord_system,
                "det_R_min": float(min(x["det_R"] for x in camera_debug)) if camera_debug else None,
                "det_R_max": float(max(x["det_R"] for x in camera_debug)) if camera_debug else None,
                "radius_min": float(min(x["cam_radius"] for x in camera_debug)) if camera_debug else None,
                "radius_max": float(max(x["cam_radius"] for x in camera_debug)) if camera_debug else None,
                "alpha_ratio_min": float(min(x["alpha_ratio"] for x in camera_debug)) if camera_debug else None,
                "alpha_ratio_max": float(max(x["alpha_ratio"] for x in camera_debug)) if camera_debug else None,
                "views": camera_debug,
            }
            with open(case_dir / "camera_debug.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

        if not self._printed_export_hint:
            cmd = (
                f"python demos/reli3d/infer_from_transforms.py "
                f"--input-root {export_root} --objects {case_name} "
                f"--output-root {export_root / 'official_outputs'} --overwrite"
            )
            with open(export_root / "official_infer_cmd.txt", "w", encoding="utf-8") as f:
                f.write(cmd + "\n")
            print(f"[ReLi3D export] case exported to: {case_dir}")
            print(f"[ReLi3D export] official command saved: {export_root / 'official_infer_cmd.txt'}")
            self._printed_export_hint = True

        return case_dir

    def _reconstruct_mesh_official(
        self,
        case_dir: Path,
        sample_dir: Path,
        target_lighting: torch.Tensor,
    ) -> Tuple[Path, Path]:
        mesh_path = sample_dir / "mesh.glb"
        hdr_path = sample_dir / "illumination.hdr"

        out_root = sample_dir / "official_outputs"
        cmd = [
            sys.executable,
            "demos/reli3d/infer_from_transforms.py",
            "--input-root",
            str(case_dir.parent),
            "--objects",
            case_dir.name,
            "--output-root",
            str(out_root),
            "--config",
            str(self.config_path),
            "--checkpoint",
            str(self.checkpoint_path),
            "--texture-size",
            str(self.texture_size),
            "--remesh",
            str(self.remesh),
            "--overwrite",
        ]
        if self.vertex_count > 0:
            cmd += ["--vertex-count", str(self.vertex_count)]

        proc = subprocess.run(
            cmd,
            cwd=str(self.reli3d_root),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "Official ReLi3D infer failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )

        src_obj_dir = out_root / case_dir.name
        src_mesh = src_obj_dir / "mesh.glb"
        if not src_mesh.exists():
            raise FileNotFoundError(
                f"Official infer finished but mesh not found: {src_mesh}"
            )

        shutil.copy2(src_mesh, mesh_path)
        # For relighting benchmark, always use dataset target lighting.
        self._write_hdr(target_lighting, hdr_path)

        return mesh_path, hdr_path

    def _build_mapper_batch(
        self,
        object_uid: str,
        source_images: torch.Tensor,
        source_mask: torch.Tensor,
        source_view: torch.Tensor,
        source_Ks: torch.Tensor,
    ) -> Dict[Any, Any]:
        assert source_images.ndim == 4, "source_images should be [F, C, H, W]"
        F, C, H, W = source_images.shape
        if C != 3:
            raise ValueError(f"Expected RGB source images, got C={C}")

        sft_dict: Dict[str, torch.Tensor] = {
            "object_uid": torch.tensor(
                np.frombuffer(object_uid.encode("utf-8"), dtype=np.uint8)
            )
        }

        for i in range(F):
            rgb = source_images[i].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            rgb_u8 = (rgb * 255.0).astype(np.uint8)

            mask = source_mask[i].detach().cpu().clamp(0, 1).numpy()
            if mask.ndim == 3:
                mask = mask[0]
            mask_u8 = (mask * 255.0).astype(np.uint8)

            ok_rgb, rgb_buf = cv2.imencode(".jpg", rgb_u8[..., ::-1])
            if not ok_rgb:
                raise RuntimeError("Failed to encode source RGB as jpeg.")

            mr = np.zeros((H, W, 3), dtype=np.uint8)
            mr[..., 2] = mask_u8
            ok_mr, mr_buf = cv2.imencode(".jpg", mr)
            if not ok_mr:
                raise RuntimeError("Failed to encode source mask as jpeg.")

            K = source_Ks[i].detach().cpu().float()
            fx = float(K[0, 0].item())
            fy = float(K[1, 1].item())
            cx = float(K[0, 2].item())
            cy = float(K[1, 2].item())
            fov_x = 2.0 * np.arctan(float(W) / (2.0 * max(fx, 1e-8)))
            fov_y = 2.0 * np.arctan(float(H) / (2.0 * max(fy, 1e-8)))

            sft_dict[f"rgb_{i:04d}"] = torch.from_numpy(rgb_buf.copy())
            sft_dict[f"metallicroughmask_{i:04d}"] = torch.from_numpy(mr_buf.copy())
            sft_dict[f"c2w_{i:04d}"] = source_view[i].detach().cpu().float()
            sft_dict[f"fov_rad_{i:04d}"] = torch.tensor(
                [float(fov_x), float(fov_y)], dtype=torch.float32
            )
            sft_dict[f"principal_point_{i:04d}"] = torch.tensor(
                [cx, cy], dtype=torch.float32
            )

        mapper_cfg = {
            "num_views_input": F,
            "num_views_output": F,
            "train_input_views": list(range(F)),
            "train_sup_views": "random",
            "cond_height": 512,
            "cond_width": 512,
            "eval_height": 512,
            "eval_width": 512,
            "binarize_mask": True,
            "dataset_is_repaired": self.mapper_dataset_is_repaired,
            "add_pose_noise": False,
            "pose_noise_std_trans": 0.0,
            "pose_noise_std_rot": 0.0,
        }
        mapper = self.ReLi3DMapper(cfg=mapper_cfg, sft_key="safetensors", split="test")
        mapper_batch = mapper(
            {"safetensors": sft_dict, "dataset_name": "custom", "dataset_type": "pbr"}
        )

        Names = self.Names
        batch_elem = {
            Names.IMAGE.cond: mapper_batch[Names.IMAGE.cond],
            Names.IMAGE.add_suffix("mask").cond: mapper_batch[
                Names.IMAGE.add_suffix("mask").cond
            ],
            Names.OPACITY.cond: mapper_batch[Names.OPACITY.cond],
            Names.CAMERA_TO_WORLD.cond: mapper_batch[Names.CAMERA_TO_WORLD.cond],
            Names.CAMERA_POSITION.cond: mapper_batch[Names.CAMERA_POSITION.cond],
            Names.INTRINSICS.cond: mapper_batch[Names.INTRINSICS.cond],
            Names.INTRINSICS_NORMED.cond: mapper_batch[Names.INTRINSICS_NORMED.cond],
            Names.VIEW_SIZE: mapper_batch[Names.VIEW_SIZE],
            Names.VIEW_SIZE.cond: mapper_batch[Names.VIEW_SIZE.cond],
        }
        bg_key = Names.IMAGE.add_suffix("bg").cond
        if bg_key in mapper_batch:
            batch_elem[bg_key] = mapper_batch[bg_key]

        batch_out = {k: v.unsqueeze(0).to(self.device_obj) for k, v in batch_elem.items()}
        batch_out[Names.BATCH_SIZE] = 1
        return batch_out

    def _sample_paths(self, batch_idx: int, sample_idx: int) -> Tuple[Path, Path, Path]:
        sample_key = f"s{sample_idx}_b{batch_idx}"
        sample_dir = self.cache_dir / sample_key
        sample_dir.mkdir(parents=True, exist_ok=True)
        mesh_path = sample_dir / "mesh.glb"
        hdr_path = sample_dir / "illumination.hdr"
        return sample_dir, mesh_path, hdr_path

    def _prepare_official_meshes_for_batch(self, batch) -> None:
        if not self.use_official_infer:
            return

        source_images = batch["source_images"]
        source_view = batch["source_view"]
        source_Ks = batch["source_Ks"]
        source_mask = batch.get("source_mask")
        target_lighting = batch["target_lighting"]

        if source_mask is None:
            source_mask = torch.ones_like(source_images[:, :, :1])

        export_root = self.export_case_inputs_dir
        if export_root is None:
            export_root = self.cache_dir / "_official_inputs"
            export_root.mkdir(parents=True, exist_ok=True)
        out_root = self.cache_dir / "_official_outputs"
        out_root.mkdir(parents=True, exist_ok=True)

        jobs = []
        B = source_images.shape[0]
        for b in range(B):
            sample_idx = int(batch["idx"][b].item()) if "idx" in batch else b
            sample_dir, mesh_path, hdr_path = self._sample_paths(b, sample_idx)
            if mesh_path.exists():
                # keep benchmark lighting from dataset
                self._write_hdr(target_lighting[b], hdr_path)
                continue

            source_view_reli3d = self._convert_source_view_for_reli3d(source_view[b])
            case_dir = self._export_case_inputs(
                sample_idx=sample_idx,
                source_images=source_images[b],
                source_mask=source_mask[b],
                source_view=source_view_reli3d,
                source_Ks=source_Ks[b],
            )
            if case_dir is None:
                raise RuntimeError("Official infer requires exported case inputs, but export failed.")
            jobs.append(
                {
                    "case_name": case_dir.name,
                    "sample_dir": sample_dir,
                    "mesh_path": mesh_path,
                    "hdr_path": hdr_path,
                    "target_lighting": target_lighting[b],
                }
            )

        if not jobs:
            return

        cmd = [
            sys.executable,
            "demos/reli3d/infer_from_transforms.py",
            "--input-root",
            str(export_root),
            "--objects",
        ]
        cmd.extend([j["case_name"] for j in jobs])
        cmd.extend(
            [
                "--output-root",
                str(out_root),
                "--config",
                str(self.config_path),
                "--checkpoint",
                str(self.checkpoint_path),
                "--texture-size",
                str(self.texture_size),
                "--remesh",
                str(self.remesh),
                "--overwrite",
            ]
        )
        if self.vertex_count > 0:
            cmd += ["--vertex-count", str(self.vertex_count)]

        proc = subprocess.run(cmd, cwd=str(self.reli3d_root), capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "Official ReLi3D batch infer failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )

        for j in jobs:
            src_mesh = out_root / j["case_name"] / "mesh.glb"
            if not src_mesh.exists():
                raise FileNotFoundError(f"Official batch infer missing mesh: {src_mesh}")
            shutil.copy2(src_mesh, j["mesh_path"])
            # keep benchmark lighting from dataset
            self._write_hdr(j["target_lighting"], j["hdr_path"])

    def _reconstruct_mesh(
        self,
        batch_idx: int,
        sample_idx: int,
        source_images: torch.Tensor,
        source_mask: torch.Tensor,
        source_view: torch.Tensor,
        source_Ks: torch.Tensor,
        target_lighting: torch.Tensor,
    ) -> Tuple[Path, Path]:
        sample_dir, mesh_path, hdr_path = self._sample_paths(batch_idx, sample_idx)

        if mesh_path.exists():
            if hdr_path.exists():
                return mesh_path, hdr_path
            self._write_hdr(target_lighting, hdr_path)
            return mesh_path, hdr_path

        source_view_reli3d = self._convert_source_view_for_reli3d(source_view)
        case_dir = self._export_case_inputs(
            sample_idx=sample_idx,
            source_images=source_images,
            source_mask=source_mask,
            source_view=source_view_reli3d,
            source_Ks=source_Ks,
        )
        if self.use_official_infer:
            if case_dir is None:
                raise RuntimeError("Official infer requires exported case inputs, but export failed.")
            return self._reconstruct_mesh_official(
                case_dir=case_dir,
                sample_dir=sample_dir,
                target_lighting=target_lighting,
            )

        mapper_batch = self._build_mapper_batch(
            object_uid=sample_key,
            source_images=source_images,
            source_mask=source_mask,
            source_view=source_view_reli3d,
            source_Ks=source_Ks,
        )

        with torch.no_grad(), (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.device_obj.type == "cuda"
            else nullcontext()
        ):
            mesh_list, global_dict = self.model.get_mesh(
                mapper_batch,
                texture_resolution=self.texture_size,
                remesh=self.remesh,
                vertex_count=self.vertex_count if self.vertex_count > 0 else None,
            )

        mesh = mesh_list[-1]
        mesh.export(mesh_path, file_type="glb", include_normals=True)

        # For relighting benchmark, always use dataset target lighting.
        self._write_hdr(target_lighting, hdr_path)

        return mesh_path, hdr_path

    def _write_hdr(self, target_lighting: torch.Tensor, path: Path) -> None:
        env = target_lighting.detach().cpu().float().permute(1, 2, 0).numpy()
        env = np.maximum(env, 0.0).astype(np.float32)
        cv2.imwrite(str(path), env[..., ::-1])

    def _render_with_blender(
        self,
        mesh_path: Path,
        hdr_path: Path,
        target_view: torch.Tensor,
        target_Ks: torch.Tensor,
        height: int,
        width: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with tempfile.TemporaryDirectory(prefix="reli3d_render_") as tmpdir:
            tmpdir_p = Path(tmpdir)
            work_dir = tmpdir_p / "work"
            work_dir.mkdir(parents=True, exist_ok=True)

            targets = []
            F = target_view.shape[0]
            for i in range(F):
                targets.append(
                    {
                        "c2w": target_view[i].detach().cpu().tolist(),
                        "K": target_Ks[i].detach().cpu().tolist(),
                    }
                )

            job_path = tmpdir_p / "job.json"
            with open(job_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "mesh_path": str(mesh_path),
                        "hdr_path": str(hdr_path),
                        "out_dir": str(work_dir),
                        "width": int(width),
                        "height": int(height),
                        "targets": targets,
                    },
                    f,
                )

            script_path = (
                Path(__file__).resolve().parent / "reli3d_blender_render.py"
            ).resolve()
            cmd = [
                self.blender_path,
                "--background",
                "--python",
                str(script_path),
                "--",
                "--job",
                str(job_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    "Blender render failed.\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"stdout:\n{proc.stdout}\n"
                    f"stderr:\n{proc.stderr}"
                )
            done_file = work_dir / "done.txt"
            rgb_dir = work_dir / "rgb"
            any_rgb = sorted(rgb_dir.glob("*.png")) if rgb_dir.exists() else []
            if (not done_file.exists()) and (len(any_rgb) == 0):
                raise RuntimeError(
                    "Blender command returned success but render script did not finish.\n"
                    "This often happens when using a launcher wrapper instead of the real blender binary.\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"stdout:\n{proc.stdout}\n"
                    f"stderr:\n{proc.stderr}"
                )

            rgb_list: List[np.ndarray] = []
            mask_list: List[np.ndarray] = []
            depth_list: List[np.ndarray] = []
            all_rgb_pngs = sorted((work_dir / "rgb").glob("*.png"))
            for i in range(F):
                rgba_path = work_dir / "rgb" / f"{i:04d}.png"
                if not rgba_path.exists():
                    cand = sorted((work_dir / "rgb").glob(f"{i:04d}*.png"))
                    if cand:
                        rgba_path = cand[-1]
                    elif i < len(all_rgb_pngs):
                        rgba_path = all_rgb_pngs[i]
                depth_pattern = str(work_dir / "depth" / f"depth_{i:04d}_*.exr")
                depth_files = sorted(Path(p) for p in glob_glob(depth_pattern))

                rgba = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
                if rgba is None:
                    rgb_files = [p.name for p in sorted((work_dir / "rgb").glob("*"))]
                    depth_files_dbg = [p.name for p in sorted((work_dir / "depth").glob("*"))]
                    raise FileNotFoundError(
                        f"Missing rendered image: {rgba_path}\n"
                        f"Available rgb files: {rgb_files}\n"
                        f"Available depth files: {depth_files_dbg}\n"
                        f"Blender stdout:\n{proc.stdout}\n"
                        f"Blender stderr:\n{proc.stderr}"
                    )

                if rgba.ndim == 2:
                    rgba = np.stack([rgba, rgba, rgba, np.full_like(rgba, 255)], axis=-1)
                if rgba.shape[-1] == 3:
                    alpha = np.full(rgba.shape[:2], 255, dtype=np.uint8)
                    rgba = np.concatenate([rgba, alpha[..., None]], axis=-1)

                rgb = rgba[..., :3][:, :, ::-1].astype(np.float32) / 255.0
                mask = (rgba[..., 3:4].astype(np.float32) / 255.0).clip(0.0, 1.0)

                if depth_files:
                    depth = self._read_exr_depth(depth_files[-1], height=height, width=width)
                else:
                    depth_files_dbg = [p.name for p in sorted((work_dir / "depth").glob("*"))]
                    warnings.warn(
                        "Missing rendered depth EXR; fallback to zeros. "
                        f"view={i}, pattern={depth_pattern}, available={depth_files_dbg}"
                    )
                    depth = np.zeros((height, width, 1), dtype=np.float32)

                rgb_list.append(np.transpose(rgb, (2, 0, 1)))
                mask_list.append(np.transpose(mask, (2, 0, 1)))
                depth_list.append(np.transpose(depth, (2, 0, 1)))

            rgb_arr = np.stack(rgb_list, axis=0).astype(np.float32)
            depth_arr = np.stack(depth_list, axis=0).astype(np.float32)
            mask_arr = np.stack(mask_list, axis=0).astype(np.float32)
            return rgb_arr, depth_arr, mask_arr

    def _read_exr_depth(self, exr_path: Path, height: int, width: int) -> np.ndarray:
        try:
            import OpenEXR
            import Imath

            exr = OpenEXR.InputFile(str(exr_path))
            dw = exr.header()["dataWindow"]
            w = dw.max.x - dw.min.x + 1
            h = dw.max.y - dw.min.y + 1
            chan = exr.channel("R", Imath.PixelType(Imath.PixelType.FLOAT))
            depth = np.frombuffer(chan, dtype=np.float32).reshape((h, w)).copy()
            exr.close()
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            return depth[..., None]
        except Exception as e:
            warnings.warn(f"Failed to read EXR depth {exr_path}; fallback to zeros. err={e}")
            return np.zeros((height, width, 1), dtype=np.float32)

    def __call__(self, batch, **kwargs):
        source_images = batch["source_images"]
        source_view = batch["source_view"]
        source_Ks = batch["source_Ks"]
        target_view = batch["target_view"]
        target_Ks = batch["target_Ks"]
        target_lighting = batch["target_lighting"]
        source_mask = batch.get("source_mask")

        if source_mask is None:
            source_mask = torch.ones_like(source_images[:, :, :1])

        if self.use_official_infer:
            self._prepare_official_meshes_for_batch(batch)

        B, F, C, H, W = source_images.shape
        rgb_out = torch.zeros((B, F, 3, H, W), dtype=torch.float32)
        depth_out = torch.zeros((B, F, 1, H, W), dtype=torch.float32)
        mask_out = torch.ones((B, F, 1, H, W), dtype=torch.float32)

        for b in range(B):
            sample_idx = int(batch["idx"][b].item()) if "idx" in batch else b

            if self.use_official_infer:
                _, mesh_path, hdr_path = self._sample_paths(b, sample_idx)
                if (not mesh_path.exists()) or (not hdr_path.exists()):
                    mesh_path, hdr_path = self._reconstruct_mesh(
                        batch_idx=b,
                        sample_idx=sample_idx,
                        source_images=source_images[b],
                        source_mask=source_mask[b],
                        source_view=source_view[b],
                        source_Ks=source_Ks[b],
                        target_lighting=target_lighting[b],
                    )
            else:
                mesh_path, hdr_path = self._reconstruct_mesh(
                    batch_idx=b,
                    sample_idx=sample_idx,
                    source_images=source_images[b],
                    source_mask=source_mask[b],
                    source_view=source_view[b],
                    source_Ks=source_Ks[b],
                    target_lighting=target_lighting[b],
                )

            render_view = source_view[b] if self.render_source_for_debug else target_view[b]
            render_Ks = source_Ks[b] if self.render_source_for_debug else target_Ks[b]
            if self.render_source_for_debug and (not self._printed_source_render_debug_hint):
                print("[ReLi3D debug] Rendering with source_view/source_K for sanity check.")
                self._printed_source_render_debug_hint = True

            rgb_np, depth_np, mask_np = self._render_with_blender(
                mesh_path=mesh_path,
                hdr_path=hdr_path,
                target_view=render_view,
                target_Ks=render_Ks,
                height=H,
                width=W,
            )

            rgb_out[b] = torch.from_numpy(rgb_np)
            if depth_np.shape[1:] == (1, H, W):
                depth_out[b] = torch.from_numpy(depth_np)
            else:
                depth_out[b] = torch.zeros((F, 1, H, W), dtype=torch.float32)
            if mask_np.shape[1:] == (1, H, W):
                mask_out[b] = torch.from_numpy(mask_np)

            if self.debug:
                mask_ratio = (mask_np > 0.01).astype(np.float32).mean(axis=(1, 2, 3))
                depth_ratio = ((np.isfinite(depth_np)) & (depth_np > 0.0)).astype(np.float32).mean(axis=(1, 2, 3))
                msg_count = min(4, len(mask_ratio))
                mask_head = ", ".join([f"{x:.3f}" for x in mask_ratio[:msg_count]])
                depth_head = ", ".join([f"{x:.3f}" for x in depth_ratio[:msg_count]])
                print(
                    f"[ReLi3D debug] sample_idx={sample_idx} "
                    f"mask_nonzero_mean={float(mask_ratio.mean()):.4f} "
                    f"depth_valid_mean={float(depth_ratio.mean()):.4f} "
                    f"mask_head=[{mask_head}] depth_head=[{depth_head}]"
                )

        rgb_out = rgb_out.to(device=self.device, dtype=self.dtype)
        depth_out = depth_out.to(device=self.device, dtype=torch.float32)
        mask_out = mask_out.to(device=self.device, dtype=self.dtype)
        return rgb_out, depth_out, mask_out


def glob_glob(pattern: str) -> List[str]:
    import glob

    return glob.glob(pattern)

