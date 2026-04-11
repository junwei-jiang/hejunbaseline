import torch
import gc
import os
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
from pipeline import BaselinePipeline

# Keep your original imports
from cosmos_predict1.diffusion.inference.inference_forward_renderer import *
from cosmos_predict1.diffusion.inference.inference_inverse_renderer import *
from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.rendering_utils import envmap_vec
from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.utils_env_proj import hdr_mapping

@dataclass
class RendererConfig:
    checkpoint_dir: str = "diffusion_renderer/checkpoints"
    inverse_transformer_dir: str = "Diffusion_Renderer_Inverse_Cosmos_7B"
    forward_transformer_dir: str = "Diffusion_Renderer_Forward_Cosmos_7B"
    seed: int = 0
    resolution: int = 512
    enable_xformers_memory_efficient_attention: bool = False
    revision: Optional[str] = None
    offload_diffusion_transformer: bool = False
    offload_tokenizer: bool = False
    offload_text_encoder_model: bool = False
    offload_guardrail_models: bool = False
    guidance: float = 1.8
    num_steps: int = 15
    height: int = 512
    width: int = 512
    fps: int = 24
    num_video_frames: int = 1
    normalize_normal: bool = False

G_BUFFER_LABEL = ["basecolor", "normal", "depth", "roughness", "metallic"]
GBUFFER_INDEX_MAPPING = {
    'basecolor': 0, 'metallic': 1, 'roughness': 2,
    'normal': 3, 'depth': 4, 'diffuse_albedo': 5, 'specular_albedo': 6,
}
CPU_DEVICES = 'cpu'

class Diffusion_RendererPipeline(BaselinePipeline):
    def __init__(self, device: str = 'cuda', dtype: torch.dtype = torch.float16, **kwargs):
        super().__init__(device, dtype)
        self.config = RendererConfig()
        self.num_video_frames = kwargs.pop("num_video_frames", 1)
        self.t5_embed_dummy = _prepare_dummy_data_i4()
        self.resolution = (self.config.height, self.config.width)
        
        # Lazy-loaded pipeline caches
        self._inverse_pipe = None
        self._forward_pipe = None

    def _load_inverse(self) -> Any:
        """Load inverse pipeline, unloading forward first if needed."""
        # Free forward pipeline VRAM
        if self._forward_pipe is not None:
            # del self._forward_pipe
            self._forward_pipe = self._forward_pipe.to(CPU_DEVICES)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()

        if self._inverse_pipe is not None:
            return self._inverse_pipe.to(self.device)
            
        self._inverse_pipe = DiffusionRendererPipeline(
            checkpoint_dir=self.config.checkpoint_dir,
            checkpoint_name=self.config.inverse_transformer_dir,
            offload_network=self.config.offload_diffusion_transformer,
            offload_tokenizer=self.config.offload_tokenizer,
            offload_text_encoder_model=self.config.offload_text_encoder_model,
            offload_guardrail_models=self.config.offload_guardrail_models,
            guidance=self.config.guidance,
            num_steps=self.config.num_steps,
            height=self.config.height,
            width=self.config.width,
            fps=self.config.fps,
            num_video_frames=self.config.num_video_frames,
            seed=self.config.seed,
        ).to(self.device, self.dtype)
        return self._inverse_pipe

    def _load_forward(self) -> Any:
        """Load forward pipeline, unloading inverse first if needed."""
        # Free inverse pipeline VRAM
        if self._inverse_pipe is not None:
            # del self._inverse_pipe
            # self._inverse_pipe = None
            self._inverse_pipe = self._inverse_pipe.to(CPU_DEVICES)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()

        if self._forward_pipe is not None:
            return self._forward_pipe.to(self.device)
            
        self._forward_pipe = DiffusionRendererPipeline(
            checkpoint_dir=self.config.checkpoint_dir,
            checkpoint_name=self.config.forward_transformer_dir,
            offload_network=self.config.offload_diffusion_transformer,
            offload_tokenizer=self.config.offload_tokenizer,
            offload_text_encoder_model=self.config.offload_text_encoder_model,
            offload_guardrail_models=self.config.offload_guardrail_models,
            guidance=self.config.guidance,
            num_steps=self.config.num_steps,
            height=self.config.height,
            width=self.config.width,
            fps=self.config.fps,
            num_video_frames=self.config.num_video_frames,
            seed=self.config.seed,
        ).to(self.device, self.dtype)
        return self._forward_pipe

    def inverse_process(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Estimate G-buffers (materials) from input images."""
        g_buf = {}
        inverse_pipe = self._load_inverse()
        
        for gbuffer_pass in G_BUFFER_LABEL:
            context_index = GBUFFER_INDEX_MAPPING[gbuffer_pass]
            batch["context_index"].fill_(context_index)

            g_buf[gbuffer_pass] = inverse_pipe.generate_video(
                data_batch=batch,
                normalize_normal=(gbuffer_pass == 'normal' and self.config.normalize_normal),
            )
        return g_buf

    def forward_process(self, g_buf: Dict[str, torch.Tensor], others: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Relight using estimated G-buffers and environment maps."""
        data_batch = {**g_buf, **others}
        
        forward_pipe = self._load_forward()
        return forward_pipe.generate_video(
            data_batch=data_batch,
            seed=self.config.seed,
        )

    def __call__(self, batch: Dict[str, Any]) -> torch.Tensor:
        ori_F = batch["source_images"].size()[1]
        processed = self.batch_preprocess(batch)
        
        # 1. Inverse: Material/G-buffer Estimation
        g_buffer = self.inverse_process(processed)
        
        # 2. Forward: Relighting
        others = {
            'env_ldr': processed['env_ldr'],
            'env_log': processed['env_log'],
            'env_nrm': processed['env_nrm'],
            't5_text_embeddings': processed['t5_text_embeddings'],
            't5_text_mask': processed['t5_text_mask'],
            'fps': processed['fps'],
            'context_index': processed['context_index'],
            'padding_mask': processed['padding_mask'],
            'num_frames': processed['num_frames'],
            'image_size': processed['image_size'],
            'rgb':  processed['rgb'],
        }

        relit_images = self.forward_process(g_buffer, others)
        relit_images = relit_images.permute(0, 2, 1, 3, 4)[:, :ori_F, ...]

        return relit_images

    def batch_preprocess(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Flatten and move tensors to device/dtype, then pad F dim until F % 8 == 1."""
        batch_size = batch["source_images"].shape[0]
        device = self.device
        H, W = self.resolution

        res = {
            "clip_name": [None] * batch_size,
            "chunk_index": [f"{0:04d}"] * batch_size,
            "is_preprocessed": [True] * batch_size,
            "num_frames": torch.full((batch_size,), self.num_video_frames, dtype=torch.float, device=device),
            "fps": torch.full((batch_size,), self.config.fps, dtype=torch.float, device=device),
            "image_size": torch.tensor(self.resolution, device=device).unsqueeze(0).expand(batch_size, -1).contiguous(),
            "context_index": torch.zeros(batch_size, dtype=torch.long, device=device),
            "padding_mask": torch.zeros(batch_size, H, W, device=device),
            "t5_text_embeddings": self.t5_embed_dummy["t5_text_embeddings"].expand(batch_size, -1, -1).contiguous(),
            "t5_text_mask": self.t5_embed_dummy["t5_text_mask"].expand(batch_size, -1).contiguous(),
        }
        
        # 1. Prepare & normalize spatial tensors (B, C, F, H, W)
        res["rgb"] = batch["source_images"].permute(0, 2, 1, 3, 4).to(device, dtype=self.dtype) * 2 - 1
        
        raw_lighting = batch["target_lighting"]
        c2w = batch["source_view"]
        env = process_environment_map_from_tensor(raw_lighting, c2w=c2w)
        res['env_ldr'] = env['env_ldr'].permute(0, 2, 1, 3, 4).to(device, dtype=self.dtype) * 2 - 1
        res['env_log'] = env['env_log'].permute(0, 2, 1, 3, 4).to(device, dtype=self.dtype) * 2 - 1
        res['env_nrm'] = env['env_nrm'].permute(0, 2, 1, 3, 4).to(device, dtype=self.dtype)

        # 2. Pad F dimension until (F % 8 == 1)
        orig_F = res["rgb"].shape[2]
        if orig_F == 1:
            return res
        # target_F = ((orig_F - 1 + 7) // 8) * 8 + 1
        pad_F = (57 - (orig_F % 57)) % 57
        # pad_F = target_F - orig_F

        if pad_F > 0:
            # F.pad expects padding in reverse dimension order: (W_left, W_right, H_left, H_right, F_left, F_right)
            pad_tuple = (0, 0, 0, 0, 0, pad_F)
            res["rgb"] = torch.nn.functional.pad(res["rgb"], pad_tuple, mode='constant', value=0)
            res['env_ldr'] = torch.nn.functional.pad(res['env_ldr'], pad_tuple, mode='constant', value=0)
            res['env_log'] = torch.nn.functional.pad(res['env_log'], pad_tuple, mode='constant', value=0)
            res['env_nrm'] = torch.nn.functional.pad(res['env_nrm'], pad_tuple, mode='constant', value=0)
        
        return res

    def cleanup(self):
        """Explicitly free all pipeline VRAM. Call at end of script if needed."""
        if self._inverse_pipe is not None:
            del self._inverse_pipe
            self._inverse_pipe = None
        if self._forward_pipe is not None:
            del self._forward_pipe
            self._forward_pipe = None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

# Keep your helper functions exactly as they were
def _prepare_dummy_data_i4():
    dummy_text_embedding = torch.zeros(512, 1024)
    dummy_text_mask = torch.zeros(512)
    dummy_text_mask[0] = 1
    return {"t5_text_embeddings": dummy_text_embedding, "t5_text_mask": dummy_text_mask}

def process_environment_map_from_tensor(
    raw_env_data: torch.Tensor,
    c2w: torch.Tensor,
    resolution: Tuple[int, int] = (512, 512),
    log_scale: int = 10000,
    env_strength: float = 1.0,
    device: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    """
    Process raw environment map tensors, expand to F frames, and generate rotated normal maps.
    
    Args:
        raw_env_data Tensor of shape (B, 1, C, H, W) containing HDR lighting data.
        c2w: Camera-to-World poses of shape (B, F, 4, 4).
        resolution: Expected (H, W) of the input tensors.
        log_scale: Log scale factor for HDR mapping.
        env_strength: Multiplier for environment intensity.
        device: Target compute device.

    Returns:
        dict with keys 'env_ldr', 'env_log', 'env_nrm'.
        Shape: (B, F, 3, H, W) in [0, 1].
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw_env_data = raw_env_data.unsqueeze(1)
    B, _, C, H, W = raw_env_data.shape
    F = c2w.shape[1]
    assert c2w.shape[0] == B, "Batch size mismatch between env data and poses."
    assert (H, W) == resolution, f"Resolution mismatch: expected {resolution}, got {(H, W)}"

    # 1. Expand single-frame env to F frames & apply transforms
    env = raw_env_data.to(device, dtype=torch.float32) * env_strength
    env = env.expand(B, F, C, H, W).contiguous()  # Zero-copy broadcast
    env = torch.flip(env, dims=[-1])              # Fixed horizontal flip (Width)

    N = B * F
    env_flat = env.view(N, C, H, W)

    # 2. HDR Mapping (proj format only)
    env_ldr_list, env_log_list = [], []
    for i in range(N):
        env_map = env_flat[i].permute(1, 2, 0)  # (H, W, C)
        mapping_results = hdr_mapping(env_map, log_scale=log_scale)
        env_ldr_list.append(mapping_results['env_ev0'])
        env_log_list.append(mapping_results['env_log'])
        
    env_ldr = torch.stack(env_ldr_list, dim=0)  # (N, H, W, 3)
    env_log = torch.stack(env_log_list, dim=0)

    # 3. Generate & Rotate Environment Normal Map
    poses_flat = c2w.to(device, dtype=torch.float32).view(N, 4, 4)
    R = poses_flat[:, :3, :3]  # [N, 3, 3]

    if F == 1:
        # For single-frame sequences, force identity rotation
        R = torch.eye(3, device=device, dtype=torch.float32).expand(N, 3, 3)
    else:
        # Normalize/Orthonormalize rotation matrices via Gram-Schmidt
        x = torch.nn.functional.normalize(R[..., 0], dim=-1)
        y = torch.nn.functional.normalize(R[..., 1] - (x * R[..., 1]).sum(-1, keepdim=True) * x, dim=-1)
        z = torch.cross(x, y, dim=-1)
        R = torch.stack([x, y, z], dim=-1)  # [N, 3, 3]

    base_vecs = envmap_vec(resolution, device=device)  # (H, W, 3)
    # Rotate direction vectors by camera rotation: (N, H, W, 3)
    rotated_vecs = torch.einsum('nij, hwj -> nhwi', R, base_vecs)
    env_nrm = torch.nn.functional.normalize(rotated_vecs, dim=-1)
    
    # 4. Reshape to (B, F, H, W, 3) -> Permute to (B, F, 3, H, W)
    def to_bfchw(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(B, F, H, W, 3).permute(0, 1, 4, 2, 3)

    return {
        'env_ldr': to_bfchw(env_ldr),
        'env_log': to_bfchw(env_log),
        'env_nrm': to_bfchw(env_nrm)
    }