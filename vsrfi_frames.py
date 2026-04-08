"""VSRFI Frames Node - Processes IMAGE tensor batches instead of video files"""
import os, sys, torch, math, numpy as np, bisect, gc
from pathlib import Path
from einops import rearrange
import torch.nn.functional as F

# ComfyUI imports
try:
    import folder_paths
    import comfy.utils
    import comfy.model_management
    from comfy.model_management import soft_empty_cache
except ImportError:
    print("Warning: ComfyUI imports not available. Using fallback paths.")
    folder_paths = None
    comfy = None
    def soft_empty_cache(): pass

# Add local paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "flashvsr_src"))
sys.path.insert(0, current_dir)

# Path to comfyui-frame-interpolation for RIFE/FILM support
_cfi_path = os.path.join(current_dir, "..", "comfyui-frame-interpolation")
_cfi_available = os.path.isdir(_cfi_path)

from flashvsr_src import ModelManager, FlashVSRTinyLongPipeline
from flashvsr_src.models.TCDecoder import build_tcdecoder
from flashvsr_src.models.utils import Causal_LQ4x_Proj, clean_vram
from flashvsr_src.models import wan_video_dit
from gimmvfi.generalizable_INR.gimmvfi_r import GIMMVFI_R
from gimmvfi.generalizable_INR.configs import GIMMVFIConfig
from gimmvfi.generalizable_INR.raft import RAFT
from omegaconf import OmegaConf
from safetensors.torch import load_file as load_safetensors
import yaml, argparse

# Import model download function from main node
from vsrfi_stream import download_models_if_needed

class InputPadder:
    def __init__(self, dims, factor=32):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // factor) + 1) * factor - self.ht) % factor
        pad_wd = (((self.wd // factor) + 1) * factor - self.wd) % factor
        self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
    def pad(self, *inputs): return [F.pad(x, self._pad, mode='replicate') for x in inputs]
    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

class VSRFIFramesNode:
    @classmethod
    def INPUT_TYPES(cls):
        vfi_methods = ["GIMM-VFI"]
        if _cfi_available:
            vfi_methods += ["RIFE", "FILM"]
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "Input frames as IMAGE tensor"}),
                "scale": ("INT", {"default": 2, "min": 0, "max": 16, "tooltip": "Spatial upscale factor. 0 = skip upscaling"}),
                "interpolation_factor": ("INT", {"default": 2, "min": 0, "max": 16, "tooltip": "FPS multiplier. Values <2 = skip interpolation"}),
                "vfi_method": (vfi_methods, {"tooltip": "Frame interpolation method. RIFE and FILM require comfyui-frame-interpolation."}),
                "frames_per_chunk": ("INT", {"default": 100, "min": 1, "max": 100000, "tooltip": "Number of frames to process at a time"}),
                "max_tile_kilopixels": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "Used for VSR tiling. Set as high as your VRAM allows. 0 = auto-calculate based on available VRAM."}),
                "max_gimm_kilopixels": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "GIMM-VFI only. Max kilopixels for flow estimation. 0 = auto."}),
                "compatibility_mode": ("BOOLEAN", {"default": False, "tooltip": "Force PyTorch native attention (slower but more compatible with older GPUs)"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "video"

    def _get_device(self):
        """Get the appropriate device using ComfyUI's model management if possible."""
        if comfy is not None:
            return comfy.model_management.get_torch_device()
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def process(self, frames, scale, interpolation_factor, vfi_method, frames_per_chunk, max_tile_kilopixels, max_gimm_kilopixels, compatibility_mode=False):
        # Unload all ComfyUI models (e.g. video generation) and free VRAM before loading our models
        if comfy is not None:
            comfy.model_management.unload_all_models()
        clean_vram()

        # Set attention compatibility mode
        wan_video_dit.set_compatibility_mode(compatibility_mode)
        # Reset attention logging so we log which backend is used for this run
        wan_video_dit.reset_attention_logging()

        # Download models if needed
        download_models_if_needed()

        device = self._get_device()

        N, h, w, c = frames.shape
        print(f"[DEBUG] Original input: {w}x{h}")

        # Step 1: Upscale all frames spatially (skip if scale == 0)
        if scale > 0:
            print(f"[INFO] Processing {N} frames at {w}x{h} -> {w*scale}x{h*scale}")
            pipe = self.load_flashvsr(device)
            upscaled_frames = self.upscale_all_frames(pipe, frames, scale, frames_per_chunk, max_tile_kilopixels, device)
            del pipe
            clean_vram()
        else:
            print(f"[INFO] Skipping upscaling (scale=0), {N} frames at {w}x{h}")
            upscaled_frames = frames

        # Step 2: Frame interpolation (if needed)
        if interpolation_factor > 1:
            print(f"[INFO] Interpolating {upscaled_frames.shape[0]} frames by {interpolation_factor}x using {vfi_method}")
            if vfi_method == "RIFE":
                model = self.load_rife(device)
                final_frames = self.interpolate_rife(model, upscaled_frames, interpolation_factor, device)
            elif vfi_method == "FILM":
                model = self.load_film(device)
                final_frames = self.interpolate_film(model, upscaled_frames, interpolation_factor, device)
            else:
                model = self.load_vfi(device)
                final_frames = self.interpolate_all_frames(model, upscaled_frames, interpolation_factor, max_gimm_kilopixels, device)
            del model
            clean_vram()
        else:
            final_frames = upscaled_frames

        print(f"[INFO] Output: {final_frames.shape[0]} frames at {final_frames.shape[2]}x{final_frames.shape[1]}")
        return (final_frames,)

    def load_flashvsr(self, device):
        # Get model path using ComfyUI's folder structure
        if folder_paths is not None:
            model_path = os.path.join(folder_paths.models_dir, "FlashVSR-v1.1")
        else:
            # Fallback for testing outside ComfyUI
            model_path = os.path.join(os.path.expanduser("~"), "comfyui", "models", "FlashVSR-v1.1")

        if not os.path.exists(model_path):
            raise RuntimeError(f"FlashVSR model not found at {model_path}. Please ensure models are downloaded.")

        dtype = torch.bfloat16

        wan_video_dit.USE_BLOCK_ATTN = False
        mm = ModelManager(torch_dtype=dtype, device="cpu")
        mm.load_models([f"{model_path}/diffusion_pytorch_model_streaming_dmd.safetensors"])

        pipe = FlashVSRTinyLongPipeline.from_model_manager(mm, device=device)
        pipe.TCDecoder = build_tcdecoder([512,256,128,128], device, dtype, 16+768)
        pipe.TCDecoder.load_state_dict(torch.load(f"{model_path}/TCDecoder.ckpt", map_location=device), strict=False)
        pipe.TCDecoder.clean_mem()

        pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(3, 1536, 1).to(device, dtype=dtype)
        pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(f"{model_path}/LQ_proj_in.ckpt", map_location="cpu"), strict=True)

        pipe.to(device, dtype=dtype)
        pipe.enable_vram_management(num_persistent_param_in_dit=None)
        pipe.init_cross_kv(prompt_path=os.path.join(os.path.dirname(__file__), "posi_prompt.pth"))
        pipe.load_models_to_device(["dit", "vae"])
        pipe.offload_model()
        return pipe

    def load_vfi(self, device):
        # Get model path using ComfyUI's folder structure
        if folder_paths is not None:
            model_path = os.path.join(folder_paths.models_dir, "interpolation", "gimm-vfi")
        else:
            # Fallback for testing outside ComfyUI
            model_path = os.path.join(os.path.expanduser("~"), "comfyui", "models", "interpolation", "gimm-vfi")

        if not os.path.exists(model_path):
            raise RuntimeError(f"GIMM-VFI model not found at {model_path}. Please ensure models are downloaded.")

        dtype = torch.bfloat16

        config_path = os.path.join(os.path.dirname(__file__), "configs/gimmvfi/gimmvfi_r_arb.yaml")
        with open(config_path) as f:
            config = OmegaConf.create(yaml.load(f, Loader=yaml.FullLoader))
        config = OmegaConf.merge(GIMMVFIConfig.create(config.arch), config.arch)

        model = GIMMVFI_R(dtype, config)
        raft = RAFT(argparse.Namespace(small=False, mixed_precision=False, alternate_corr=False, dropout=0.0))
        raft.load_state_dict(load_safetensors(f"{model_path}/raft-things_fp32.safetensors"), strict=True)

        model.load_state_dict(load_safetensors(f"{model_path}/gimmvfi_r_arb_lpips_fp32.safetensors"), strict=False)
        model.flow_estimator = raft.to(torch.float32).to(device)
        model = model.eval().to(dtype).to(device)
        model.dtype = dtype
        return model

    def upscale_all_frames(self, pipe, frames, scale, frames_per_chunk, max_tile_kilopixels, device):
        """Upscale all frames spatially in chunks"""
        N, h, w, c = frames.shape
        print(f"[DEBUG] Upscaling {N} frames from {w}x{h} to {w*scale}x{h*scale}")

        # Process in chunks to manage memory
        all_upscaled = []

        # Initialize progress bar
        pbar = None
        if comfy is not None:
            pbar = comfy.utils.ProgressBar(N)

        for chunk_start in range(0, N, frames_per_chunk):
            # Check for cancellation
            if comfy is not None:
                comfy.model_management.throw_exception_if_processing_interrupted()

            chunk_end = min(chunk_start + frames_per_chunk, N)
            chunk = frames[chunk_start:chunk_end]

            print(f"[DEBUG] Processing chunk {chunk_start}-{chunk_end} ({chunk.shape[0]} frames)")

            # Upscale this chunk
            upscaled_chunk = self._upscale_chunk(pipe, chunk, scale, max_tile_kilopixels, device)
            all_upscaled.append(upscaled_chunk)

            if pbar is not None:
                pbar.update(chunk.shape[0])

            clean_vram()

        return torch.cat(all_upscaled, dim=0)

    def _auto_max_input_pixels(self, scale, num_frames, device):
        """Estimate max input tile pixels from available VRAM, scale factor, and frame count."""
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        free_gb = free_bytes / (1024**3)
        
        # Reserve for model weights loaded during inference (DiT + TCDecoder)
        is_compat = getattr(wan_video_dit, 'COMPATIBILITY_MODE', False)
        model_inference_gb = 4.5 if is_compat else 3.5
        
        available_gb = max(0.2, free_gb - model_inference_gb) * 0.8
        
        # Efficiency factor: lower is safer. 
        # Standard attention is hungrier.
        base_factor = 80000 if is_compat else 150000
        
        # Penalty for high frame counts (Temporal Decoder overhead)
        temporal_penalty = min(1.0, 30.0 / max(1, num_frames))
        # Don't let it shrink to nothing, but give it a floor
        temporal_penalty = max(0.4, temporal_penalty)
        
        max_pixels = available_gb * base_factor * temporal_penalty / (scale ** 2.6)
        equivalent_kpx = int(max_pixels * (scale ** 3) / 1000)
        
        mode_str = " (COMPAT MODE)" if is_compat else ""
        print(f"[INFO] Auto tile{mode_str}: max {max_pixels/1000:.0f}k input pixels for scale={scale}, frames={num_frames}, equivalent max_tile_kilopixels={equivalent_kpx} (GPU {total_bytes/(1024**3):.1f}GB total, {free_gb:.1f}GB free, {available_gb:.1f}GB available)")
        return max_pixels

    def _upscale_chunk(self, pipe, frames, scale, max_tile_kilopixels, device):
        """Upscale a chunk of frames"""
        N, h, w, c = frames.shape

        if max_tile_kilopixels == 0:
            max_pixels = self._auto_max_input_pixels(scale, N, device)
        else:
            max_pixels = max_tile_kilopixels * 1000 / (scale ** 3)

        splits_w = splits_h = 1
        while (w / splits_w) * (h / splits_h) > max_pixels:
            if w / splits_w > h / splits_h:
                splits_w += 1
            else:
                splits_h += 1

        if splits_w == 1 and splits_h == 1:
            print(f"[DEBUG] No tiling needed: {w}x{h} = {w*h/1000:.0f}k pixels")
            return self._upscale_full(pipe, frames, scale, device)

        return self._upscale_tiled(pipe, frames, scale, splits_w, splits_h, device)

    def _upscale_full(self, pipe, frames, scale, device):
        """Upscale frames without tiling"""
        N, h, w, _ = frames.shape

        # Pad input to multiple of 128/scale (so output will be multiple of 128)
        pad_multiple = 128 // scale if scale >= 1 else 128
        pad_h = (pad_multiple - h % pad_multiple) % pad_multiple
        pad_w = (pad_multiple - w % pad_multiple) % pad_multiple

        if pad_h > 0 or pad_w > 0:
            frames_padded = torch.nn.functional.pad(
                frames.permute(0, 3, 1, 2),
                (0, pad_w, 0, pad_h),
                mode='replicate'
            ).permute(0, 2, 3, 1)
        else:
            frames_padded = frames

        # Pad to 8n+5 requirement
        num_frames = len(frames_padded)
        target_frames = 21 if num_frames < 21 else ((num_frames - 5 + 7) // 8) * 8 + 5
        add = target_frames - num_frames

        if add > 0:
            padding = frames_padded[-1:].repeat(add, 1, 1, 1)
            frames_padded = torch.cat([frames_padded, padding], dim=0)

        # Prepare frames (using largest_8n1_leq logic)
        num_frames_padded = len(frames_padded) + 4
        num_frames_model = 0 if num_frames_padded < 1 else ((num_frames_padded - 1) // 8) * 8 + 1

        h_padded, w_padded = frames_padded.shape[1], frames_padded.shape[2]
        tw = w_padded * scale
        th = h_padded * scale

        processed = []
        for i in range(num_frames_model):
            # Check for cancellation
            if comfy is not None:
                comfy.model_management.throw_exception_if_processing_interrupted()

            idx = min(i, len(frames_padded) - 1)
            frame = frames_padded[idx].to(device)
            upscaled = torch.nn.functional.interpolate(frame.permute(2,0,1).unsqueeze(0), size=(th, tw), mode='bicubic')
            processed.append((upscaled.squeeze(0).to('cpu').to(torch.bfloat16) * 2.0 - 1.0))

        vid = torch.stack(processed, 0).permute(1,0,2,3).unsqueeze(0)
        output = pipe(prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=420,
                     tiled=False, LQ_video=vid, num_frames=num_frames_model, height=th, width=tw, is_full_block=False,
                     if_buffer=True, topk_ratio=2.0*768*1280/(th*tw), kv_ratio=3.0, local_range=11,
                     color_fix=True, unload_dit=False, force_offload=True)

        result = rearrange(output.squeeze(0), "C F H W -> F H W C")
        result = (result.float() + 1.0) / 2.0

        # Crop back to exact output size (remove model padding)
        out_h, out_w = h * scale, w * scale
        result = result[:N, :out_h, :out_w, :]

        return result

    def _upscale_tiled(self, pipe, frames, scale, splits_w, splits_h, device):
        """Upscale frames with spatial tiling"""
        N, h, w, c = frames.shape
        overlap = 32

        print(f"[DEBUG] Tiling {w}x{h} into {splits_w}x{splits_h} grid with {overlap}px overlap, output: {w*scale}x{h*scale}")

        # Calculate tile regions
        tiles = []
        for row in range(splits_h):
            for col in range(splits_w):
                # Calculate base tile boundaries
                x1 = (w * col) // splits_w
                x2 = (w * (col + 1)) // splits_w
                y1 = (h * row) // splits_h
                y2 = (h * (row + 1)) // splits_h

                # Add overlap
                if col > 0:
                    x1 = max(0, x1 - overlap)
                if col < splits_w - 1:
                    x2 = min(w, x2 + overlap)
                if row > 0:
                    y1 = max(0, y1 - overlap)
                if row < splits_h - 1:
                    y2 = min(h, y2 + overlap)

                tiles.append((x1, y1, x2, y2))

        print(f"[DEBUG] Created {len(tiles)} tiles")

        output_canvas = torch.zeros((N, h * scale, w * scale, c), dtype=torch.float32)
        weight_canvas = torch.zeros_like(output_canvas)

        for i, (x1, y1, x2, y2) in enumerate(tiles):
            # Check for cancellation
            if comfy is not None:
                comfy.model_management.throw_exception_if_processing_interrupted()

            tile_w, tile_h = x2 - x1, y2 - y1
            out_pixels = tile_w * tile_h * scale * scale
            print(f"[DEBUG] Tile {i+1}/{len(tiles)}: {tile_w}x{tile_h} -> {tile_w*scale}x{tile_h*scale} = {out_pixels/1000:.0f}k output pixels at ({x1},{y1})")

            tile_frames = frames[:, y1:y2, x1:x2, :]
            tile_output = self._upscale_full(pipe, tile_frames, scale, device)

            # Create blend mask
            tile_out_h, tile_out_w = tile_output.shape[1], tile_output.shape[2]
            overlap_scaled = overlap * scale

            mask = torch.ones(1, 1, tile_out_h, tile_out_w)

            # Determine which edges need blending
            col = i % splits_w
            row = i // splits_w

            if col > 0:  # Has left neighbor
                blend_w = min(overlap_scaled, tile_out_w)
                mask[:, :, :, :blend_w] *= torch.linspace(0, 1, blend_w).view(1, 1, 1, -1)
            if col < splits_w - 1:  # Has right neighbor
                blend_w = min(overlap_scaled, tile_out_w)
                mask[:, :, :, -blend_w:] *= torch.linspace(1, 0, blend_w).view(1, 1, 1, -1)
            if row > 0:  # Has top neighbor
                blend_h = min(overlap_scaled, tile_out_h)
                mask[:, :, :blend_h, :] *= torch.linspace(0, 1, blend_h).view(1, 1, -1, 1)
            if row < splits_h - 1:  # Has bottom neighbor
                blend_h = min(overlap_scaled, tile_out_h)
                mask[:, :, -blend_h:, :] *= torch.linspace(1, 0, blend_h).view(1, 1, -1, 1)

            mask = mask.permute(0, 2, 3, 1)

            # Place in canvas
            out_y1, out_x1 = y1 * scale, x1 * scale
            out_y2, out_x2 = out_y1 + tile_out_h, out_x1 + tile_out_w

            output_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += tile_output * mask
            weight_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += mask

            clean_vram()

        weight_canvas[weight_canvas == 0] = 1.0
        result = output_canvas / weight_canvas

        print(f"[DEBUG] Tiled output: {result.shape}")

        return result

    def _get_cfi_path(self):
        if not _cfi_available:
            raise RuntimeError(
                "RIFE/FILM require comfyui-frame-interpolation to be installed. "
                "Install from: https://github.com/Fannovel16/ComfyUI-Frame-Interpolation"
            )
        return _cfi_path

    def load_rife(self, device):
        cfi = self._get_cfi_path()
        if cfi not in sys.path:
            sys.path.insert(0, cfi)
        from vfi_models.rife.rife_arch import IFNet
        from vfi_utils import load_file_from_github_release

        model_path = load_file_from_github_release("rife", "rife49.pth")
        model = IFNet(arch_ver="4.7")
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval().to(device)
        return model

    def load_film(self, device):
        cfi = self._get_cfi_path()
        if cfi not in sys.path:
            sys.path.insert(0, cfi)
        from vfi_utils import load_file_from_github_release

        model_path = load_file_from_github_release("film", "film_net_fp32.pt")
        model = torch.jit.load(model_path, map_location="cpu")
        model.eval().to(device)
        return model

    def interpolate_rife(self, model, frames, factor, device):
        """Interpolate frames using RIFE (rife49, ensemble=True)."""
        frames = frames.cpu()
        clean_vram()

        frames_chw = rearrange(frames, "N H W C -> N C H W")
        scale_list = [8, 4, 2, 1]
        result = []

        pbar = comfy.utils.ProgressBar(len(frames) - 1) if comfy is not None else None

        for i in range(len(frames) - 1):
            if comfy is not None:
                comfy.model_management.throw_exception_if_processing_interrupted()

            result.append(frames[i])
            frame_0 = frames_chw[i:i+1].to(device).float()
            frame_1 = frames_chw[i+1:i+2].to(device).float()

            for j in range(1, factor):
                timestep = j / factor
                with torch.no_grad():
                    mid = model(frame_0, frame_1, timestep, scale_list, True, True)
                result.append(mid[0].detach().cpu().permute(1, 2, 0).clamp(0, 1))

            if pbar is not None:
                pbar.update(1)

            if (i + 1) % 10 == 0:
                soft_empty_cache()

        result.append(frames[-1])
        return torch.stack(result)

    def interpolate_film(self, model, frames, factor, device):
        """Interpolate frames using FILM (recursive binary subdivision)."""
        frames = frames.cpu()
        clean_vram()

        frames_chw = rearrange(frames, "N H W C -> N C H W")
        result = []

        pbar = comfy.utils.ProgressBar(len(frames) - 1) if comfy is not None else None

        for i in range(len(frames) - 1):
            if comfy is not None:
                comfy.model_management.throw_exception_if_processing_interrupted()

            frame_0 = frames_chw[i:i+1].to(device).float()
            frame_1 = frames_chw[i+1:i+2].to(device).float()

            mid_frames = self._film_inference(model, frame_0, frame_1, factor - 1)
            # mid_frames includes both endpoints; take all but last to avoid duplicates
            for f in mid_frames[:-1]:
                result.append(f[0].detach().cpu().permute(1, 2, 0).clamp(0, 1))

            if pbar is not None:
                pbar.update(1)

            if (i + 1) % 10 == 0:
                soft_empty_cache()

        result.append(frames[-1])
        return torch.stack(result)

    @staticmethod
    def _film_inference(model, img_batch_1, img_batch_2, inter_frames):
        """Recursive binary subdivision interpolation for FILM."""
        results = [img_batch_1, img_batch_2]
        idxes = [0, inter_frames + 1]
        remains = list(range(1, inter_frames + 1))
        splits = torch.linspace(0, 1, inter_frames + 2)

        for _ in range(len(remains)):
            starts = splits[idxes[:-1]]
            ends = splits[idxes[1:]]
            distances = ((splits[None, remains] - starts[:, None]) / (ends[:, None] - starts[:, None]) - .5).abs()
            matrix = torch.argmin(distances).item()
            start_i, step = np.unravel_index(matrix, distances.shape)
            end_i = start_i + 1

            x0 = results[start_i]
            x1 = results[end_i]
            dt = x0.new_full((1, 1), (splits[remains[step]] - splits[idxes[start_i]])) / (splits[idxes[end_i]] - splits[idxes[start_i]])

            with torch.no_grad():
                prediction = model(x0, x1, dt)
            insert_position = bisect.bisect_left(idxes, remains[step])
            idxes.insert(insert_position, remains[step])
            results.insert(insert_position, prediction.clamp(0, 1).float())
            del remains[step]

        return results

    def interpolate_all_frames(self, vfi, frames, factor, max_gimm_kilopixels, device):
        """Interpolate frames to increase frame count"""
        # Move frames to CPU to free GPU memory for VFI computation
        frames = frames.cpu()
        clean_vram()

        # Auto-calculate ds_factor for flow estimation downscaling.
        # ds_factor must produce dimensions divisible by 8 (required by RAFT / build_coord).
        # Padded dims are multiples of 32; valid ds_factors are multiples of 1/(4*gcd(h/32, w/32)).
        h, w = frames.shape[1], frames.shape[2]
        pad_factor = 32
        padded_h = h + (pad_factor - h % pad_factor) % pad_factor
        padded_w = w + (pad_factor - w % pad_factor) % pad_factor
        padded_pixels = padded_h * padded_w
        if max_gimm_kilopixels == 0:
            max_pixels = 1000000
        else:
            max_pixels = max_gimm_kilopixels * 1000
        if padded_pixels > max_pixels:
            raw_ds = (max_pixels / padded_pixels) ** 0.5
            a, b = padded_h // 32, padded_w // 32
            ds_step = 1.0 / (4 * math.gcd(a, b))
            ds_factor = max(ds_step, round(raw_ds / ds_step) * ds_step)
            result_h = int(padded_h * ds_factor)
            result_w = int(padded_w * ds_factor)
            print(f"[DEBUG] VFI ds_factor: {ds_factor:.4f} ({w}x{h} padded to {padded_w}x{padded_h}, downscaled to {result_w}x{result_h} = {result_h*result_w/1000:.0f}k pixels)")
        else:
            ds_factor = None
            print(f"[DEBUG] VFI ds_factor: None ({w}x{h} = {h*w/1000:.0f}k pixels, no downscale)")

        # frames: (N, H, W, C) in range [0, 1]
        frames_chw = frames.permute(0, 3, 1, 2)
        result = []

        # Initialize progress bar
        pbar = None
        if comfy is not None:
            pbar = comfy.utils.ProgressBar(len(frames) - 1)

        for i in range(len(frames) - 1):
            # Check for cancellation
            if comfy is not None:
                comfy.model_management.throw_exception_if_processing_interrupted()

            I0, I2 = frames_chw[i].unsqueeze(0), frames_chw[i+1].unsqueeze(0)
            result.append(frames[i])

            padder = InputPadder(I0.shape, 32)
            I0_pad, I2_pad = padder.pad(I0, I2)
            xs = torch.cat((I0_pad.unsqueeze(2), I2_pad.unsqueeze(2)), dim=2).to(device, dtype=vfi.dtype)

            upsample_ratio = ds_factor if ds_factor is not None else 1.0
            coord_inputs = [(vfi.sample_coord_input(1, xs.shape[-2:], [j/factor], device=xs.device, upsample_ratio=upsample_ratio), None)
                           for j in range(1, factor)]
            timesteps = [j/factor * torch.ones(1, device=xs.device) for j in range(1, factor)]

            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=vfi.dtype):
                outputs = vfi(xs, coord_inputs, t=timesteps, ds_factor=ds_factor)

            for im in outputs["imgt_pred"]:
                result.append(padder.unpad(im)[0].detach().cpu().permute(1,2,0).clamp(0,1))

            if pbar is not None:
                pbar.update(1)

        result.append(frames[-1])
        return torch.stack(result)

NODE_CLASS_MAPPINGS = {"VSRFIFrames": VSRFIFramesNode}
NODE_DISPLAY_NAME_MAPPINGS = {"VSRFIFrames": "VSRFI (Frames)"}
