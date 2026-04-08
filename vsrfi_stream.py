"""VSRFI - Video Super Resolution + Frame Interpolation Node"""
import os, sys, cv2, torch, math, numpy as np, bisect, gc
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
from einops import rearrange
import subprocess

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

# HuggingFace for model downloading
from huggingface_hub import snapshot_download

# Add local paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "flashvsr_src"))
sys.path.insert(0, current_dir)

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

# Path to comfyui-frame-interpolation for RIFE/FILM support
_cfi_path = os.path.join(current_dir, "..", "comfyui-frame-interpolation")
_cfi_available = os.path.isdir(_cfi_path)

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

def get_unique_filename(filepath):
    """Generate unique filename by appending (1), (2), etc. if file exists."""
    if not os.path.exists(filepath):
        return filepath

    path = Path(filepath)
    directory = path.parent
    stem = path.stem
    extension = path.suffix

    counter = 1
    while True:
        new_name = f"{stem} ({counter}){extension}"
        new_path = directory / new_name
        if not os.path.exists(new_path):
            return str(new_path)
        counter += 1

def download_models_if_needed():
    """Download FlashVSR and GIMM-VFI models if they don't exist."""
    if folder_paths is None:
        print("Warning: folder_paths not available, skipping model download check")
        return

    models_dir = folder_paths.models_dir

    # FlashVSR model
    flashvsr_dir = os.path.join(models_dir, "FlashVSR-v1.1")
    if not os.path.exists(flashvsr_dir):
        print("Downloading FlashVSR-v1.1 model from HuggingFace...")
        snapshot_download(
            repo_id="JunhaoZhuang/FlashVSR-v1.1",
            local_dir=flashvsr_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"FlashVSR-v1.1 downloaded to {flashvsr_dir}")
    else:
        print(f"FlashVSR-v1.1 found at {flashvsr_dir}")

    # GIMM-VFI model
    gimmvfi_dir = os.path.join(models_dir, "interpolation", "gimm-vfi")
    if not os.path.exists(gimmvfi_dir):
        print("Downloading GIMM-VFI models from HuggingFace...")
        os.makedirs(os.path.dirname(gimmvfi_dir), exist_ok=True)
        snapshot_download(
            repo_id="Kijai/GIMM-VFI_safetensors",
            local_dir=gimmvfi_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"GIMM-VFI downloaded to {gimmvfi_dir}")
    else:
        print(f"GIMM-VFI found at {gimmvfi_dir}")

VIDEO_EXTENSIONS = ['webm', 'mp4', 'mkv', 'gif', 'mov', 'avi']

class VSRFINode:
    @classmethod
    def INPUT_TYPES(cls):
        vfi_methods = ["GIMM-VFI"]
        if _cfi_available:
            vfi_methods += ["RIFE", "FILM"]
        return {
            "required": {
                "video_path": ("STRING", {"default": ""}),
                "output_path": ("STRING", {"default": "", "tooltip": "Leave blank to save to comfyui/output/VSRFI/"}),
                "scale": ("INT", {"default": 2, "min": 0, "max": 16, "tooltip": "Spatial upscale factor. 0 = skip upscaling"}),
                "interpolation_factor": ("INT", {"default": 2, "min": 0, "max": 16, "tooltip": "FPS multiplier. Values <2 = skip interpolation"}),
                "vfi_method": (vfi_methods, {"tooltip": "Frame interpolation method. RIFE and FILM require comfyui-frame-interpolation."}),
                "frames_per_chunk": ("INT", {"default": 100, "min": 1, "max": 100000, "tooltip": "Number of frames to process at a time"}),
                "max_tile_kilopixels": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "Used for VSR tiling. Set as high as your VRAM allows. 0 = auto-calculate based on available VRAM."}),
                "max_gimm_kilopixels": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "GIMM-VFI only. Max kilopixels for flow estimation. 0 = auto."}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_NODE = True
    FUNCTION = "process"
    CATEGORY = "video"

    @classmethod
    def _resolve_video_path(cls, video_path):
        """Resolve video path: check as-is first, then ComfyUI input directory."""
        if os.path.exists(video_path):
            return video_path
        if folder_paths is not None:
            candidate = os.path.join(folder_paths.get_input_directory(), video_path)
            if os.path.exists(candidate):
                return candidate
        return video_path

    @classmethod
    def IS_CHANGED(cls, video_path, **kwargs):
        resolved = cls._resolve_video_path(video_path)
        if os.path.exists(resolved):
            return os.path.getmtime(resolved)
        return ""

    @classmethod
    def VALIDATE_INPUTS(cls, video_path, **kwargs):
        if not video_path:
            return "No video path provided"
        resolved = cls._resolve_video_path(video_path)
        if not os.path.exists(resolved):
            return "Invalid video file: {}".format(video_path)
        return True

    def process(self, video_path, output_path, scale, frames_per_chunk, max_tile_kilopixels, max_gimm_kilopixels, interpolation_factor, vfi_method="GIMM-VFI", skip_first_frames=0, frame_load_cap=0, compatibility_mode=False):

        # Unload all ComfyUI models (e.g. video generation) and free VRAM before loading our models
        if comfy is not None:
            comfy.model_management.unload_all_models()
        clean_vram()

        # Set attention compatibility mode
        wan_video_dit.set_compatibility_mode(compatibility_mode)
        # Reset attention logging so we log which backend is used for this run
        wan_video_dit.reset_attention_logging()

        # Resolve video path: check as-is first, then ComfyUI input directory (for uploaded files)
        if not os.path.exists(video_path) and folder_paths is not None:
            candidate = os.path.join(folder_paths.get_input_directory(), video_path)
            if os.path.exists(candidate):
                video_path = candidate
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")

        # Download models if needed
        download_models_if_needed()

        device = self._get_device()

        # Default output directory: comfyui/output/VSRFI/
        default_output_dir = os.path.join(
            folder_paths.get_output_directory() if folder_paths else os.getcwd(), "VSRFI"
        )

        # Determine output path
        if output_path:
            # User provided a path — validate the parent directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.isdir(output_dir):
                print(f"[WARNING] Output directory does not exist: {output_dir} — falling back to {default_output_dir}")
                output_path = os.path.join(default_output_dir, os.path.basename(output_path))
        else:
            # No output path — save to default VSRFI folder with input filename + _VSRFI
            input_name = Path(video_path).stem + "_VSRFI" + Path(video_path).suffix
            output_path = os.path.join(default_output_dir, input_name)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Handle filename conflicts
        output_path = get_unique_filename(output_path)
        print(f"Output will be saved to: {output_path}")

        # Process video
        self.process_video(video_path, output_path, scale, frames_per_chunk, max_tile_kilopixels, max_gimm_kilopixels, interpolation_factor, device, vfi_method, skip_first_frames, frame_load_cap)
        return (output_path,)
    
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
        pipe.TCDecoder.load_state_dict(torch.load(f"{model_path}/TCDecoder.ckpt", map_location=device, weights_only=True), strict=False)
        pipe.TCDecoder.clean_mem()
        
        pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(3, 1536, 1).to(device, dtype=dtype)
        pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(f"{model_path}/LQ_proj_in.ckpt", map_location="cpu", weights_only=True), strict=True)
        
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
    
    def process_video(self, input_path, output_path, scale, frames_per_chunk, max_tile_kilopixels, max_gimm_kilopixels, interp_factor, device, vfi_method="GIMM-VFI", skip_first_frames=0, frame_load_cap=0):
        self._vfi_method = vfi_method
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w_orig, h_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_in_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Apply skip_first_frames
        if skip_first_frames > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, skip_first_frames)
            print(f"[INFO] Skipping first {skip_first_frames} frames")

        total = max(0, total_in_file - skip_first_frames)

        # Apply frame_load_cap (0 = no limit)
        if frame_load_cap > 0:
            total = min(total, frame_load_cap)
            print(f"[INFO] Frame load cap: {frame_load_cap} (will process {total} frames)")

        if total == 0:
            cap.release()
            raise ValueError(f"No frames to process (video has {total_in_file} frames, skip_first_frames={skip_first_frames}, frame_load_cap={frame_load_cap})")

        # Audio start offset for syncing when frames are skipped
        audio_start_time = skip_first_frames / fps if fps > 0 else 0

        print(f"[DEBUG] Original input: {w_orig}x{h_orig}, processing frames {skip_first_frames}-{skip_first_frames + total - 1} of {total_in_file}")

        # Check if input video has audio (no extraction needed)
        has_audio = False
        probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries',
                     'stream=codec_type', '-of', 'default=noprint_wrappers=1:nokey=1', input_path]
        try:
            result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=5)
            has_audio = result.stdout.strip() == 'audio'
            if has_audio:
                print("[INFO] Audio track detected in input video")
            else:
                print("[INFO] No audio track found in input video")
        except FileNotFoundError:
            print("[WARNING] ffprobe not found. Please install ffmpeg and ensure it is on your system PATH.")
            print("[WARNING] Download from https://ffmpeg.org/download.html")
        except Exception as e:
            print(f"[WARNING] Could not probe audio: {e}")

        w, h = w_orig, h_orig

        # Output dimensions
        if scale > 0:
            out_w = w * scale
            out_h = h * scale
            print(f"[DEBUG] Output: {out_w}x{out_h} (scale={scale})")
        else:
            out_w = w
            out_h = h
            print(f"[DEBUG] Output: {out_w}x{out_h} (no upscaling, VFI only)")

        # Use temporary file for video-only output if we have audio to add later
        video_only_path = output_path if not has_audio else str(Path(output_path).with_suffix('.temp_video.mp4'))

        # Start ffmpeg process for encoding (video only for now)
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', f'{out_w}x{out_h}',
            '-r', str(fps * max(1, interp_factor)),
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '20',
            '-pix_fmt', 'yuv420p',
            video_only_path
        ]

        try:
            process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg not found. Please install ffmpeg and ensure it is on your system PATH.\n"
                "Download from https://ffmpeg.org/download.html"
            )

        # Initialize progress bars
        # ComfyUI progress bar (if available)
        pbar = None
        if comfy is not None:
            pbar = comfy.utils.ProgressBar(total)

        buffer = []
        frames_processed = 0
        was_cancelled = False

        try:
            # Use tqdm for console output
            with tqdm(total=total, desc="Processing frames") as tqdm_bar:
                for _ in range(total):
                    # Check for cancellation at the start of each iteration
                    if comfy is not None:
                        comfy.model_management.throw_exception_if_processing_interrupted()

                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = frame.astype(np.float32) / 255.0
                    buffer.append(frame)

                    if len(buffer) >= frames_per_chunk:
                        # Check for cancellation before processing chunk
                        if comfy is not None:
                            comfy.model_management.throw_exception_if_processing_interrupted()

                        try:
                            chunk = self._process_chunk(buffer, scale, max_tile_kilopixels, max_gimm_kilopixels, interp_factor, device)

                            for frame in chunk:
                                frame_out = (frame.numpy() * 255).clip(0, 255).astype(np.uint8)
                                process.stdin.write(frame_out.tobytes())

                            chunk_size = len(buffer)
                            frames_processed += chunk_size
                            tqdm_bar.update(chunk_size)
                            if pbar is not None:
                                pbar.update_absolute(frames_processed)

                            buffer = []
                            clean_vram()
                        except (comfy.model_management.InterruptProcessingException if comfy else Exception) as e:
                            raise

                # Process remaining frames (if not cancelled)
                if buffer:
                    try:
                        chunk = self._process_chunk(buffer, scale, max_tile_kilopixels, max_gimm_kilopixels, interp_factor, device)

                        for frame in chunk:
                            frame_out = (frame.numpy() * 255).clip(0, 255).astype(np.uint8)
                            process.stdin.write(frame_out.tobytes())

                        chunk_size = len(buffer)
                        frames_processed += chunk_size
                        tqdm_bar.update(chunk_size)
                        if pbar is not None:
                            pbar.update_absolute(frames_processed)
                    except (comfy.model_management.InterruptProcessingException if comfy else Exception) as e:
                        raise

        except (comfy.model_management.InterruptProcessingException if comfy else Exception) as e:
            # Handle cancellation - close ffmpeg gracefully to save partial video
            if comfy is not None and isinstance(e, comfy.model_management.InterruptProcessingException):
                was_cancelled = True
                print(f"\n[INFO] Processing cancelled by user. Saving partial video ({frames_processed}/{total} frames processed)...")
                tqdm_bar.write(f"Cancelled - saving partial video to: {output_path}")
            else:
                raise  # Re-raise if it's not a cancellation

        finally:
            # Always cleanup resources
            cap.release()

            # Close ffmpeg stdin to signal end of input
            if process.stdin:
                try:
                    process.stdin.close()
                except:
                    pass

            # Wait for ffmpeg to finish encoding
            if process:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

            # Mux audio from original video if it has audio
            if has_audio and os.path.exists(video_only_path):
                try:
                    print("[INFO] Adding audio to output video...")

                    # Mux audio, seeking to match skipped frames
                    mux_cmd = [
                        'ffmpeg', '-y',
                        '-i', video_only_path,
                        '-ss', str(audio_start_time),  # Seek audio to match skipped frames
                        '-i', input_path,  # Use original video's audio
                        '-map', '0:v:0',   # Video from processed file
                        '-map', '1:a:0',   # Audio from original file
                        '-c:v', 'copy',    # Don't re-encode video
                        '-c:a', 'aac',     # Re-encode audio for accurate seeking
                        '-shortest',       # Match duration of shortest stream
                        output_path
                    ]

                    result = subprocess.run(mux_cmd, capture_output=True, timeout=30)

                    if result.returncode == 0 and os.path.exists(output_path):
                        print("[INFO] Audio successfully added to output video")
                        # Clean up temporary video-only file
                        try:
                            os.remove(video_only_path)
                        except:
                            pass
                    else:
                        print(f"[WARNING] Audio muxing failed, keeping video-only file at: {video_only_path}")
                        # If muxing failed, rename video-only file to final output
                        if os.path.exists(video_only_path):
                            try:
                                os.rename(video_only_path, output_path)
                            except:
                                pass

                except (subprocess.TimeoutExpired, Exception) as e:
                    print(f"[WARNING] Could not add audio: {e}, keeping video-only file")
                    # Ensure we have an output file
                    if os.path.exists(video_only_path) and not os.path.exists(output_path):
                        try:
                            os.rename(video_only_path, output_path)
                        except:
                            pass

            if was_cancelled:
                # Re-raise the cancellation exception after saving
                if comfy is not None:
                    print(f"[INFO] Partial video saved successfully.")
                    raise comfy.model_management.InterruptProcessingException()
    
    def _process_chunk(self, buffer, scale, max_tile_kilopixels, max_gimm_kilopixels, interp_factor, device):
        """Process a single chunk: load FlashVSR, upscale, destroy it, then load VFI, interpolate, destroy it."""
        frames = torch.from_numpy(np.stack(buffer))

        # Upscale (skip if scale == 0)
        if scale > 0:
            pipe = self.load_flashvsr(device)
            chunk = self.upscale_chunk(pipe, frames, scale, max_tile_kilopixels, device)
            del pipe
            clean_vram()
        else:
            chunk = frames

        # Interpolate
        if interp_factor > 1:
            vfi_method = getattr(self, '_vfi_method', 'GIMM-VFI')
            if vfi_method == "RIFE":
                model = self.load_rife(device)
                chunk = self.interpolate_rife(model, chunk, interp_factor, device)
            elif vfi_method == "FILM":
                model = self.load_film(device)
                chunk = self.interpolate_film(model, chunk, interp_factor, device)
            else:
                model = self.load_vfi(device)
                chunk = self.interpolate_chunk(model, chunk, interp_factor, max_gimm_kilopixels, device)
            del model
            clean_vram()

        return chunk

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

    def upscale_chunk(self, pipe, frames, scale, max_tile_kilopixels, device):
        if max_tile_kilopixels == 0:
            max_pixels = self._auto_max_input_pixels(scale, frames.shape[0], device)
        else:
            max_pixels = max_tile_kilopixels * 1000 / (scale ** 3)

        # Calculate optimal splits
        N, h, w, c = frames.shape
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
        N, h, w, _ = frames.shape

        # Pad input to multiple of 128/scale (so output will be multiple of 128)
        # This maintains aspect ratio
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
        # Target dimensions after scaling (will be multiples of 128)
        tw = w_padded * scale
        th = h_padded * scale

        processed = []
        for i in range(num_frames_model):
            idx = min(i, len(frames_padded) - 1)
            frame = frames_padded[idx].to(device)
            # Upscale maintaining aspect ratio
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
            # Check for cancellation before processing each tile
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
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
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

        for i in range(len(frames) - 1):
            if comfy is not None:
                comfy.model_management.throw_exception_if_processing_interrupted()

            frame_0 = frames_chw[i:i+1].to(device).float()
            frame_1 = frames_chw[i+1:i+2].to(device).float()

            mid_frames = self._film_inference(model, frame_0, frame_1, factor - 1)
            for f in mid_frames[:-1]:
                result.append(f[0].detach().cpu().permute(1, 2, 0).clamp(0, 1))

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

    def interpolate_chunk(self, vfi, frames, factor, max_gimm_kilopixels, device):
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

        frames_chw = frames.permute(0,3,1,2)
        result = []

        for i in range(len(frames) - 1):
            # Check for cancellation before processing each frame pair
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

        result.append(frames[-1])
        return torch.stack(result)

NODE_CLASS_MAPPINGS = {"VSRFI": VSRFINode}
NODE_DISPLAY_NAME_MAPPINGS = {"VSRFI": "VSRFI (Stream)"}
