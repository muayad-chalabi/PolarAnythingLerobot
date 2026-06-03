import argparse
import math
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionControlNetPipeline, UNet2DConditionModel, ControlNetModel
from transformers import PretrainedConfig

from model.PolarControlnet import PolarControl
from model.utils import load_params, remove_module_prefix

class PolarControlTest(ControlNetModel):
    def __init__(self, unet):
        super().__init__(cross_attention_dim=768)
        self.controlnet = PolarControl(PretrainedConfig())
        load_params(self.controlnet, unet)

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        controlnet_cond,
        conditioning_scale=1.0,
        class_labels=None,
        timestep_cond=None,
        attention_mask=None,
        cross_attention_kwargs=None,
        return_dict=True,
        guess_mode=None,
    ):
        timestep = timestep.reshape(1)
        out_down, out_mid = self.controlnet(
            out_vae_noise=sample,
            noise_step=timestep,
            out_encoder=encoder_hidden_states,
            condition=controlnet_cond
        )
        if return_dict:
            return {"down_block_res_samples": out_down, "mid_block_res_sample": out_mid}
        return out_down, out_mid

def prepare_rgb_frame(frame, target_size=None):
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if target_size is None:
        h, w, _ = frame.shape
        h, w = (h // 8) * 8, (w // 8) * 8
        target_size = (w, h)
    frame = cv2.resize(frame, target_size)
    return frame, target_size

def normalize_to_unit(image):
    if image.dtype == np.uint8:
        scale = 255.0
    elif image.dtype == np.uint16:
        scale = 65535.0
    else:
        max_val = float(np.max(image))
        scale = max_val if max_val > 1.0 else 1.0
    image = image.astype(np.float32) / scale
    return np.clip(image, 0.0, 1.0)

def preprocess_frame(frame, target_size=None):
    rgb_frame, target_size = prepare_rgb_frame(frame, target_size)
    frame = normalize_to_unit(rgb_frame)
    tensor_frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()
    return tensor_frame, target_size, rgb_frame

def output_to_bgr_uint8(output):
    output = np.clip(output, 0, 1)
    output = (output * 255).round().astype(np.uint8)
    return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

def save_output(output, save_path):
    output = np.clip(output, 0, 1)
    ext = os.path.splitext(save_path)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        output = (output * 255).round().astype(np.uint8)
    else:
        output = (output * 65535).round().astype(np.uint16)
    cv2.imwrite(save_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

def format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"

def count_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        count += 1
    cap.release()
    return count

def decode_latents(pipeline, latents, device, decode_on_cpu):
    if decode_on_cpu:
        latents = latents.detach().to("cpu")
        latents = latents / pipeline.vae.config.scaling_factor
        image = pipeline.vae.decode(latents, return_dict=False)[0]
    else:
        latents = latents / pipeline.vae.config.scaling_factor
        if device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                image = pipeline.vae.decode(latents, return_dict=False)[0]
        else:
            image = pipeline.vae.decode(latents, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    return image

def infer_frame(pipeline, prompt, tensor_frame, h, w, steps, device, generator, latents, guidance_scale, decode_on_cpu):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        result = pipeline(
            prompt,
            tensor_frame,
            height=h,
            width=w,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            latents=latents,
            output_type="latent"
        )
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    latents_out = result.images if hasattr(result, "images") else result[0]
    image = decode_latents(pipeline, latents_out, device, decode_on_cpu)[0]
    return image, latents_out, elapsed

def prepare_initial_latents(pipeline, base_latent, generator, device, height, width, dtype, timesteps, sigma):
    if base_latent is None:
        latents = pipeline.prepare_latents(
            1,
            pipeline.unet.config.in_channels,
            height,
            width,
            dtype,
            device,
            generator,
            latents=None,
        )
        return latents

    latents = base_latent.to(device)
    if sigma > 0:
        noise = torch.randn(latents.shape, generator=generator, device=device, dtype=dtype)
        latents = latents + sigma * noise
    noise = torch.randn(latents.shape, generator=generator, device=device, dtype=dtype)
    latents = pipeline.scheduler.add_noise(latents, noise, timesteps[0])
    return latents

def log_latent_stats(latents, frame_idx, tag):
    latents_cpu = latents.detach().float().cpu()
    stats = (
        float(latents_cpu.mean()),
        float(latents_cpu.std()),
        float(latents_cpu.min()),
        float(latents_cpu.max()),
    )
    print(
        f"[frame {frame_idx:04d}] {tag} latents "
        f"mean={stats[0]:.6f} std={stats[1]:.6f} min={stats[2]:.6f} max={stats[3]:.6f}"
    )

def make_generator(device, seed):
    if seed is None:
        return None
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen

def get_effective_steps(steps, denoise_strength):
    if not (0.0 < denoise_strength <= 1.0):
        raise ValueError("denoise_strength must be in (0, 1].")
    return max(1, int(round(steps * denoise_strength)))

class TemporalConsistencyModule:
    def __init__(
        self,
        mode,
        latent_noise_sigma,
        flow_backend,
        output_blend_alpha,
        polar_smoothing_mode,
    ):
        self.mode = mode
        self.latent_noise_sigma = latent_noise_sigma
        self.flow_backend = flow_backend
        self.output_blend_alpha = output_blend_alpha
        self.polar_smoothing_mode = polar_smoothing_mode
        self._grid_cache = {}

    def uses_latent(self):
        return self.mode in ["latent_reuse", "latent_warp", "combined"]

    def uses_flow(self):
        return self.mode in ["latent_warp", "output_blend", "combined"]

    def uses_output_blend(self):
        return self.mode in ["output_blend", "combined"] or self.polar_smoothing_mode != "none"

    def compute_flow(self, prev_rgb, curr_rgb):
        if self.flow_backend == "none":
            return None
        prev_gray = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        return flow

    def _get_grid(self, height, width, device, dtype):
        key = (height, width, device, dtype)
        if key in self._grid_cache:
            return self._grid_cache[key]
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
            indexing="ij",
        )
        grid = torch.stack([xs, ys], dim=-1)
        self._grid_cache[key] = grid
        return grid

    def warp_latent(self, latent, flow):
        if flow is None:
            return latent
        latent_h, latent_w = latent.shape[-2:]
        flow_h, flow_w = flow.shape[:2]
        flow_latent = cv2.resize(flow, (latent_w, latent_h), interpolation=cv2.INTER_LINEAR)
        flow_latent[..., 0] *= latent_w / float(flow_w)
        flow_latent[..., 1] *= latent_h / float(flow_h)
        flow_t = torch.from_numpy(flow_latent).to(latent.device, dtype=latent.dtype)

        grid = self._get_grid(latent_h, latent_w, latent.device, latent.dtype)
        flow_x = flow_t[..., 0] * (2.0 / max(latent_w - 1, 1))
        flow_y = flow_t[..., 1] * (2.0 / max(latent_h - 1, 1))
        flow_norm = torch.stack([flow_x, flow_y], dim=-1)
        warped = F.grid_sample(
            latent,
            (grid + flow_norm).unsqueeze(0),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return warped

    def warp_output(self, prev_output, flow):
        if flow is None:
            return prev_output
        h, w = prev_output.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)
        warped = cv2.remap(
            prev_output,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        return warped

    def blend_output(self, current_output, prev_output, flow):
        if prev_output is None:
            return current_output
        alpha = self.output_blend_alpha
        warped_prev = self.warp_output(prev_output, flow)
        if self.polar_smoothing_mode == "aolp_dolp":
            sin2_c = current_output[..., 0] * 2.0 - 1.0
            cos2_c = current_output[..., 1] * 2.0 - 1.0
            dolp_c = current_output[..., 2]
            sin2_p = warped_prev[..., 0] * 2.0 - 1.0
            cos2_p = warped_prev[..., 1] * 2.0 - 1.0
            dolp_p = warped_prev[..., 2]

            sin2 = alpha * sin2_c + (1.0 - alpha) * sin2_p
            cos2 = alpha * cos2_c + (1.0 - alpha) * cos2_p
            norm = np.sqrt(sin2 ** 2 + cos2 ** 2) + 1e-8
            sin2 /= norm
            cos2 /= norm
            dolp = alpha * dolp_c + (1.0 - alpha) * dolp_p
            blended = np.stack(((sin2 + 1.0) / 2.0, (cos2 + 1.0) / 2.0, dolp), axis=-1)
            return blended
        if self.polar_smoothing_mode == "stokes":
            return alpha * current_output + (1.0 - alpha) * warped_prev
        return alpha * current_output + (1.0 - alpha) * warped_prev

class TemporalMetrics:
    def __init__(self, output_dir, fps, enable_side_by_side, warp_output_fn):
        self.output_dir = output_dir
        self.fps = fps
        self.enable_side_by_side = enable_side_by_side
        self.warp_output_fn = warp_output_fn
        self.frame_count = 0
        self.flicker_scores = []
        self.mean = None
        self.m2 = None
        self.residual_writer = None
        self.side_by_side_writer = None

    def _ensure_writers(self, shape):
        h, w, _ = shape
        if self.residual_writer is None:
            residual_path = os.path.join(self.output_dir, "temporal_residuals.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.residual_writer = cv2.VideoWriter(residual_path, fourcc, self.fps, (w, h))
        if self.enable_side_by_side and self.side_by_side_writer is None:
            side_path = os.path.join(self.output_dir, "side_by_side.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.side_by_side_writer = cv2.VideoWriter(side_path, fourcc, self.fps, (w * 2, h * 2))

    def update(self, rgb_frame, raw_output, stabilized_output, prev_output, flow):
        self._ensure_writers(stabilized_output.shape)
        self.frame_count += 1
        if self.mean is None:
            self.mean = stabilized_output.astype(np.float32)
            self.m2 = np.zeros_like(self.mean, dtype=np.float32)
        else:
            delta = stabilized_output - self.mean
            self.mean += delta / self.frame_count
            delta2 = stabilized_output - self.mean
            self.m2 += delta * delta2

        heatmap = None
        if prev_output is not None:
            warped_prev = prev_output if flow is None else self.warp_output_fn(prev_output, flow)
            diff = np.abs(stabilized_output - warped_prev)
            flicker = float(diff.mean())
            self.flicker_scores.append(flicker)
            diff_gray = np.clip(diff.mean(axis=2) * 255.0, 0, 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
            self.residual_writer.write(heatmap)

        if self.enable_side_by_side and heatmap is not None:
            rgb_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            raw_bgr = output_to_bgr_uint8(raw_output)
            stabilized_bgr = output_to_bgr_uint8(stabilized_output)
            grid_top = np.concatenate([rgb_bgr, raw_bgr], axis=1)
            grid_bottom = np.concatenate([stabilized_bgr, heatmap], axis=1)
            grid = np.concatenate([grid_top, grid_bottom], axis=0)
            self.side_by_side_writer.write(grid)

    def finalize(self):
        if self.residual_writer:
            self.residual_writer.release()
        if self.side_by_side_writer:
            self.side_by_side_writer.release()
        if self.frame_count > 1:
            variance = self.m2 / max(self.frame_count - 1, 1)
            variance_gray = np.clip(variance.mean(axis=2), 0, 1)
            variance_img = (variance_gray * 255).astype(np.uint8)
            variance_color = cv2.applyColorMap(variance_img, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(self.output_dir, "temporal_variance.png"), variance_color)
        if self.flicker_scores:
            avg_flicker = sum(self.flicker_scores) / len(self.flicker_scores)
            print(f"Flicker score (mean abs diff): {avg_flicker:.6f}")

def main():
    parser = argparse.ArgumentParser(description="Polar ControlNet Video Inference")
    parser.add_argument('--checkpoint', type=str, default='runwayml/stable-diffusion-v1-5',
                        help='Base Stable Diffusion checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default='./model/PA_Final_Model.pth',
                        help='Checkpoint .pth file for model weights')
    parser.add_argument('--input_video', type=str, required=True,
                        help='Path to input video')
    parser.add_argument('--results_folder', type=str, default='./results',
                        help='Folder to save results')
    parser.add_argument('--steps', type=int, default=20, help='Number of denoising steps')
    parser.add_argument('--output_name', type=str, default='',
                        help='Output video name (mp4). Default: <input>_polar.mp4')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Classifier-free guidance scale')
    parser.add_argument('--fixed_seed', type=int, default=1234, help='Fixed seed for deterministic inference (-1 to disable)')
    parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable deterministic CUDA behavior')
    parser.add_argument('--denoise_strength', type=float, default=1.0,
                        help='Denoise strength in (0, 1]. Lower values reduce diffusion steps')
    parser.add_argument('--temporal_mode', type=str, default='none',
                        choices=['none', 'latent_reuse', 'latent_warp', 'output_blend', 'combined'],
                        help='Temporal consistency mode')
    parser.add_argument('--latent_noise_sigma', type=float, default=0.0,
                        help='Extra noise injected into reused latents [0.0, 0.05]')
    parser.add_argument('--flow_backend', type=str, default='farneback',
                        choices=['farneback', 'none'], help='Optical flow backend')
    parser.add_argument('--output_blend_alpha', type=float, default=0.7,
                        help='EMA blending alpha for output stabilization')
    parser.add_argument('--polar_smoothing_mode', type=str, default='none',
                        choices=['none', 'stokes', 'aolp_dolp'],
                        help='Temporal smoothing in polarization space')
    parser.add_argument('--metrics', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable temporal metrics and visualizations')
    parser.add_argument('--metrics_side_by_side', action=argparse.BooleanOptionalAction, default=False,
                        help='Generate side-by-side video when metrics are enabled')
    parser.add_argument('--vae_slicing', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable VAE slicing to reduce memory usage')
    parser.add_argument('--vae_tiling', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable VAE tiling to reduce memory usage')
    parser.add_argument('--vae_on_cpu', action=argparse.BooleanOptionalAction, default=True,
                        help='Keep VAE on CPU to avoid CUDA OOM during decode')
    args = parser.parse_args()

    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    fixed_seed = None if args.fixed_seed < 0 else args.fixed_seed
    if fixed_seed is not None:
        torch.manual_seed(fixed_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(fixed_seed)

    effective_steps = get_effective_steps(args.steps, args.denoise_strength)
    print(f"Denoise strength: {args.denoise_strength:.2f} -> {effective_steps} steps")

    unet = UNet2DConditionModel.from_pretrained(args.checkpoint, subfolder='unet')
    controlnet = PolarControlTest(unet)
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.checkpoint,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None
    )
    pipeline.unet.requires_grad_(False)
    pipeline.controlnet.requires_grad_(False)
    if args.vae_slicing:
        pipeline.enable_vae_slicing()
    if args.vae_tiling:
        pipeline.enable_vae_tiling()

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    pipeline.unet.load_state_dict(remove_module_prefix(checkpoint['unet_state_dict']))
    pipeline.controlnet.controlnet.load_state_dict(remove_module_prefix(checkpoint['controlnet_state_dict']))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = pipeline.to(device)
    if args.vae_on_cpu:
        pipeline.vae.to("cpu")

    temporal_module = TemporalConsistencyModule(
        args.temporal_mode,
        args.latent_noise_sigma,
        args.flow_backend,
        args.output_blend_alpha,
        args.polar_smoothing_mode,
    )

    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {args.input_video}")

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if not fps_in or math.isnan(fps_in) or fps_in <= 0:
        fps_in = 30.0

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError(f"Cannot read first frame from: {args.input_video}")

    tensor_frame, target_size, first_rgb = preprocess_frame(first_frame)
    tensor_frame = tensor_frame.to(device)
    h, w = target_size[1], target_size[0]

    prompt = 'denoised polarized images'
    pipeline.scheduler.set_timesteps(effective_steps, device=device)
    timesteps = pipeline.scheduler.timesteps
    timesteps_preview = [int(t.item()) for t in timesteps[:5]]
    timesteps_tail = [int(t.item()) for t in timesteps[-5:]]
    print(f"Scheduler timesteps (len={len(timesteps)}): {timesteps_preview} ... {timesteps_tail}")

    first_generator = make_generator(device, fixed_seed)
    first_latents = prepare_initial_latents(
        pipeline,
        None,
        first_generator,
        device,
        h,
        w,
        pipeline.unet.dtype,
        timesteps,
        args.latent_noise_sigma,
    )
    log_latent_stats(first_latents, 0, "init")
    first_output, first_latents_out, first_time = infer_frame(
        pipeline,
        prompt,
        tensor_frame,
        h,
        w,
        effective_steps,
        device,
        first_generator,
        first_latents,
        args.guidance_scale,
        args.vae_on_cpu,
    )
    log_latent_stats(first_latents_out, 0, "out")

    os.makedirs(args.results_folder, exist_ok=True)
    round_folder = os.path.join(args.results_folder, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(round_folder)

    first_frame_path = os.path.join(round_folder, 'first_frame.png')
    save_output(first_output, first_frame_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        frame_count = count_frames(args.input_video)
    if frame_count <= 0:
        cap.release()
        raise ValueError("Unable to determine frame count for video.")

    full_time = first_time * frame_count
    target_fps = 10.0
    if fps_in > target_fps:
        reduced_frames = math.ceil(frame_count * (target_fps / fps_in))
    else:
        reduced_frames = frame_count
    reduced_time = first_time * reduced_frames

    print(
        f"First frame inference took {first_time:.2f}s. "
        f"Estimated full video time ({frame_count} frames @ {fps_in:.2f}fps): {format_duration(full_time)}."
    )
    print(
        f"Estimated 10fps run ({reduced_frames} frames): {format_duration(reduced_time)}."
    )
    print("Options:")
    print("1) Cancel (only first frame saved as PNG)")
    print("2) Run full video")
    print("3) Run video at 10fps")

    choice = input("Select option [1/2/3]: ").strip()
    if choice != '2' and choice != '3':
        cap.release()
        print(f"Cancelled. First frame saved to {first_frame_path}")
        return

    video_name = args.output_name.strip() if args.output_name else f"{Path(args.input_video).stem}_polar.mp4"
    output_path = os.path.join(round_folder, video_name)

    if choice == '2':
        output_fps = fps_in
        sample_interval = 1.0
    else:
        if fps_in > target_fps:
            output_fps = target_fps
            sample_interval = fps_in / target_fps
        else:
            output_fps = fps_in
            sample_interval = 1.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, output_fps, (w, h))
    stabilized_first = temporal_module.blend_output(first_output, None, None)
    writer.write(output_to_bgr_uint8(stabilized_first))

    metrics = None
    if args.metrics:
        metrics_dir = os.path.join(round_folder, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        metrics = TemporalMetrics(metrics_dir, output_fps, args.metrics_side_by_side, temporal_module.warp_output)
        metrics.update(first_rgb, first_output, stabilized_first, None, None)

    prev_rgb = first_rgb
    prev_output = stabilized_first
    prev_latents = first_latents_out.detach()
    if temporal_module.uses_latent():
        prev_latents = prev_latents.to("cpu")
    else:
        prev_latents = None

    frame_index = 1
    next_sample = sample_interval
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if choice == '3' and frame_index + 1e-6 < next_sample:
            frame_index += 1
            continue
        if choice == '3':
            next_sample += sample_interval

        tensor_frame, _, curr_rgb = preprocess_frame(frame, target_size=(w, h))
        tensor_frame = tensor_frame.to(device)

        flow = None
        if (temporal_module.uses_flow() or args.metrics) and prev_rgb is not None:
            flow = temporal_module.compute_flow(prev_rgb, curr_rgb)

        base_latent = prev_latents.to(device) if temporal_module.uses_latent() and prev_latents is not None else None
        if base_latent is not None and temporal_module.uses_flow():
            base_latent = temporal_module.warp_latent(base_latent, flow)

        generator = make_generator(device, fixed_seed)
        latents = prepare_initial_latents(
            pipeline,
            base_latent,
            generator,
            device,
            h,
            w,
            pipeline.unet.dtype,
            timesteps,
            args.latent_noise_sigma,
        )
        log_latent_stats(latents, frame_index, "init")

        raw_output, latents_out, _ = infer_frame(
            pipeline,
            prompt,
            tensor_frame,
            h,
            w,
            effective_steps,
            device,
            generator,
            latents,
            args.guidance_scale,
            args.vae_on_cpu,
        )
        log_latent_stats(latents_out, frame_index, "out")

        if temporal_module.uses_output_blend():
            output = temporal_module.blend_output(raw_output, prev_output, flow)
        else:
            output = raw_output

        writer.write(output_to_bgr_uint8(output))
        if metrics:
            metrics.update(curr_rgb, raw_output, output, prev_output, flow)

        prev_rgb = curr_rgb
        prev_output = output
        prev_latents = latents_out.detach()
        if temporal_module.uses_latent():
            prev_latents = prev_latents.to("cpu")
        else:
            prev_latents = None
        frame_index += 1

    cap.release()
    writer.release()
    if metrics:
        metrics.finalize()
    print(f"Results saved to {output_path}")

if __name__ == '__main__':
    main()
