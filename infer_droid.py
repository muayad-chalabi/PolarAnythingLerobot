import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tensorflow_datasets as tfds
import torch
from diffusers import StableDiffusionControlNetPipeline, UNet2DConditionModel

from infer_vid import (
    PolarControlTest,
    TemporalConsistencyModule,
    TemporalMetrics,
    append_optional_flag,
    encode_prompt_for_device,
    format_duration,
    get_effective_steps,
    make_generator,
    output_to_bgr_uint8,
    prepare_initial_latents,
    resolve_device,
    save_output,
    infer_frame,
    log_latent_stats,
    normalize_to_unit,
)
from model.utils import remove_module_prefix


def tensor_to_numpy(value):
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def scalar_to_json(value):
    value = tensor_to_numpy(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return scalar_to_json(value.item())
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def preprocess_rgb_frame(frame, target_size=None):
    frame = tensor_to_numpy(frame)
    if frame.ndim == 2:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 4:
        rgb_frame = frame[..., :3]
    else:
        rgb_frame = frame

    if target_size is None:
        h, w = rgb_frame.shape[:2]
        h, w = (h // 8) * 8, (w // 8) * 8
        target_size = (w, h)
    rgb_frame = cv2.resize(rgb_frame, target_size)
    normalized = normalize_to_unit(rgb_frame)
    tensor_frame = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float()
    return tensor_frame, target_size, rgb_frame


def load_pipeline(args, device):
    unet = UNet2DConditionModel.from_pretrained(args.checkpoint, subfolder="unet")
    controlnet = PolarControlTest(unet)
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.checkpoint,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
    )
    pipeline.unet.requires_grad_(False)
    pipeline.controlnet.requires_grad_(False)
    if args.vae_slicing:
        pipeline.enable_vae_slicing()
    if args.vae_tiling:
        pipeline.enable_vae_tiling()

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    pipeline.unet.load_state_dict(remove_module_prefix(checkpoint["unet_state_dict"]))
    pipeline.controlnet.controlnet.load_state_dict(remove_module_prefix(checkpoint["controlnet_state_dict"]))

    pipeline = pipeline.to(device)
    if args.vae_on_cpu:
        pipeline.vae.to("cpu")
    return pipeline


def get_episode(dataset, episode_index):
    for episode in dataset.skip(episode_index).take(1):
        return episode
    raise ValueError(f"Could not load episode index {episode_index} from the requested split.")


def episode_metadata_to_dict(episode):
    metadata = {}
    for key, value in episode.get("episode_metadata", {}).items():
        metadata[key] = scalar_to_json(value)
    return metadata


def step_action_to_dict(step):
    action = {"action": scalar_to_json(step["action"])}
    if "action_dict" in step:
        action["action_dict"] = {key: scalar_to_json(value) for key, value in step["action_dict"].items()}
    return action


def step_language_to_dict(step):
    output = {}
    for key in ["language_instruction", "language_instruction_2", "language_instruction_3"]:
        if key in step:
            output[key] = scalar_to_json(step[key])
    return output


def make_episode_output_dir(args, episode_index):
    os.makedirs(args.results_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dirname = f"{timestamp}_episode_{episode_index:06d}_{args.observation}"
    output_dir = os.path.join(args.results_folder, dirname)
    suffix = 1
    while os.path.exists(output_dir):
        output_dir = os.path.join(args.results_folder, f"{dirname}_{suffix}")
        suffix += 1
    os.makedirs(output_dir)
    return output_dir


def write_episode_info(output_dir, episode_index, metadata, language, args, frame_count, output_path):
    info = {
        "episode_index": episode_index,
        "dataset_name": args.dataset_name,
        "data_dir": args.data_dir,
        "split": args.split,
        "observation": args.observation,
        "frame_count": frame_count,
        "output_video": output_path,
        "episode_metadata": metadata,
        "language": language,
    }
    with open(os.path.join(output_dir, "episode_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def maybe_write_action(actions_file, step_index, step):
    if actions_file is None:
        return
    record = {"step_index": step_index}
    record.update(step_action_to_dict(step))
    actions_file.write(json.dumps(record) + "\n")


def iter_selected_steps(episode, observation, frame_stride, max_frames):
    emitted = 0
    for step_index, step in enumerate(episode["steps"]):
        if step_index % frame_stride != 0:
            continue
        if observation not in step["observation"]:
            available = ", ".join(step["observation"].keys())
            raise KeyError(f"Observation '{observation}' is not present. Available observations: {available}")
        yield step_index, step, step["observation"][observation]
        emitted += 1
        if max_frames > 0 and emitted >= max_frames:
            break


def run_episode_inference(args):
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    fixed_seed = None if args.fixed_seed < 0 else args.fixed_seed
    if fixed_seed is not None:
        torch.manual_seed(fixed_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(fixed_seed)

    device = resolve_device(args.device)
    print(f"Using inference device: {device}")
    effective_steps = get_effective_steps(args.steps, args.denoise_strength)
    print(f"Denoise strength: {args.denoise_strength:.2f} -> {effective_steps} steps")

    dataset = tfds.load(args.dataset_name, data_dir=args.data_dir, split=args.split, shuffle_files=False)
    episode = get_episode(dataset, args.episode_index)
    metadata = episode_metadata_to_dict(episode)

    pipeline = load_pipeline(args, device)
    prompt_kwargs = encode_prompt_for_device(pipeline, args.prompt, device, args.guidance_scale)
    pipeline.scheduler.set_timesteps(effective_steps, device=device)
    timesteps = pipeline.scheduler.timesteps

    temporal_module = TemporalConsistencyModule(
        args.temporal_mode,
        args.latent_noise_sigma,
        args.flow_backend,
        args.output_blend_alpha,
        args.polar_smoothing_mode,
    )

    output_dir = make_episode_output_dir(args, args.episode_index)
    output_path = os.path.join(output_dir, f"episode_{args.episode_index:06d}_{args.observation}_polar.mp4")
    first_frame_path = os.path.join(output_dir, "first_frame.png")
    actions_file = None
    if args.save_actions:
        actions_file = open(os.path.join(output_dir, "actions.jsonl"), "w", encoding="utf-8")

    writer = None
    metrics = None
    target_size = None
    prev_rgb = None
    prev_output = None
    prev_latents = None
    language = {}
    frame_count = 0
    start_time = time.perf_counter()

    try:
        for step_index, step, image in iter_selected_steps(episode, args.observation, args.frame_stride, args.max_frames):
            tensor_frame, target_size, curr_rgb = preprocess_rgb_frame(image, target_size)
            tensor_frame = tensor_frame.to(device)
            h, w = target_size[1], target_size[0]

            if frame_count == 0:
                language = step_language_to_dict(step)
                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    args.output_fps,
                    (w, h),
                )
                if args.metrics:
                    metrics_dir = os.path.join(output_dir, "metrics")
                    os.makedirs(metrics_dir, exist_ok=True)
                    metrics = TemporalMetrics(metrics_dir, args.output_fps, args.metrics_side_by_side, temporal_module.warp_output)

            flow = None
            if frame_count > 0 and (temporal_module.uses_flow() or args.metrics) and prev_rgb is not None:
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
            if args.log_latents:
                log_latent_stats(latents, step_index, "init")

            raw_output, latents_out, elapsed = infer_frame(
                pipeline,
                prompt_kwargs,
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
            if args.log_latents:
                log_latent_stats(latents_out, step_index, "out")

            if temporal_module.uses_output_blend():
                output = temporal_module.blend_output(raw_output, prev_output, flow)
            else:
                output = raw_output

            if frame_count == 0:
                save_output(output, first_frame_path)
            writer.write(output_to_bgr_uint8(output))
            if metrics:
                metrics.update(curr_rgb, raw_output, output, prev_output, flow)
            maybe_write_action(actions_file, step_index, step)

            prev_rgb = curr_rgb
            prev_output = output
            prev_latents = latents_out.detach()
            if temporal_module.uses_latent():
                prev_latents = prev_latents.to("cpu")
            else:
                prev_latents = None
            frame_count += 1
            print(f"Episode {args.episode_index} step {step_index}: frame {frame_count} in {elapsed:.2f}s")
    finally:
        if actions_file is not None:
            actions_file.close()
        if writer is not None:
            writer.release()
        if metrics:
            metrics.finalize()

    if frame_count == 0:
        raise ValueError(f"Episode {args.episode_index} produced no frames for observation '{args.observation}'.")

    write_episode_info(output_dir, args.episode_index, metadata, language, args, frame_count, output_path)
    elapsed_total = time.perf_counter() - start_time
    print(f"Processed {frame_count} frame(s) from episode {args.episode_index} in {format_duration(elapsed_total)}")
    print(f"Results saved to {output_path}")


def build_episode_worker_command(args, episode_index):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--checkpoint", args.checkpoint,
        "--checkpoint_path", args.checkpoint_path,
        "--dataset_name", args.dataset_name,
        "--data_dir", args.data_dir,
        "--split", args.split,
        "--episode_index", str(episode_index),
        "--observation", args.observation,
        "--prompt", args.prompt,
        "--results_folder", args.results_folder,
        "--steps", str(args.steps),
        "--guidance_scale", str(args.guidance_scale),
        "--fixed_seed", str(args.fixed_seed),
        "--denoise_strength", str(args.denoise_strength),
        "--temporal_mode", args.temporal_mode,
        "--latent_noise_sigma", str(args.latent_noise_sigma),
        "--flow_backend", args.flow_backend,
        "--output_blend_alpha", str(args.output_blend_alpha),
        "--polar_smoothing_mode", args.polar_smoothing_mode,
        "--device", args.device,
        "--output_fps", str(args.output_fps),
        "--frame_stride", str(args.frame_stride),
        "--max_frames", str(args.max_frames),
    ]
    append_optional_flag(command, args, "deterministic", "deterministic")
    append_optional_flag(command, args, "metrics", "metrics")
    append_optional_flag(command, args, "metrics_side_by_side", "metrics_side_by_side")
    append_optional_flag(command, args, "vae_slicing", "vae_slicing")
    append_optional_flag(command, args, "vae_tiling", "vae_tiling")
    append_optional_flag(command, args, "vae_on_cpu", "vae_on_cpu")
    append_optional_flag(command, args, "save_actions", "save_actions")
    append_optional_flag(command, args, "log_latents", "log_latents")
    return command


def get_episode_indices(args):
    if args.episode_indices:
        return args.episode_indices
    if args.max_episodes <= 0:
        raise ValueError("Set --max_episodes to a positive value or pass explicit --episode_indices for batch mode.")
    return list(range(args.skip_episodes, args.skip_episodes + args.max_episodes))


def estimate_memory_per_episode_gb(args):
    if args.memory_per_episode_gb > 0:
        return args.memory_per_episode_gb
    base = 7.0
    if not args.vae_on_cpu:
        base += 1.5
    if not args.vae_slicing:
        base += 0.75
    if args.vae_tiling:
        base -= 0.25
    return max(base, 4.0)

def get_device_memory_info_gb(device_arg):
    device = resolve_device(device_arg)
    if device.type != "cuda":
        return None, None
    index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    total_gb = props.total_memory / (1024 ** 3)
    try:
        free_bytes, _ = torch.cuda.mem_get_info(index)
        free_gb = free_bytes / (1024 ** 3)
    except (AttributeError, RuntimeError):
        free_gb = total_gb
    return free_gb, total_gb

def auto_episode_parallelism(args, episode_count):
    if args.batch_max_parallel > 0:
        return max(1, min(args.batch_max_parallel, episode_count))
    free_gb, total_gb = get_device_memory_info_gb(args.device)
    if total_gb is None:
        return 1
    usable_gb = max(0.0, free_gb * args.gpu_memory_fraction - args.gpu_memory_reserve_gb)
    per_episode_gb = estimate_memory_per_episode_gb(args)
    parallel = max(1, int(usable_gb // per_episode_gb))
    parallel = min(parallel, episode_count)
    print(
        f"Detected {total_gb:.1f} GiB total / {free_gb:.1f} GiB free on {resolve_device(args.device)}; "
        f"using {usable_gb:.1f} GiB after reserve/fraction and estimating "
        f"{per_episode_gb:.1f} GiB per episode worker -> {parallel} parallel episode(s)."
    )
    return parallel

def run_episode_batch(args):
    episode_indices = get_episode_indices(args)
    max_parallel = auto_episode_parallelism(args, len(episode_indices))
    os.makedirs(args.results_folder, exist_ok=True)
    print(f"Batching {len(episode_indices)} DROID episode(s) with up to {max_parallel} concurrent worker(s).")

    pending = list(episode_indices)
    running = []
    failures = []
    while pending or running:
        while pending and len(running) < max_parallel:
            episode_index = pending.pop(0)
            command = build_episode_worker_command(args, episode_index)
            print(f"Starting DROID episode {episode_index}")
            running.append((episode_index, subprocess.Popen(command)))

        time.sleep(2.0)
        still_running = []
        for episode_index, proc in running:
            code = proc.poll()
            if code is None:
                still_running.append((episode_index, proc))
            elif code != 0:
                failures.append((episode_index, code))
                print(f"FAILED DROID episode {episode_index} with exit code {code}")
            else:
                print(f"Finished DROID episode {episode_index}")
        running = still_running

    if failures:
        failed = ", ".join(f"episode {episode_index} (exit {code})" for episode_index, code in failures)
        raise RuntimeError(f"DROID batch inference failed for: {failed}")


def main():
    parser = argparse.ArgumentParser(description="Polar ControlNet inference for DROID RLDS episodes")
    parser.add_argument("--checkpoint", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Base Stable Diffusion checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default="./model/PA_Final_Model.pth",
                        help="Checkpoint .pth file for model weights")
    parser.add_argument("--dataset_name", type=str, default="droid",
                        help="TFDS dataset name, e.g. droid or droid_100")
    parser.add_argument("--data_dir", type=str, default="gs://gresearch/robotics",
                        help="TFDS data directory or Google Cloud Storage path")
    parser.add_argument("--split", type=str, default="train", help="TFDS split")
    parser.add_argument("--episode_index", type=int, default=-1,
                        help="Single episode index to process. Negative values run batch mode.")
    parser.add_argument("--episode_indices", type=int, nargs="*", default=None,
                        help="Explicit episode indices to process in batch mode")
    parser.add_argument("--skip_episodes", type=int, default=0,
                        help="First episode index when --episode_indices is not set")
    parser.add_argument("--max_episodes", type=int, default=1,
                        help="Number of episodes to process when --episode_indices is not set")
    parser.add_argument("--observation", type=str, default="exterior_image_1_left",
                        choices=["exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"],
                        help="DROID observation image stream to run inference on")
    parser.add_argument("--prompt", type=str, default="denoised polarized images", help="Inference prompt")
    parser.add_argument("--results_folder", type=str, default="./results_droid", help="Folder to save results")
    parser.add_argument("--steps", type=int, default=20, help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--fixed_seed", type=int, default=1234, help="Fixed seed for deterministic inference (-1 to disable)")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable deterministic CUDA behavior")
    parser.add_argument("--denoise_strength", type=float, default=1.0,
                        help="Denoise strength in (0, 1]. Lower values reduce diffusion steps")
    parser.add_argument("--temporal_mode", type=str, default="none",
                        choices=["none", "latent_reuse", "latent_warp", "output_blend", "combined"],
                        help="Temporal consistency mode")
    parser.add_argument("--latent_noise_sigma", type=float, default=0.0,
                        help="Extra noise injected into reused latents [0.0, 0.05]")
    parser.add_argument("--flow_backend", type=str, default="farneback",
                        choices=["farneback", "none"], help="Optical flow backend")
    parser.add_argument("--output_blend_alpha", type=float, default=0.7,
                        help="EMA blending alpha for output stabilization")
    parser.add_argument("--polar_smoothing_mode", type=str, default="none",
                        choices=["none", "stokes", "aolp_dolp"],
                        help="Temporal smoothing in polarization space")
    parser.add_argument("--metrics", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable temporal metrics and visualizations")
    parser.add_argument("--metrics_side_by_side", action=argparse.BooleanOptionalAction, default=False,
                        help="Generate side-by-side video when metrics are enabled")
    parser.add_argument("--vae_slicing", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable VAE slicing to reduce memory usage")
    parser.add_argument("--vae_tiling", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable VAE tiling to reduce memory usage")
    parser.add_argument("--vae_on_cpu", action=argparse.BooleanOptionalAction, default=True,
                        help="Keep VAE on CPU to avoid CUDA OOM during decode")
    parser.add_argument("--device", type=str, default="auto",
                        help="Inference device: auto, cpu, cuda, cuda:0, cuda:1, etc.")
    parser.add_argument("--batch_max_parallel", type=int, default=0,
                        help="Max episodes to run in parallel. 0 auto-detects from GPU memory.")
    parser.add_argument("--gpu_memory_fraction", type=float, default=0.85,
                        help="Fraction of detected free GPU memory available to batch workers.")
    parser.add_argument("--gpu_memory_reserve_gb", type=float, default=2.0,
                        help="GPU memory to leave unused when auto-sizing batch parallelism.")
    parser.add_argument("--memory_per_episode_gb", type=float, default=0.0,
                        help="Expected GPU memory per episode worker. 0 uses a conservative heuristic.")
    parser.add_argument("--output_fps", type=float, default=15.0, help="FPS for output episode videos")
    parser.add_argument("--frame_stride", type=int, default=1, help="Use every Nth DROID step")
    parser.add_argument("--max_frames", type=int, default=0, help="Max frames per episode; 0 processes all frames")
    parser.add_argument("--save_actions", action=argparse.BooleanOptionalAction, default=True,
                        help="Save DROID action/action_dict records to actions.jsonl")
    parser.add_argument("--log_latents", action=argparse.BooleanOptionalAction, default=False,
                        help="Print latent statistics for every processed frame")
    args = parser.parse_args()

    if args.frame_stride <= 0:
        raise ValueError("--frame_stride must be positive.")
    if args.output_fps <= 0:
        raise ValueError("--output_fps must be positive.")

    if args.episode_index >= 0:
        run_episode_inference(args)
    else:
        run_episode_batch(args)


if __name__ == "__main__":
    main()
