#!/bin/bash
set -euo pipefail

# ========== DROID / model config ==========
CHECKPOINT="runwayml/stable-diffusion-v1-5"
CHECKPOINT_PATH=./model/PA_Final_Model.pth
DATASET_NAME=droid
DATA_DIR="gs://gresearch/robotics"
SPLIT=train
RESULTS_FOLDER=./results_droid

# Set EPISODE_INDEX to a non-negative value for one episode, or keep -1 for batch mode.
EPISODE_INDEX=-1
# Batch mode: either list explicit indices or use SKIP_EPISODES + MAX_EPISODES.
# Example: EPISODE_INDICES=(0 3 7)
EPISODE_INDICES=()
SKIP_EPISODES=0
MAX_EPISODES=2

# DROID camera stream to process: exterior_image_1_left, exterior_image_2_left, wrist_image_left.
OBSERVATION=exterior_image_1_left
PROMPT="denoised polarized images"
OUTPUT_FPS=15
FRAME_STRIDE=1
MAX_FRAMES=0
SAVE_ACTIONS=true
LOG_LATENTS=false

# ========== Inference config ==========
STEPS=20
GUIDANCE_SCALE=7.5
FIXED_SEED=1234
DETERMINISTIC=true
DENOISE_STRENGTH=0.50
TEMPORAL_MODE=latent_warp
LATENT_NOISE_SIGMA=0.01
FLOW_BACKEND=farneback
OUTPUT_BLEND_ALPHA=0.8
POLAR_SMOOTHING_MODE=none
METRICS=false
METRICS_SIDE_BY_SIDE=false
VAE_SLICING=true
VAE_TILING=false
VAE_ON_CPU=true

# Device selection: auto, cpu, cuda, cuda:0, cuda:1, etc.
DEVICE=auto
# Batch parallelism: 0 auto-detects from free GPU memory. Override if needed.
BATCH_MAX_PARALLEL=0
GPU_MEMORY_FRACTION=0.85
GPU_MEMORY_RESERVE_GB=2.0
# 0 uses the same conservative per-worker heuristic as video batch inference.
MEMORY_PER_EPISODE_GB=0

PY_SCRIPT=infer_droid.py

DETERMINISTIC_ARG=(--no-deterministic)
if [ "$DETERMINISTIC" = true ]; then
  DETERMINISTIC_ARG=(--deterministic)
fi

METRICS_ARG=()
if [ "$METRICS" = true ]; then
  METRICS_ARG=(--metrics)
fi

METRICS_SIDE_BY_SIDE_ARG=()
if [ "$METRICS_SIDE_BY_SIDE" = true ]; then
  METRICS_SIDE_BY_SIDE_ARG=(--metrics_side_by_side)
fi

VAE_SLICING_ARG=(--no-vae_slicing)
if [ "$VAE_SLICING" = true ]; then
  VAE_SLICING_ARG=(--vae_slicing)
fi

VAE_TILING_ARG=()
if [ "$VAE_TILING" = true ]; then
  VAE_TILING_ARG=(--vae_tiling)
fi

VAE_ON_CPU_ARG=(--no-vae_on_cpu)
if [ "$VAE_ON_CPU" = true ]; then
  VAE_ON_CPU_ARG=(--vae_on_cpu)
fi

SAVE_ACTIONS_ARG=(--no-save_actions)
if [ "$SAVE_ACTIONS" = true ]; then
  SAVE_ACTIONS_ARG=(--save_actions)
fi

LOG_LATENTS_ARG=()
if [ "$LOG_LATENTS" = true ]; then
  LOG_LATENTS_ARG=(--log_latents)
fi

EPISODE_ARGS=()
if [ "$EPISODE_INDEX" -ge 0 ]; then
  EPISODE_ARGS=(--episode_index "$EPISODE_INDEX")
elif [ "${#EPISODE_INDICES[@]}" -gt 0 ]; then
  EPISODE_ARGS=(--episode_indices "${EPISODE_INDICES[@]}")
else
  EPISODE_ARGS=(--skip_episodes "$SKIP_EPISODES" --max_episodes "$MAX_EPISODES")
fi

python "$PY_SCRIPT" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --dataset_name "$DATASET_NAME" \
  --data_dir "$DATA_DIR" \
  --split "$SPLIT" \
  "${EPISODE_ARGS[@]}" \
  --observation "$OBSERVATION" \
  --prompt "$PROMPT" \
  --results_folder "$RESULTS_FOLDER" \
  --steps "$STEPS" \
  --guidance_scale "$GUIDANCE_SCALE" \
  --fixed_seed "$FIXED_SEED" \
  "${DETERMINISTIC_ARG[@]}" \
  --denoise_strength "$DENOISE_STRENGTH" \
  --temporal_mode "$TEMPORAL_MODE" \
  --latent_noise_sigma "$LATENT_NOISE_SIGMA" \
  --flow_backend "$FLOW_BACKEND" \
  --output_blend_alpha "$OUTPUT_BLEND_ALPHA" \
  --polar_smoothing_mode "$POLAR_SMOOTHING_MODE" \
  "${METRICS_ARG[@]}" \
  "${METRICS_SIDE_BY_SIDE_ARG[@]}" \
  "${VAE_SLICING_ARG[@]}" \
  "${VAE_TILING_ARG[@]}" \
  "${VAE_ON_CPU_ARG[@]}" \
  "${SAVE_ACTIONS_ARG[@]}" \
  "${LOG_LATENTS_ARG[@]}" \
  --device "$DEVICE" \
  --batch_max_parallel "$BATCH_MAX_PARALLEL" \
  --gpu_memory_fraction "$GPU_MEMORY_FRACTION" \
  --gpu_memory_reserve_gb "$GPU_MEMORY_RESERVE_GB" \
  --memory_per_episode_gb "$MEMORY_PER_EPISODE_GB" \
  --output_fps "$OUTPUT_FPS" \
  --frame_stride "$FRAME_STRIDE" \
  --max_frames "$MAX_FRAMES"
