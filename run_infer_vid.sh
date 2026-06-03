#!/bin/bash

# ========== 配置 ==========
CHECKPOINT="runwayml/stable-diffusion-v1-5"
CHECKPOINT_PATH=./model/PA_Final_Model.pth
INPUT_VIDEO=./data/VAL/demo_vid.mp4
RESULTS_FOLDER=./results
STEPS=20
OUTPUT_NAME=
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

PY_SCRIPT=infer_vid.py

OUTPUT_NAME_ARG=
if [ -n "$OUTPUT_NAME" ]; then
  OUTPUT_NAME_ARG="--output_name $OUTPUT_NAME"
fi

DETERMINISTIC_ARG=--no-deterministic
if [ "$DETERMINISTIC" = true ]; then
  DETERMINISTIC_ARG=--deterministic
fi

METRICS_ARG=
if [ "$METRICS" = true ]; then
  METRICS_ARG="--metrics"
fi

METRICS_SIDE_BY_SIDE_ARG=
if [ "$METRICS_SIDE_BY_SIDE" = true ]; then
  METRICS_SIDE_BY_SIDE_ARG="--metrics_side_by_side"
fi

VAE_SLICING_ARG=--no-vae_slicing
if [ "$VAE_SLICING" = true ]; then
  VAE_SLICING_ARG=--vae_slicing
fi

VAE_TILING_ARG=
if [ "$VAE_TILING" = true ]; then
  VAE_TILING_ARG=--vae_tiling
fi

VAE_ON_CPU_ARG=--no-vae_on_cpu
if [ "$VAE_ON_CPU" = true ]; then
  VAE_ON_CPU_ARG=--vae_on_cpu
fi

# ========== 启动区 ==========
python $PY_SCRIPT \
  --checkpoint $CHECKPOINT \
  --checkpoint_path $CHECKPOINT_PATH \
  --input_video $INPUT_VIDEO \
  --results_folder $RESULTS_FOLDER \
  --steps $STEPS \
  --guidance_scale $GUIDANCE_SCALE \
  --fixed_seed $FIXED_SEED \
  $DETERMINISTIC_ARG \
  --denoise_strength $DENOISE_STRENGTH \
  --temporal_mode $TEMPORAL_MODE \
  --latent_noise_sigma $LATENT_NOISE_SIGMA \
  --flow_backend $FLOW_BACKEND \
  --output_blend_alpha $OUTPUT_BLEND_ALPHA \
  --polar_smoothing_mode $POLAR_SMOOTHING_MODE \
  $METRICS_ARG \
  $METRICS_SIDE_BY_SIDE_ARG \
  $VAE_SLICING_ARG \
  $VAE_TILING_ARG \
  $VAE_ON_CPU_ARG \
  $OUTPUT_NAME_ARG
