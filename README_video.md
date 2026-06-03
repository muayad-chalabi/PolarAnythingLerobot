# PolarAnything Video Inference & Temporal Consistency

This document details the enhancements and adaptation from single-frame polarimetric image synthesis (`infer.py`) to temporally consistent video-to-video polarimetric synthesis (`infer_vid.py`).

---

## Overview

While `infer.py` successfully synthesizes polarimetric parameters (such as Stokes vectors, Degree of Linear Polarization (DoLP), and Angle of Linear Polarization (AoLP)) from individual RGB images, applying it frame-by-frame to video sequences introduces severe high-frequency temporal flickering. This is a common issue in diffusion models due to independent noise initialization per frame.

`infer_vid.py` introduces a specialized video inference pipeline designed to enforce temporal consistency both in the **latent space** (during diffusion denoising) and in the **polarization parameter space** (during pixel-level post-processing).

---

## Comparison: Image vs. Video Inference

| Feature / Capability | Single Image Inference (`infer.py`) | Video Inference (`infer.py` $\rightarrow$ `infer_vid.py`) |
| :--- | :--- | :--- |
| **Input Source** | Directory of static images | Input Video File (`.mp4`, `.avi`, etc.) |
| **Output Format** | Static output images (PNG, JPEG) | Output Video File (`.mp4`) + Frame-level caching |
| **Temporal Consistency** | None (Independent per-frame) | Optical flow-guided latent warping, latent reuse, and output blending |
| **Polarization Smoothing**| None | Polarization-specific smoothing (AoLP/DoLP vector math) |
| **Memory Optimization** | Base model loading | VAE slicing, VAE tiling, and VAE-on-CPU offloading |
| **Metrics & Diagnostic** | None | Flicker scores, residual heatmap videos, and variance maps |
| **User Control** | Direct batch run | First-frame time estimation & interactive execution modes |

---

## Core Enhancements & Implementations

### 1. Temporal Consistency Module
A dedicated `TemporalConsistencyModule` is introduced to bridge subsequent frames. It supports several methods configured via `--temporal_mode`:

* **`latent_reuse`**: The final output latents of frame $t-1$ are carried over to initialize the starting latents for frame $t$, instead of starting from complete Gaussian noise. A small amount of Gaussian noise (governed by `--latent_noise_sigma`) is added to maintain sample diversity.
* **`latent_warp`**: To prevent ghosting and misalignment in dynamic scenes, dense optical flow is computed between RGB frame $t-1$ and frame $t$ using the Gunnar Farneback algorithm. The latents of frame $t-1$ are warped according to the motion vectors before initializing the diffusion process of frame $t$.
* **`output_blend`**: Computes an Exponential Moving Average (EMA) blend in pixel space between the current frame's output and the warped previous output, scaled by `--output_blend_alpha`.
* **`combined`**: Enforces both `latent_warp` during denoising and `output_blend` during post-processing.

### 2. Polarization-Specific Smoothing
Filtering raw polarimetric values directly in the pixel space (RGB/BGR representation) ruins physical constraints, especially for cyclic values like Angle of Linear Polarization (AoLP). To resolve this, `--polar_smoothing_mode` provides physical-space filtering:
* **`aolp_dolp`**: Decomposes the AoLP ($\theta$) and DoLP ($\rho$) into trigonometric vector representations:
  $$q = \rho \cos(2\theta), \quad u = \rho \sin(2\theta)$$
  The temporal smoothing is applied to the decomposed vectors $q$ and $u$, which are then mapped back to the normalized polarization representation. This prevents interpolation errors across the $\pi$-periodic boundary of linear polarization angles.
* **`stokes`**: Performs smoothing across linear combinations of Stokes vector channels.

### 3. VRAM & Memory Management
Denoising high-resolution videos using Stable Diffusion and ControlNet simultaneously can easily lead to CUDA Out-Of-Memory (OOM) errors. The video pipeline implements three main mitigations:
* **VAE Slicing (`--vae_slicing`)**: Decodes VAE latents in small batches rather than all at once.
* **VAE Tiling (`--vae_tiling`)**: Splits images into overlapping tiles to perform VAE decoding on smaller chunks.
* **VAE on CPU (`--vae_on_cpu`)**: Keeps the heavy VAE decoder on the system RAM (CPU) instead of VRAM, avoiding OOM during the final frame rendering phase.

### 4. Interactive Performance Estimation
Because video diffusion is computationally heavy, the script performs a dry-run inference on the **first frame**. It then outputs:
* Time taken for the first frame.
* Estimated duration for the entire video at the source frame rate.
* Estimated duration if downsampled to a target 10 FPS.

It prompts the user to choose:
1. **Cancel**: Exits and saves only the first frame as a PNG.
2. **Run full video**: Runs inference on all frames.
3. **Run video at 10 FPS**: Sub-samples the video to 10 FPS to save time and compute.

### 5. Diagnostics & Metrics
When `--metrics` is enabled, the pipeline automatically compiles:
* **Flicker Score**: Measures the Mean Absolute Difference (MAD) between the warped previous frame and the current frame.
* **Residual Heatmap Video (`temporal_residuals.mp4`)**: Visualizes high-frequency temporal changes in jet-colormap heatmaps.
* **Side-by-Side Video (`side_by_side.mp4`)**: Compiles a $2\times2$ grid comparing the input RGB video, the raw frame-by-frame synthesis, the stabilized synthesis, and the temporal residuals.
* **Variance Map (`temporal_variance.png`)**: A heatmap summarizing which regions in the scene suffered the most temporal variance across the entire sequence.

---

## How to Run Video Inference

You can run the video pipeline using the provided shell script wrapper:

```bash
./run_infer_vid.sh
```

### CLI Arguments Reference

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--input_video` | *(Required)* | Path to the source video file |
| `--checkpoint_path` | `./model/PA_Final_Model.pth` | Path to the Polar ControlNet weights |
| `--steps` | `20` | Denoising steps per frame |
| `--denoise_strength` | `1.0` | Fractional strength `(0, 1]`. Lower values skip early diffusion steps for speed |
| `--temporal_mode` | `none` | Temporal consistency mode: `none`, `latent_reuse`, `latent_warp`, `output_blend`, `combined` |
| `--latent_noise_sigma` | `0.0` | Extra noise added to reused latents (suggested: `0.01` to `0.05`) |
| `--flow_backend` | `farneback` | Optical flow backend: `farneback`, `none` |
| `--output_blend_alpha` | `0.7` | Blending factor for EMA smoothing (higher = more current frame, lower = more previous frame) |
| `--polar_smoothing_mode`| `none` | Polarization space smoothing: `none`, `aolp_dolp`, `stokes` |
| `--metrics` | `False` | Enable logging flicker score, residual video, and variance map |
| `--metrics_side_by_side`| `False` | Compile the $2\times2$ side-by-side comparison video |
| `--vae_on_cpu` | `True` | Run VAE decoding on CPU to prevent CUDA OOM |
