#!/usr/bin/env python3
"""
Overfitting evaluation script for TesserAct.

This script:
1. Loads samples from a data-cache JSONL file
2. Runs inference on the first frame of each sample
3. Compares predicted video with ground truth
4. Generates 2x2 comparison visualization (GT, SOTA, Checkpoint1, Checkpoint2)

Supports:
- SOTA model with caching (avoids re-inference if cached)
- Multiple checkpoint comparison
- 2x2 grid layout for visualization
"""

import os
import gc
import sys
import json
import argparse
from pathlib import Path

# Add project root to Python path for imports
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import cv2
import numpy as np
import imageio
from PIL import Image
from diffusers.utils import export_to_video
from diffusers import CogVideoXDPMScheduler, AutoencoderKLCogVideoX
from transformers import AutoTokenizer, T5EncoderModel

from tesseract.modules.tesseract_pipeline import TesserActImageToDepthNormalVideoPipeline
from tesseract.modules.tesseract_model import TesserActDepthNormal
from tesseract.utils import print_memory, crop_and_resize_frames

torch.set_grad_enabled(False)


# =============================================================================
# Cache utilities for predictions
# =============================================================================


def get_cache_path(cache_dir: str, scene_id: str, model_name: str = None) -> str:
    """Get the cache file path for a scene and model."""
    if model_name:
        return os.path.join(cache_dir, f"{scene_id}_{model_name}_pred.npy")
    return os.path.join(cache_dir, f"{scene_id}_pred.npy")


def load_cached_prediction(cache_dir: str, scene_id: str, model_name: str = None) -> np.ndarray:
    """Load cached prediction if exists, otherwise return None."""
    path = get_cache_path(cache_dir, scene_id, model_name)
    if os.path.exists(path):
        print(f"  Loading cached prediction from {path}")
        return np.load(path)
    return None


def save_cached_prediction(cache_dir: str, scene_id: str, pred_frames: np.ndarray, model_name: str = None):
    """Save prediction to cache."""
    os.makedirs(cache_dir, exist_ok=True)
    path = get_cache_path(cache_dir, scene_id, model_name)
    np.save(path, pred_frames)
    print(f"  Saved prediction to cache: {path}")


def get_checkpoint_path(base_dir: str, step: int) -> str:
    """Construct checkpoint path from base directory and step number."""
    return os.path.join(base_dir, f"checkpoint-{step}", "transformer")


# =============================================================================
# Shared model components loading
# =============================================================================


def load_shared_components(pretrained_model_path: str, weight_dtype: torch.dtype, device: torch.device) -> dict:
    """Load shared model components (tokenizer, text_encoder, vae, scheduler)."""
    print("Loading shared model components...")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
        pretrained_model_path, subfolder="text_encoder", torch_dtype=weight_dtype
    ).to(device)
    vae = AutoencoderKLCogVideoX.from_pretrained(
        pretrained_model_path, subfolder="vae", torch_dtype=weight_dtype
    ).to(device)
    scheduler = CogVideoXDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")

    return {
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "vae": vae,
        "scheduler": scheduler,
    }


def load_transformer(weights_path: str, weight_dtype: torch.dtype, device: torch.device):
    """Load transformer from weights path."""
    print(f"  Loading transformer from {weights_path}...")

    if os.path.exists(weights_path):
        subfolder = None
        load_path = weights_path
    else:
        subfolder = weights_path.split("/")[-1]
        load_path = "/".join(weights_path.split("/")[:-1])

    transformer = (
        TesserActDepthNormal.from_pretrained_modify(load_path, subfolder=subfolder)
        .to(dtype=weight_dtype, device=device)
        .eval()
    )
    transformer.config.in_channels += 16 * 4
    del transformer.patch_embed.pos_embedding
    transformer.patch_embed.use_learned_positional_embeddings = False
    transformer.config.use_learned_positional_embeddings = False

    return transformer


def build_pipeline(
    weights_path: str,
    shared_components: dict,
    weight_dtype: torch.dtype,
    device: torch.device,
    memory_efficient: bool = False,
) -> TesserActImageToDepthNormalVideoPipeline:
    """Build pipeline with given weights and shared components."""
    transformer = load_transformer(weights_path, weight_dtype, device)

    pipe = TesserActImageToDepthNormalVideoPipeline(
        tokenizer=shared_components["tokenizer"],
        text_encoder=shared_components["text_encoder"],
        vae=shared_components["vae"],
        transformer=transformer,
        scheduler=shared_components["scheduler"],
    )

    if memory_efficient:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        pipe.enable_model_cpu_offload()

    return pipe


def load_cache_samples(cache_file: str, num_samples: int = None, sample_indices: list = None):
    """Load samples from data-cache JSONL file."""
    with open(cache_file, "r") as f:
        lines = f.readlines()

    # First line is header
    header = json.loads(lines[0])
    samples = [json.loads(line) for line in lines[1:]]

    print(f"Loaded {len(samples)} samples from {cache_file}")

    # Select samples
    if sample_indices is not None:
        selected = [samples[i] for i in sample_indices if i < len(samples)]
    elif num_samples is not None:
        selected = samples[:num_samples]
    else:
        selected = samples

    return selected, header


def load_ground_truth(sample: dict, height: int, width: int):
    """Load ground truth RGB, depth, and normal videos from sample paths."""
    rgb_path = sample["rgb_path"]
    depth_path = sample["depth_path"]
    normal_path = sample["normal_path"]

    # Load RGB video
    rgb_reader = imageio.get_reader(rgb_path)
    rgb_frames = [frame for frame in rgb_reader]
    rgb_reader.close()
    rgb_frames = np.array(rgb_frames)

    # Load depth
    depth_data = np.load(depth_path)["arr_0"]  # [T, H, W] or [T, H, W, 1]
    if len(depth_data.shape) == 3:
        depth_data = depth_data[..., np.newaxis]

    # Load normal video
    normal_reader = imageio.get_reader(normal_path)
    normal_frames = [frame for frame in normal_reader]
    normal_reader.close()
    normal_frames = np.array(normal_frames)

    # Resize all to target resolution
    rgb_resized = crop_and_resize_frames(rgb_frames, (height, width))
    normal_resized = crop_and_resize_frames(normal_frames, (height, width))

    # Resize depth (handle single channel)
    depth_resized = []
    for d in depth_data:
        d_3ch = np.repeat(d, 3, axis=-1) if d.shape[-1] == 1 else d
        d_resized = crop_and_resize_frames([d_3ch], (height, width))[0]
        depth_resized.append(d_resized[..., :1])  # Keep single channel
    depth_resized = np.array(depth_resized)

    return rgb_resized, depth_resized, normal_resized


def process_gt_for_display(gt_rgb, gt_depth, gt_normal, num_frames: int):
    """Process ground truth data for display (convert to 0-1 float, visualize depth)."""
    # Ensure numpy arrays
    if isinstance(gt_rgb, list):
        gt_rgb = np.array(gt_rgb)
    if isinstance(gt_depth, list):
        gt_depth = np.array(gt_depth)
    if isinstance(gt_normal, list):
        gt_normal = np.array(gt_normal)

    # Truncate to num_frames
    gt_rgb = gt_rgb[:num_frames]
    gt_depth = gt_depth[:num_frames]
    gt_normal = gt_normal[:num_frames]

    # Process depth for visualization
    # Note: Model outputs inverted depth (1-depth), so invert GT to match
    gt_depth_vis = gt_depth.copy().astype(np.float32)
    gt_depth_vis = (gt_depth_vis - gt_depth_vis.min()) / (gt_depth_vis.max() - gt_depth_vis.min() + 1e-8)
    gt_depth_vis = 1 - gt_depth_vis  # Invert to match model output convention
    gt_depth_vis = np.repeat(gt_depth_vis, 3, axis=-1)  # [T, H, W, 3]

    # Convert to 0-1 float
    gt_rgb = gt_rgb.astype(np.float32) / 255.0
    gt_normal = gt_normal.astype(np.float32) / 255.0

    # Concatenate RGB-Depth-Normal horizontally [T, H, W*3, C]
    gt_combined = np.concatenate([gt_rgb, gt_depth_vis, gt_normal], axis=2)
    return gt_combined


def create_2x2_comparison_video(
    gt_rgb: np.ndarray,
    gt_depth: np.ndarray,
    gt_normal: np.ndarray,
    predictions: dict,  # {"SOTA": frames, "ckpt-100": frames, "ckpt-200": frames}
    output_path: str,
    fps: int = 8,
):
    """
    Create 2x2 comparison video.

    Layout:
    | GT           | SOTA         |
    | Checkpoint1  | Checkpoint2  |

    Each cell contains RGB|Depth|Normal horizontally concatenated.
    """
    # Get checkpoint names (sorted by step number)
    ckpt_names = sorted([k for k in predictions if k.startswith("ckpt-")], key=lambda x: int(x.split("-")[1]))
    sota_frames = predictions.get("SOTA")

    # Determine number of frames (use minimum across all)
    all_frame_counts = [len(gt_rgb)]
    if sota_frames is not None:
        all_frame_counts.append(len(sota_frames))
    for name in ckpt_names:
        all_frame_counts.append(len(predictions[name]))
    num_frames = min(all_frame_counts)

    # Process GT
    gt_combined = process_gt_for_display(gt_rgb, gt_depth, gt_normal, num_frames)
    T, H, W_combined, C = gt_combined.shape
    W = W_combined // 3

    # Process predictions (truncate to num_frames)
    def get_pred_frames(name):
        frames = predictions.get(name)
        if frames is None:
            # Return black frames as placeholder
            return np.zeros((num_frames, H, W_combined, C), dtype=np.float32)
        if isinstance(frames, list):
            frames = np.array(frames)
        return frames[:num_frames]

    sota = get_pred_frames("SOTA") if sota_frames is not None else np.zeros((num_frames, H, W_combined, C), dtype=np.float32)
    ckpt1 = get_pred_frames(ckpt_names[0]) if len(ckpt_names) > 0 else np.zeros((num_frames, H, W_combined, C), dtype=np.float32)
    ckpt2 = get_pred_frames(ckpt_names[1]) if len(ckpt_names) > 1 else np.zeros((num_frames, H, W_combined, C), dtype=np.float32)

    # Build labels
    labels = [
        "GT",
        "SOTA" if sota_frames is not None else "",
        ckpt_names[0] if len(ckpt_names) > 0 else "",
        ckpt_names[1] if len(ckpt_names) > 1 else "",
    ]

    # Create 2x2 frames
    comparison_frames = []
    for i in range(num_frames):
        # Row 1: GT | SOTA
        row1 = np.concatenate([gt_combined[i], sota[i]], axis=1)
        # Row 2: ckpt1 | ckpt2
        row2 = np.concatenate([ckpt1[i], ckpt2[i]], axis=1)
        # Combined
        frame = np.concatenate([row1, row2], axis=0)
        comparison_frames.append(frame)

    # Add labels
    labeled_frames = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    for frame in comparison_frames:
        frame_uint8 = (frame * 255).astype(np.uint8)
        # Row 1 labels
        cv2.putText(frame_uint8, labels[0], (10, 25), font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_uint8, labels[1], (W_combined + 10, 25), font, 0.7, (255, 255, 255), 2)
        # Row 2 labels
        cv2.putText(frame_uint8, labels[2], (10, H + 25), font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_uint8, labels[3], (W_combined + 10, H + 25), font, 0.7, (255, 255, 255), 2)
        labeled_frames.append(frame_uint8.astype(np.float32) / 255.0)

    # Save video
    export_to_video(labeled_frames, output_path, fps=fps)
    print(f"Saved 2x2 comparison video to: {output_path}")


def compute_metrics(pred_frames: np.ndarray, gt_rgb: np.ndarray, gt_depth: np.ndarray, gt_normal: np.ndarray):
    """Compute evaluation metrics between prediction and ground truth."""
    # Ensure inputs are numpy arrays
    if isinstance(pred_frames, list):
        pred_frames = np.array(pred_frames)
    if isinstance(gt_rgb, list):
        gt_rgb = np.array(gt_rgb)
    if isinstance(gt_depth, list):
        gt_depth = np.array(gt_depth)
    if isinstance(gt_normal, list):
        gt_normal = np.array(gt_normal)

    T, H, W_combined, C = pred_frames.shape
    W = W_combined // 3

    # Split prediction
    pred_rgb = pred_frames[:, :, :W].astype(np.float32)
    pred_depth = pred_frames[:, :, W : W * 2].astype(np.float32)
    pred_normal = pred_frames[:, :, W * 2 :].astype(np.float32)

    # Match frame counts
    min_frames = min(len(pred_rgb), len(gt_rgb), len(gt_depth), len(gt_normal))

    gt_rgb = gt_rgb[:min_frames].astype(np.float32)
    gt_normal = gt_normal[:min_frames].astype(np.float32)

    # Process ground truth depth
    # Note: Model outputs inverted depth (1-depth), so invert GT to match
    gt_depth_proc = gt_depth[:min_frames].astype(np.float32)
    gt_depth_proc = (gt_depth_proc - gt_depth_proc.min()) / (gt_depth_proc.max() - gt_depth_proc.min() + 1e-8)
    gt_depth_proc = 1 - gt_depth_proc  # Invert to match model output convention
    gt_depth_proc = gt_depth_proc * 255
    gt_depth_proc = np.repeat(gt_depth_proc, 3, axis=-1)

    metrics = {}

    # RGB metrics
    rgb_mse = np.mean((pred_rgb[:min_frames] - gt_rgb) ** 2)
    rgb_psnr = 10 * np.log10(255**2 / (rgb_mse + 1e-8))
    metrics["rgb_mse"] = float(rgb_mse)
    metrics["rgb_psnr"] = float(rgb_psnr)

    # Depth metrics (normalize pred depth for fair comparison)
    pred_depth_gray = pred_depth[:min_frames].mean(axis=-1, keepdims=True)
    pred_depth_norm = (pred_depth_gray - pred_depth_gray.min()) / (pred_depth_gray.max() - pred_depth_gray.min() + 1e-8)
    gt_depth_gray = gt_depth_proc.mean(axis=-1, keepdims=True) / 255.0
    depth_mse = np.mean((pred_depth_norm - gt_depth_gray) ** 2)
    metrics["depth_mse"] = float(depth_mse)

    # Normal metrics
    normal_mse = np.mean((pred_normal[:min_frames] - gt_normal) ** 2)
    normal_psnr = 10 * np.log10(255**2 / (normal_mse + 1e-8))
    metrics["normal_mse"] = float(normal_mse)
    metrics["normal_psnr"] = float(normal_psnr)

    return metrics


def run_inference(
    pipe,
    sample: dict,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    image_guidance_scale: float,
    use_dynamic_cfg: bool,
    num_frames: int,
    weight_dtype: torch.dtype,
    device: torch.device,
):
    """Run inference on a single sample."""
    rgb_path = sample["rgb_path"]
    depth_path = sample["depth_path"]
    normal_path = sample["normal_path"]
    instruction = sample.get("instruction", "")

    # Load first frame of RGB
    cap = cv2.VideoCapture(rgb_path)
    ret, first_frame = cap.read()
    cap.release()
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    first_frame = crop_and_resize_frames([first_frame.astype(np.float32)], (height, width))[0]
    rgb_image = torch.from_numpy(first_frame).to(dtype=weight_dtype, device=device) / 255.0

    # Load first frame of depth
    depth_data = np.load(depth_path)["arr_0"]
    if len(depth_data.shape) == 4:
        depth_first = depth_data[0]  # [H, W, 1]
    else:
        depth_first = depth_data[0, ..., np.newaxis]  # [H, W, 1]

    # Normalize and invert depth
    depth_first = depth_first.astype(np.float32)
    depth_first = (depth_first - depth_first.min()) / (depth_first.max() - depth_first.min() + 1e-8)
    depth_first = 1 - depth_first  # Invert

    depth_first_3ch = np.repeat(depth_first, 3, axis=-1)
    depth_first_3ch = crop_and_resize_frames([depth_first_3ch], (height, width))[0]
    depth_image = torch.from_numpy(depth_first_3ch).to(dtype=weight_dtype, device=device)

    # Load first frame of normal
    normal_reader = imageio.get_reader(normal_path)
    normal_first = next(iter(normal_reader))
    normal_reader.close()
    normal_first = crop_and_resize_frames([normal_first.astype(np.float32)], (height, width))[0]
    normal_image = torch.from_numpy(normal_first).to(dtype=weight_dtype, device=device) / 255.0

    # Combine inputs: [H, W, 9] -> [1, 9, H, W]
    image = torch.cat([rgb_image, depth_image, normal_image], dim=2)
    image = image.permute(2, 0, 1).unsqueeze(0)

    # Run inference
    gc.collect()
    torch.cuda.empty_cache()

    result = pipe(
        image=image,
        prompt=instruction,
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale,
        use_dynamic_cfg=use_dynamic_cfg,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
    )

    return result.frames[0], instruction


def main():
    parser = argparse.ArgumentParser(description="Multi-checkpoint comparison evaluation for TesserAct")

    # Model paths
    parser.add_argument("--pretrained_model_path", type=str, default="THUDM/CogVideoX-5b-I2V")

    # SOTA model (optional, with caching)
    parser.add_argument("--sota_weights_path", type=str, default=None, help="Path to SOTA model weights")

    # Checkpoint comparison
    parser.add_argument("--checkpoint_steps", type=int, nargs=2, default=None, help="Two checkpoint steps to compare (e.g., 100 200)")
    parser.add_argument("--checkpoint_base_dir", type=str, default="output/sft", help="Base directory for checkpoints")

    # Data (simplified: single sample)
    parser.add_argument("--cache_file", type=str, required=True, help="Path to data-cache JSONL file")
    parser.add_argument("--sample_index", type=int, default=0, help="Sample index to evaluate")

    # Inference settings
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--image_guidance_scale", type=float, default=1.5)
    parser.add_argument("--use_dynamic_cfg", action="store_true", default=False)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--memory_efficient", action="store_true", default=False)

    # Output (all results and cache saved under results/overfit_sample_{index}/)
    parser.add_argument("--output_base_dir", type=str, default="./results", help="Base output directory")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Validate arguments
    if args.sota_weights_path is None and args.checkpoint_steps is None:
        parser.error("At least one of --sota_weights_path or --checkpoint_steps must be specified")

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup output directory: results/overfit_sample_{index}/
    output_dir = os.path.join(args.output_base_dir, f"overfit_sample_{args.sample_index}")
    os.makedirs(output_dir, exist_ok=True)

    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load single sample
    samples, header = load_cache_samples(args.cache_file, num_samples=None, sample_indices=[args.sample_index])
    if not samples:
        print(f"Error: Sample index {args.sample_index} not found in cache file")
        return
    sample = samples[0]
    scene_id = sample["scene_id"]

    print(f"\n{'=' * 60}")
    print(f"Sample: {scene_id}")
    print(f"Instruction: {sample.get('instruction', 'N/A')}")
    print(f"{'=' * 60}")

    # Load shared components
    shared_components = load_shared_components(args.pretrained_model_path, weight_dtype, device)

    # Inference parameters
    inference_kwargs = {
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "image_guidance_scale": args.image_guidance_scale,
        "use_dynamic_cfg": args.use_dynamic_cfg,
        "num_frames": args.num_frames,
        "weight_dtype": weight_dtype,
        "device": device,
    }

    # Collect predictions (all cached in output_dir)
    predictions = {}

    # SOTA inference (with caching)
    if args.sota_weights_path:
        print("\n[SOTA Model]")
        cached_pred = load_cached_prediction(output_dir, scene_id, model_name="SOTA")
        if cached_pred is not None:
            predictions["SOTA"] = cached_pred
        else:
            print("  Running SOTA inference...")
            pipe_sota = build_pipeline(args.sota_weights_path, shared_components, weight_dtype, device, args.memory_efficient)
            pred_frames, _ = run_inference(pipe=pipe_sota, sample=sample, **inference_kwargs)
            if isinstance(pred_frames, list):
                pred_frames = np.array(pred_frames)
            predictions["SOTA"] = pred_frames
            save_cached_prediction(output_dir, scene_id, pred_frames, model_name="SOTA")
            # Free memory
            del pipe_sota
            gc.collect()
            torch.cuda.empty_cache()

    # Checkpoint inference (with caching)
    if args.checkpoint_steps:
        for step in args.checkpoint_steps:
            model_name = f"ckpt-{step}"
            print(f"\n[Checkpoint-{step}]")

            # Check cache first
            cached_pred = load_cached_prediction(output_dir, scene_id, model_name=model_name)
            if cached_pred is not None:
                predictions[model_name] = cached_pred
            else:
                ckpt_path = get_checkpoint_path(args.checkpoint_base_dir, step)
                print(f"  Loading from {ckpt_path}...")
                pipe_ckpt = build_pipeline(ckpt_path, shared_components, weight_dtype, device, args.memory_efficient)
                pred_frames, _ = run_inference(pipe=pipe_ckpt, sample=sample, **inference_kwargs)
                if isinstance(pred_frames, list):
                    pred_frames = np.array(pred_frames)
                predictions[model_name] = pred_frames
                save_cached_prediction(output_dir, scene_id, pred_frames, model_name=model_name)
                print_memory(device)
                # Free memory
                del pipe_ckpt
                gc.collect()
                torch.cuda.empty_cache()

    # Load ground truth
    print("\n[Loading Ground Truth]")
    gt_rgb, gt_depth, gt_normal = load_ground_truth(sample, args.height, args.width)

    # Build output filename with checkpoint steps
    ckpt_suffix = ""
    if args.checkpoint_steps:
        ckpt_suffix = f"_ckpt{args.checkpoint_steps[0]}-{args.checkpoint_steps[1]}"
    output_name = f"comparison{ckpt_suffix}"

    # Create 2x2 comparison video
    comparison_output = os.path.join(output_dir, f"{output_name}.mp4")
    create_2x2_comparison_video(gt_rgb, gt_depth, gt_normal, predictions, comparison_output, fps=args.fps)

    # Compute and save metrics for each model
    all_metrics = {"scene_id": scene_id, "instruction": sample.get("instruction", "")}
    for name, pred_frames in predictions.items():
        metrics = compute_metrics(pred_frames, gt_rgb, gt_depth, gt_normal)
        all_metrics[name] = metrics
        print(f"\n{name} Metrics: RGB PSNR={metrics['rgb_psnr']:.2f}, Depth MSE={metrics['depth_mse']:.4f}, Normal PSNR={metrics['normal_psnr']:.2f}")

    # Save metrics
    metrics_file = os.path.join(output_dir, f"{output_name}_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved metrics to: {metrics_file}")

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
