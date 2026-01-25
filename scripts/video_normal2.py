import os
import sys
import shutil
from pathlib import Path
from tqdm import tqdm
import torch
import random
import diffusers
import numpy as np
from PIL import Image
import imageio.v3 as imageio
import argparse
import torch.multiprocessing as mp
from functools import partial

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from tesseract.config import PathConfig

from tesseract.config import setup_environment
setup_environment()

def process_video(scene_id, data_path, output_root, pipe, latent_common, device, force=False):
    """Process a single video using the provided model.

    Args:
        scene_id: Scene identifier
        data_path: Input data path (where rgb.mp4 is located)
        output_root: Output root path (where normal.mp4 will be saved). If None, use data_path.
        pipe: Model pipeline
        latent_common: Common latent tensor
        device: Torch device
        force: Force overwrite existing files
    """
    scene_path = os.path.join(data_path, scene_id)
    path_in = os.path.join(scene_path, "video", "rgb.mp4")

    # Determine output path
    if output_root is not None:
        # Output to separate disk with same relative structure
        out_scene_path = os.path.join(output_root, scene_id)
        path_out = os.path.join(out_scene_path, "video", "normal.mp4")

        # Move residual file from input directory to output if exists
        residual_path = os.path.join(scene_path, "video", "normal.mp4")
        if os.path.exists(residual_path):
            # Create output directory if needed
            os.makedirs(os.path.dirname(path_out), exist_ok=True)
            print(f"Moving residual file: {residual_path} -> {path_out}")
            shutil.move(residual_path, path_out)
    else:
        path_out = os.path.join(scene_path, "video", "normal.mp4")

    # Check if output file exists and is a valid video (skip if --force is set)
    if os.path.exists(path_out) and not force:
        try:
            # Try to read the file to verify it's a valid video
            test_reader = imageio.imiter(path_out)
            next(test_reader)  # Try to read at least one frame
            return  # File is valid, skip processing
        except Exception:
            # If file exists but is invalid, remove it and continue processing
            os.remove(path_out)

    # Create output directory if needed
    os.makedirs(os.path.dirname(path_out), exist_ok=True)

    # Read frames from MP4 file
    reader = imageio.imiter(path_in)
    frames = [frame for frame in reader]

    last_frame_latent = None
    first_frame_latent = None

    out = []
    with torch.inference_mode():  # Faster than torch.no_grad()
        for frame_id, frame in enumerate(frames):
            frame = Image.fromarray(frame)

            latents = latent_common
            if last_frame_latent is not None:
                latents = 0.9 * latents + 0.1 * last_frame_latent

            depth = pipe(
                frame,
                match_input_resolution=True,
                latents=latents,
                output_latent=True,
                ensemble_size=1,
                num_inference_steps=1,  # LCM model only needs 1 step
            )

            if first_frame_latent is None:
                first_frame_latent = depth.latent

            last_frame_latent = depth.latent

            out.append(pipe.image_processor.visualize_normals(depth.prediction)[0])

    # Save video using imageio.v3
    imageio.imwrite(path_out, out, fps=30)


def process_videos_on_gpu(scene_list, data_path, output_root, device_id, args):
    """Process multiple videos on a specific GPU."""
    # Set device for this process
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)

    # Enable TF32 for A100 (significant speedup with minimal precision loss)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True  # Auto-tune convolution algorithms

    # Initialize model on this device
    print(f"Loading pipeline components on GPU {device_id}...")
    pipe = diffusers.MarigoldNormalsPipeline.from_pretrained(
        "prs-eth/marigold-normals-lcm-v0-1", variant="fp16", torch_dtype=torch.float16
    ).to(device=device)
    pipe.set_progress_bar_config(disable=True)

    # Initialize latent common
    if "bridge" in data_path:
        size = (640, 480)
        latent_common = torch.randn((1, 4, 768 * size[1] // (8 * max(size)), 768 * size[0] // (8 * max(size)))).to(
            device=device, dtype=torch.float16
        )
    elif "fractal" in data_path:
        size = (320, 256)
        latent_common = torch.randn((1, 4, 768 * size[1] // (8 * max(size)) + 1, 768 * size[0] // (8 * max(size)))).to(
            device=device, dtype=torch.float16
        )
    else:
        raise ValueError("Unknown data path")

    # Process all assigned videos using the same model
    for scene_id in tqdm(scene_list, desc=f"Processing videos on GPU {device_id}"):
        process_video(scene_id, data_path, output_root, pipe, latent_common, device, force=args.force)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bridge")
    parser.add_argument("--exp_name", type=str, default="default", help="Experiment name for path config")
    parser.add_argument(
        "--num_gpus", type=int, default=None, help="Number of GPUs to use. If None, uses all available GPUs."
    )
    parser.add_argument(
        "--output-root", type=str, default=None,
        help="Output root path. If specified, outputs go to this path with same relative structure, and residual files in input path are cleaned up."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force overwrite existing output files instead of skipping."
    )
    args = parser.parse_args()

    # seed everything
    seed = 23
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # 使用 PathConfig 获取数据集路径
    path_config = PathConfig(args.exp_name)
    data_path = str(path_config.get_processed_dataset_path(args.dataset))
    scene_list = sorted(os.listdir(data_path), key=lambda x: int(x))

    # Determine number of GPUs to use
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU will be slow.")
        num_gpus = 0
    else:
        num_gpus = args.num_gpus if args.num_gpus is not None else torch.cuda.device_count()
        print(f"Using {num_gpus} GPUs")

    if num_gpus > 1:
        mp.set_start_method("spawn", force=True)

        # Distribute scenes evenly across GPUs
        gpu_scenes = [[] for _ in range(num_gpus)]
        for i, scene_id in enumerate(scene_list):
            gpu_scenes[i % num_gpus].append(scene_id)

        # Use Process instead of Pool to ensure each process runs on its designated GPU
        processes = []
        for gpu_id in range(num_gpus):
            if gpu_scenes[gpu_id]:  # Only start if there are scenes to process
                p = mp.Process(
                    target=process_videos_on_gpu,
                    args=(gpu_scenes[gpu_id], data_path, args.output_root, gpu_id, args)
                )
                p.start()
                processes.append(p)

        # Wait for all processes to complete
        for p in processes:
            p.join()
    else:
        # Single GPU or CPU processing
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        process_videos_on_gpu(scene_list, data_path, args.output_root, 0, args)


if __name__ == "__main__":
    main()
