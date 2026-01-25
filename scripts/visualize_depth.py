#!/usr/bin/env python3
"""Simple script to visualize depth maps from npz files."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_depth(path: str) -> np.ndarray:
    """Load depth data from npy or npz file.

    Args:
        path: Path to npy or npz file

    Returns:
        Depth array with shape [N, H, W] or [H, W]
    """
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path)
        depth = data["arr_0"]
    else:  # .npy
        depth = np.load(path)

    # Ensure 3D shape [N, H, W]
    if depth.ndim == 2:
        depth = depth[np.newaxis, ...]

    return depth


def visualize_depth(input_path: str, frame_idx: int = 0, cmap: str = "Spectral_r", save_path: str = None):
    """Visualize a depth map from npy/npz file.

    Args:
        input_path: Path to the npy/npz file containing depth data
        frame_idx: Frame index to visualize (default: 0)
        cmap: Matplotlib colormap name (default: Spectral_r)
        save_path: If provided, save the figure to this path instead of showing
    """
    # Load depth data
    depth = load_depth(input_path)  # Shape: [N, H, W]

    print(f"Depth shape: {depth.shape}")
    print(f"Depth range: [{depth.min():.4f}, {depth.max():.4f}]")
    print(f"Visualizing frame {frame_idx}/{depth.shape[0]-1}")

    # Get single frame
    depth_frame = depth[frame_idx]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Original depth (normalized)
    im1 = axes[0].imshow(depth_frame, cmap=cmap)
    axes[0].set_title(f"Depth Map (frame {frame_idx})")
    axes[0].axis("off")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Histogram
    axes[1].hist(depth_frame.flatten(), bins=100, color="steelblue", alpha=0.7)
    axes[1].set_xlabel("Depth Value")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Depth Distribution")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_depth_video(input_path: str, output_path: str, cmap: str = "Spectral_r", fps: int = 30):
    """Create a video visualization of depth maps.

    Args:
        input_path: Path to the npy/npz file containing depth data
        output_path: Output video path (mp4)
        cmap: Matplotlib colormap name
        fps: Frames per second for output video
    """
    import cv2

    # Load depth data
    depth = load_depth(input_path)  # Shape: [N, H, W]

    print(f"Depth shape: {depth.shape}")
    print(f"Creating video with {depth.shape[0]} frames...")

    # Normalize depth to 0-255 for visualization
    depth_min, depth_max = depth.min(), depth.max()
    depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-8)

    # Get colormap
    colormap = plt.get_cmap(cmap)

    # Get frame dimensions
    h, w = depth.shape[1], depth.shape[2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for i in range(depth.shape[0]):
        # Apply colormap
        colored = colormap(depth_norm[i])[:, :, :3]  # Remove alpha channel
        colored = (colored * 255).astype(np.uint8)
        colored = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
        out.write(colored)

    out.release()
    print(f"Saved video to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize depth maps from npz files")
    parser.add_argument("input", type=str, help="Path to depth.npy or depth.npz file")
    parser.add_argument("--frame", "-f", type=int, default=0, help="Frame index to visualize (default: 0)")
    parser.add_argument("--cmap", "-c", type=str, default="Spectral_r", help="Colormap (default: Spectral_r)")
    parser.add_argument("--save", "-s", type=str, default=None, help="Save figure to path instead of showing")
    parser.add_argument("--video", "-v", action="store_true", help="Create video instead of single frame")
    parser.add_argument("--fps", type=int, default=30, help="FPS for video output (default: 30)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output path for video")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} does not exist")
        return

    if args.video:
        output_path = args.output or str(input_path.parent / "depth_vis.mp4")
        visualize_depth_video(args.input, output_path, args.cmap, args.fps)
    else:
        visualize_depth(args.input, args.frame, args.cmap, args.save)


if __name__ == "__main__":
    main()
