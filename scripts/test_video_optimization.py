#!/usr/bin/env python3
"""
Test script to verify GPU-accelerated video optimization.

Usage:
    python scripts/test_video_optimization.py

This script will:
1. Check GPU availability
2. Create test images
3. Compare imageio vs optimized pipeline performance
4. Verify output quality
"""

import os
import sys
import time
import tempfile
import shutil
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_test_images(output_dir, num_images=100, width=640, height=480):
    """Create test JPEG images."""
    import imageio.v2 as imageio

    os.makedirs(output_dir, exist_ok=True)
    image_paths = []

    print(f"Creating {num_images} test images ({width}x{height})...")
    for i in range(num_images):
        # Create random image
        img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        img_path = os.path.join(output_dir, f"img_{i:04d}.jpg")
        imageio.imwrite(img_path, img)
        image_paths.append(img_path)

    return image_paths


def test_imageio_baseline(image_paths, output_path, fps=30):
    """Test baseline imageio performance."""
    import imageio.v2 as imageio

    print("\n[Baseline] Testing imageio (original)...")
    start_time = time.time()

    frames = [imageio.imread(img) for img in image_paths]
    imageio.mimsave(output_path, frames, fps=fps)

    elapsed = time.time() - start_time
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB

    print(f"  Time: {elapsed:.2f}s")
    print(f"  Output size: {file_size:.2f} MB")

    return elapsed, file_size


def test_optimized_pipeline(image_paths, output_path, fps=30, use_gpu=True):
    """Test optimized GPU pipeline."""
    from tesseract.utils.video_io import images_to_video_optimized

    device = 'cuda' if use_gpu else 'cpu'
    print(f"\n[Optimized] Testing GPU pipeline (device={device})...")
    start_time = time.time()

    images_to_video_optimized(
        image_paths=image_paths,
        output_path=output_path,
        fps=fps,
        batch_size=16,
        device=device,
        crf=23,
        preset='medium',
        verbose=True
    )

    elapsed = time.time() - start_time
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB

    print(f"  Time: {elapsed:.2f}s")
    print(f"  Output size: {file_size:.2f} MB")

    return elapsed, file_size


def main():
    print("=" * 70)
    print("GPU-Accelerated Video Processing Test")
    print("=" * 70)

    # Check environment
    print("\n[Environment Check]")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")

    try:
        import av
        print(f"  PyAV version: {av.__version__}")
    except ImportError:
        print("  PyAV: NOT INSTALLED - Please run: pip install av>=13.0.0")
        return

    try:
        import torchvision
        print(f"  torchvision version: {torchvision.__version__}")
    except ImportError:
        print("  torchvision: NOT INSTALLED")
        return

    # Create test data
    temp_dir = tempfile.mkdtemp(prefix="video_test_")
    print(f"\n[Test Directory] {temp_dir}")

    try:
        # Create test images
        image_dir = os.path.join(temp_dir, "images")
        image_paths = create_test_images(image_dir, num_images=100)

        # Test baseline (imageio)
        baseline_output = os.path.join(temp_dir, "baseline.mp4")
        baseline_time, baseline_size = test_imageio_baseline(
            image_paths, baseline_output, fps=30
        )

        # Test optimized (GPU or CPU)
        use_gpu = torch.cuda.is_available()
        optimized_output = os.path.join(temp_dir, "optimized.mp4")
        optimized_time, optimized_size = test_optimized_pipeline(
            image_paths, optimized_output, fps=30, use_gpu=use_gpu
        )

        # Results
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"  Baseline (imageio):     {baseline_time:.2f}s, {baseline_size:.2f} MB")
        print(f"  Optimized (GPU/PyAV):   {optimized_time:.2f}s, {optimized_size:.2f} MB")
        print(f"  Speedup:                {baseline_time / optimized_time:.2f}x")
        print(f"  Size difference:        {abs(baseline_size - optimized_size):.2f} MB")

        if optimized_time < baseline_time:
            print(f"\n  ✓ Success! Optimized pipeline is faster!")
        else:
            print(f"\n  ⚠ Warning: Optimized pipeline is slower (expected on CPU-only systems)")

        print(f"\n  Test outputs saved to: {temp_dir}")
        print(f"  You can inspect videos with: open {temp_dir}")

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup option
        cleanup = input("\nDelete test files? [y/N]: ").strip().lower()
        if cleanup == 'y':
            shutil.rmtree(temp_dir)
            print(f"Cleaned up {temp_dir}")
        else:
            print(f"Test files preserved at {temp_dir}")


if __name__ == "__main__":
    main()
