#!/usr/bin/env python3
"""
Build dataset cache for RoboDepthNormal training.

Scans RGB files from input disk (vepfs) and checks if corresponding
depth and normal files exist on output disk (tos). Only complete samples
(rgb + depth + normal) are included in the cache.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


def check_depth_valid(depth_path: Path) -> bool:
    """Check if depth.npz file exists and is valid."""
    if not depth_path.exists():
        return False
    try:
        with np.load(depth_path) as f:
            _ = f["arr_0"]
        return True
    except Exception:
        return False


def check_normal_valid(normal_path: Path) -> bool:
    """Check if normal.mp4 file exists and is valid."""
    if not normal_path.exists():
        return False
    try:
        import imageio.v3 as imageio
        reader = imageio.imiter(str(normal_path))
        next(reader)  # Try to read at least one frame
        return True
    except Exception:
        return False


def check_rgb_valid(rgb_path: Path) -> bool:
    """Check if rgb.mp4 file exists."""
    return rgb_path.exists()


def process_scene(
    scene_id: str,
    rgb_root: Path,
    output_root: Path,
    check_validity: bool = True,
) -> Optional[Dict]:
    """Process a single scene and return sample info if complete."""
    scene_rgb_dir = rgb_root / scene_id
    scene_output_dir = output_root / scene_id

    # Check RGB
    rgb_path = scene_rgb_dir / "video" / "rgb.mp4"
    if not check_rgb_valid(rgb_path):
        return None

    # Check depth
    depth_path = scene_output_dir / "depth" / "npz" / "depth.npz"
    if check_validity:
        if not check_depth_valid(depth_path):
            return None
    else:
        if not depth_path.exists():
            return None

    # Check normal
    normal_path = scene_output_dir / "video" / "normal.mp4"
    if check_validity:
        if not check_normal_valid(normal_path):
            return None
    else:
        if not normal_path.exists():
            return None

    # Load instruction
    instruction_file = scene_rgb_dir / "instruction.txt"
    if instruction_file.exists():
        instruction = instruction_file.read_text().strip()
    else:
        instruction = ""

    return {
        "scene_id": scene_id,
        "instruction": instruction,
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "normal_path": str(normal_path),
    }


def build_cache(
    rgb_root: str,
    output_root: str,
    cache_file: str,
    num_workers: int = 8,
    check_validity: bool = True,
) -> Dict:
    """Build dataset cache by scanning directories."""
    rgb_root = Path(rgb_root)
    output_root = Path(output_root)

    # Get all scene directories
    print(f"Scanning scenes in {rgb_root}...")
    scene_ids = []
    for entry in os.scandir(rgb_root):
        if entry.is_dir():
            scene_ids.append(entry.name)

    print(f"Found {len(scene_ids)} scenes")

    # Process scenes in parallel
    samples = []
    complete_count = 0
    missing_depth = 0
    missing_normal = 0
    missing_rgb = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                process_scene, scene_id, rgb_root, output_root, check_validity
            ): scene_id
            for scene_id in scene_ids
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Checking"):
            scene_id = futures[future]
            try:
                result = future.result()
                if result is not None:
                    samples.append(result)
                    complete_count += 1
                else:
                    # Determine what's missing for stats
                    scene_rgb_dir = rgb_root / scene_id
                    scene_output_dir = output_root / scene_id
                    if not (scene_rgb_dir / "video" / "rgb.mp4").exists():
                        missing_rgb += 1
                    elif not (scene_output_dir / "depth" / "npz" / "depth.npz").exists():
                        missing_depth += 1
                    elif not (scene_output_dir / "video" / "normal.mp4").exists():
                        missing_normal += 1
            except Exception as e:
                print(f"Error processing {scene_id}: {e}")

    # Sort by scene_id (numeric)
    samples.sort(key=lambda x: int(x["scene_id"]))

    # Build cache
    cache = {
        "version": 1,
        "created_at": datetime.now().isoformat(),
        "rgb_root": str(rgb_root),
        "output_root": str(output_root),
        "total_scenes": len(scene_ids),
        "complete_samples": len(samples),
        "samples": samples,
    }

    # Save cache as jsonl (one sample per line for efficient streaming)
    cache_path = Path(cache_file)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        # Write header as first line
        header = {
            "version": 1,
            "created_at": datetime.now().isoformat(),
            "rgb_root": str(rgb_root),
            "output_root": str(output_root),
            "total_scenes": len(scene_ids),
            "complete_samples": len(samples),
        }
        f.write(json.dumps(header) + "\n")
        # Write each sample as a separate line
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Cache built successfully: {cache_file}")
    print(f"{'=' * 60}")
    print(f"Total scenes:     {len(scene_ids)}")
    print(f"Complete samples: {complete_count}")
    print(f"Missing RGB:      {missing_rgb}")
    print(f"Missing depth:    {missing_depth}")
    print(f"Missing normal:   {missing_normal}")
    print(f"{'=' * 60}")

    return cache


def main():
    parser = argparse.ArgumentParser(
        description="Build dataset cache for RoboDepthNormal training"
    )
    parser.add_argument(
        "-i", "--rgb-root",
        type=str,
        required=True,
        help="Root directory containing RGB videos (vepfs disk)",
    )
    parser.add_argument(
        "-o", "--output-root",
        type=str,
        required=True,
        help="Root directory containing depth/normal outputs (tos disk)",
    )
    parser.add_argument(
        "-c", "--cache-file",
        type=str,
        required=True,
        help="Output path for cache JSONL file",
    )
    parser.add_argument(
        "-w", "--num-workers",
        type=int,
        default=8,
        help="Number of parallel workers for scanning (default: 8)",
    )
    parser.add_argument(
        "-s", "--skip-validity-check",
        action="store_true",
        help="Skip validity check for depth/normal files (only check existence)",
    )

    args = parser.parse_args()

    build_cache(
        rgb_root=args.rgb_root,
        output_root=args.output_root,
        cache_file=args.cache_file,
        num_workers=args.num_workers,
        check_validity=not args.skip_validity_check,
    )


if __name__ == "__main__":
    main()
