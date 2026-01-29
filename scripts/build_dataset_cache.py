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


def sample_from_cache(
    input_cache: str,
    output_cache: str,
    num_samples: int,
    method: str = "random",
    seed: int = 42,
) -> None:
    """
    Sample a subset of data from an existing cache file.

    Args:
        input_cache: Path to input JSONL cache file
        output_cache: Path to output JSONL cache file
        num_samples: Number of samples to select
        method: Sampling method
            - "random": Random sampling
            - "first": Take first N samples
            - "last": Take last N samples
            - "uniform": Uniformly distributed sampling (every K-th sample)
            - "stratified": Stratified by instruction keywords (if possible)
        seed: Random seed for reproducibility
    """
    import random
    random.seed(seed)
    np.random.seed(seed)

    # Load input cache
    input_path = Path(input_cache)
    if not input_path.exists():
        print(f"Error: Input cache file not found: {input_cache}")
        sys.exit(1)

    with open(input_path, "r") as f:
        lines = f.readlines()

    # First line is header
    header = json.loads(lines[0])
    samples = [json.loads(line) for line in lines[1:]]

    total_samples = len(samples)
    if num_samples > total_samples:
        print(f"Warning: Requested {num_samples} samples but only {total_samples} available")
        num_samples = total_samples

    print(f"Sampling {num_samples} from {total_samples} samples using '{method}' method...")

    # Apply sampling method
    if method == "random":
        selected = random.sample(samples, num_samples)

    elif method == "first":
        selected = samples[:num_samples]

    elif method == "last":
        selected = samples[-num_samples:]

    elif method == "uniform":
        # Select every K-th sample
        step = max(1, total_samples // num_samples)
        indices = list(range(0, total_samples, step))[:num_samples]
        selected = [samples[i] for i in indices]

    elif method == "stratified":
        # Group by instruction keywords and sample proportionally
        from collections import defaultdict
        keyword_groups = defaultdict(list)

        # Common robot action keywords
        keywords = ["pick", "place", "push", "pull", "open", "close", "move", "rotate", "lift", "drop"]

        for sample in samples:
            instruction = sample.get("instruction", "").lower()
            matched = False
            for kw in keywords:
                if kw in instruction:
                    keyword_groups[kw].append(sample)
                    matched = True
                    break
            if not matched:
                keyword_groups["other"].append(sample)

        # Sample proportionally from each group
        selected = []
        group_sizes = {k: len(v) for k, v in keyword_groups.items()}
        total_in_groups = sum(group_sizes.values())

        for keyword, group in keyword_groups.items():
            group_quota = max(1, int(num_samples * len(group) / total_in_groups))
            group_quota = min(group_quota, len(group))
            selected.extend(random.sample(group, group_quota))

        # If we don't have enough, add more randomly
        if len(selected) < num_samples:
            remaining = [s for s in samples if s not in selected]
            additional = min(num_samples - len(selected), len(remaining))
            selected.extend(random.sample(remaining, additional))

        # If we have too many, trim
        if len(selected) > num_samples:
            selected = random.sample(selected, num_samples)

    else:
        print(f"Error: Unknown method '{method}'")
        print("Available strategies: random, first, last, uniform, stratified")
        sys.exit(1)

    # Sort by scene_id
    selected.sort(key=lambda x: int(x["scene_id"]))

    # Update header
    new_header = {
        "version": header.get("version", 1),
        "created_at": datetime.now().isoformat(),
        "rgb_root": header.get("rgb_root", ""),
        "output_root": header.get("output_root", ""),
        "total_scenes": header.get("total_scenes", 0),
        "complete_samples": len(selected),
        "sampled_from": str(input_path),
        "sampling_method": method,
        "sampling_seed": seed,
        "original_samples": total_samples,
    }

    # Write output
    output_path = Path(output_cache)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(json.dumps(new_header) + "\n")
        for sample in selected:
            f.write(json.dumps(sample) + "\n")

    print(f"\n{'=' * 60}")
    print(f"Sampling complete: {output_cache}")
    print(f"{'=' * 60}")
    print(f"Original samples: {total_samples}")
    print(f"Selected samples: {len(selected)}")
    print(f"Method:           {method}")
    print(f"Seed:             {seed}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Build dataset cache for RoboDepthNormal training"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Build command (original functionality)
    build_parser = subparsers.add_parser("build", help="Build cache from directories")
    build_parser.add_argument(
        "-i", "--rgb-root",
        type=str,
        required=True,
        help="Root directory containing RGB videos (vepfs disk)",
    )
    build_parser.add_argument(
        "-o", "--output-root",
        type=str,
        required=True,
        help="Root directory containing depth/normal outputs (tos disk)",
    )
    build_parser.add_argument(
        "-c", "--cache-file",
        type=str,
        required=True,
        help="Output path for cache JSONL file",
    )
    build_parser.add_argument(
        "-w", "--num-workers",
        type=int,
        default=8,
        help="Number of parallel workers for scanning (default: 8)",
    )
    build_parser.add_argument(
        "-s", "--skip-validity-check",
        action="store_true",
        help="Skip validity check for depth/normal files (only check existence)",
    )

    # Sample command (new functionality)
    sample_parser = subparsers.add_parser("sample", help="Sample from existing cache")
    sample_parser.add_argument(
        "-i", "--input-cache",
        type=str,
        required=True,
        help="Input JSONL cache file",
    )
    sample_parser.add_argument(
        "-o", "--output-cache",
        type=str,
        required=True,
        help="Output JSONL cache file",
    )
    sample_parser.add_argument(
        "-n", "--num-samples",
        type=int,
        required=True,
        help="Number of samples to select",
    )
    sample_parser.add_argument(
        "-m", "--method",
        type=str,
        default="random",
        choices=["random", "first", "last", "uniform", "stratified"],
        help="Sampling method (default: random)",
    )
    sample_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    if args.command == "build":
        build_cache(
            rgb_root=args.rgb_root,
            output_root=args.output_root,
            cache_file=args.cache_file,
            num_workers=args.num_workers,
            check_validity=not args.skip_validity_check,
        )
    elif args.command == "sample":
        sample_from_cache(
            input_cache=args.input_cache,
            output_cache=args.output_cache,
            num_samples=args.num_samples,
            method=args.method,
            seed=args.seed,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
