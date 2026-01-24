"""GPU-accelerated video I/O utilities for TesserAct.

This module provides high-performance video encoding and image decoding using:
- GPU-accelerated JPEG decoding via torchvision
- Multi-threaded H.264 encoding via PyAV
- Streaming pipelines to minimize memory usage
"""

import fractions
import logging
from os import PathLike
from pathlib import Path
from typing import Iterator, List, Optional, Union

import av
import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def batch_decode_images_gpu(
    image_paths: List[Union[str, PathLike]],
    batch_size: int = 16,
    device: str = 'cuda',
    verbose: bool = False,
) -> Iterator[np.ndarray]:
    """
    Decode JPEG images in batches using GPU acceleration.

    Args:
        image_paths: List of paths to JPEG images
        batch_size: Number of images to decode simultaneously on GPU
        device: Device to use ('cuda' or 'cpu')
        verbose: Show progress bar

    Yields:
        numpy.ndarray: Decoded images as uint8 arrays [H, W, 3]

    Note:
        This function uses torchvision's GPU JPEG decoder when available,
        falling back to CPU decoding if GPU is unavailable.
    """
    # Check if CUDA is available when requested
    use_gpu = device == 'cuda' and torch.cuda.is_available()
    if device == 'cuda' and not use_gpu:
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = 'cpu'

    # Import torchvision at runtime to avoid hard dependency
    try:
        import torchvision.io
    except ImportError:
        raise ImportError(
            "torchvision is required for GPU image decoding. "
            "Install with: pip install torchvision"
        )

    # Process images in batches
    num_images = len(image_paths)
    num_batches = (num_images + batch_size - 1) // batch_size

    iterator = range(num_batches)
    if verbose:
        iterator = tqdm(iterator, desc="Decoding images", total=num_batches)

    for batch_idx in iterator:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_images)
        batch_paths = image_paths[start_idx:end_idx]

        # Decode images in batch
        for img_path in batch_paths:
            # Read file as bytes
            with open(img_path, 'rb') as f:
                img_bytes = f.read()

            # Decode JPEG on GPU/CPU
            img_tensor = torchvision.io.decode_jpeg(
                torch.frombuffer(img_bytes, dtype=torch.uint8),
                device=device
            )

            # Convert to numpy: [C, H, W] -> [H, W, C]
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

            yield img_np


def encode_video_multicore(
    frames: Union[np.ndarray, Iterator[np.ndarray]],
    output_path: Union[str, PathLike],
    fps: float = 30.0,
    codec: Optional[str] = None,
    crf: int = 23,
    preset: str = "medium",
    verbose: bool = False,
) -> None:
    """
    Encode video using multi-threaded H.264 encoding via PyAV.

    Args:
        frames: Either numpy array [N, H, W, 3] or iterator yielding frames
        output_path: Path to output MP4 file
        fps: Frame rate
        codec: Video codec (None = auto-detect, 'libx264' recommended)
        crf: Constant Rate Factor quality (0-51, lower = better, 18-28 recommended)
        preset: Encoding preset ('ultrafast', 'fast', 'medium', 'slow', 'veryslow')
        verbose: Show progress bar

    Note:
        This function uses PyAV's multi-threaded encoding for better performance.
        Based on RollingDepth's write_video_from_numpy implementation.
    """
    # Determine if frames is an array or iterator
    if isinstance(frames, np.ndarray):
        if len(frames.shape) != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Expected shape [n, height, width, 3], got {frames.shape}")
        if frames.dtype != np.uint8:
            raise ValueError(f"Expected dtype uint8, got {frames.dtype}")

        n_frames, height, width, _ = frames.shape
        frames_iterable = range(n_frames)
        is_array = True
    else:
        # Iterator - we need to peek at first frame to get dimensions
        frames = iter(frames)
        first_frame = next(frames)

        if len(first_frame.shape) != 3 or first_frame.shape[-1] != 3:
            raise ValueError(f"Expected shape [height, width, 3], got {first_frame.shape}")
        if first_frame.dtype != np.uint8:
            raise ValueError(f"Expected dtype uint8, got {first_frame.dtype}")

        height, width, _ = first_frame.shape
        n_frames = None  # Unknown for iterator
        is_array = False

        # Create iterator that includes first frame
        def frame_generator():
            yield first_frame
            yield from frames

        frames_iterable = frame_generator()

    # Try to determine codec from output format if not specified
    if codec is None:
        codecs_to_try = ["libx264", "h264", "mpeg4", "mjpeg"]
    else:
        codecs_to_try = [codec]

    fps_rational = fractions.Fraction(fps).limit_denominator()

    # Try available codecs
    for try_codec in codecs_to_try:
        try:
            container = av.open(str(output_path), mode="w")
            stream = container.add_stream(try_codec, rate=fps_rational)
            if verbose:
                logger.info(f"Using codec: {try_codec}")
            break
        except av.codec.codec.UnknownCodecError:
            if try_codec == codecs_to_try[-1]:  # Last codec in list
                raise ValueError(
                    f"No working codec found. Tried: {codecs_to_try}. "
                    "Please install ffmpeg with necessary codecs."
                )
            continue

    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    # Only set these options for x264-compatible codecs
    if try_codec in ["libx264", "h264"]:
        stream.options = {"crf": str(crf), "preset": preset}

    # Create a single VideoFrame object and reuse it
    video_frame = av.VideoFrame(width, height, "rgb24")

    # Setup progress bar
    if verbose and is_array:
        progress = tqdm(total=n_frames, desc="Encoding video")
    else:
        progress = None

    try:
        frame_count = 0
        if is_array:
            # Array mode - index directly
            for frame_idx in frames_iterable:
                current_frame = frames[frame_idx]

                # Update frame data in-place
                video_frame.to_ndarray()[:] = current_frame

                packet = stream.encode(video_frame)
                container.mux(packet)

                frame_count += 1
                if progress:
                    progress.update(1)
        else:
            # Iterator mode
            for current_frame in frames_iterable:
                # Update frame data in-place
                video_frame.to_ndarray()[:] = current_frame

                packet = stream.encode(video_frame)
                container.mux(packet)

                frame_count += 1
                if progress:
                    progress.update(1)

        # Flush the stream
        packet = stream.encode(None)
        container.mux(packet)

        if verbose:
            logger.info(f"Encoded {frame_count} frames to {output_path}")

    finally:
        if progress:
            progress.close()
        container.close()


def images_to_video_optimized(
    image_paths: List[Union[str, PathLike]],
    output_path: Union[str, PathLike],
    fps: float = 30.0,
    batch_size: int = 16,
    device: str = 'cuda',
    crf: int = 23,
    preset: str = 'medium',
    codec: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """
    Convert image sequence to video using GPU-accelerated pipeline.

    This function provides a complete optimized pipeline:
    1. GPU-accelerated JPEG decoding in batches
    2. Streaming frames to multi-threaded encoder
    3. Low memory footprint (no full buffer)

    Args:
        image_paths: List of paths to JPEG images
        output_path: Path to output MP4 file
        fps: Frame rate
        batch_size: Number of images to decode simultaneously on GPU
        device: Device for decoding ('cuda' or 'cpu')
        crf: Quality (0-51, lower = better, 18-28 recommended)
        preset: Encoding speed ('ultrafast', 'fast', 'medium', 'slow', 'veryslow')
        codec: Video codec (None = auto-detect)
        verbose: Show progress messages

    Example:
        >>> images_to_video_optimized(
        ...     image_paths=sorted(glob.glob("frames/*.jpg")),
        ...     output_path="output.mp4",
        ...     fps=30,
        ...     batch_size=16,
        ...     crf=23,
        ...     preset='medium'
        ... )
    """
    if not image_paths:
        raise ValueError("No image paths provided")

    # Create output directory if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check GPU availability
    use_gpu = device == 'cuda' and torch.cuda.is_available()
    if device == 'cuda' and not use_gpu:
        if verbose:
            logger.warning("CUDA not available, falling back to CPU decoding")
        device = 'cpu'
    elif use_gpu and verbose:
        logger.info(f"Using GPU ({torch.cuda.get_device_name(0)}) for JPEG decoding")

    # Decode images and stream to encoder
    frames_iterator = batch_decode_images_gpu(
        image_paths,
        batch_size=batch_size,
        device=device,
        verbose=verbose
    )

    # Encode video with multi-threading
    encode_video_multicore(
        frames_iterator,
        output_path,
        fps=fps,
        codec=codec,
        crf=crf,
        preset=preset,
        verbose=verbose
    )

    if verbose:
        logger.info(f"Successfully created {output_path}")
