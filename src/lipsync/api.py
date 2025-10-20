"""
High-level lip-sync API for the main pipeline.

This function provides a simple interface to apply lip-sync to a video.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional

from .detector import FaceDetector
from .preprocessor import (
    extract_frames, load_audio, get_video_info,
    crop_face_region, prepare_face_for_wav2lip,
    create_video_from_frames
)
from .synthesizer import Wav2LipSynthesizer
from .model_manager import get_model_path, check_models_exist
from .utils import (
    get_mel_spectrogram, paste_face_on_frame,
    smooth_bbox_sequence
)

logger = logging.getLogger(__name__)


def apply_lipsync(video_path: str, audio_path: str, output_path: str,
                 quality: str = 'balanced', face_detector: Optional[str] = None) -> str:
    """
    Apply lip-sync to a video using Wav2Lip.

    This is the high-level API that orchestrates the entire lip-sync process.

    Args:
        video_path: Path to input video
        audio_path: Path to audio (translated audio)
        output_path: Path for output video
        quality: Quality preset ('fast', 'balanced', 'best')
        face_detector: Face detection method ('auto', 'retinaface', 'haar')

    Returns:
        Path to output video
    """
    logger.info(f"Starting lip-sync: {video_path} → {output_path}")

    # Check if models exist
    if not check_models_exist():
        raise RuntimeError(
            "Models not downloaded. Run: python -m src.lipsync.model_manager"
        )

    # Get video info
    video_info = get_video_info(video_path)
    logger.info(f"Video: {video_info['width']}x{video_info['height']}, "
                f"{video_info['fps']} FPS, {video_info['duration']:.2f}s")

    # Extract frames
    logger.info("Extracting video frames...")
    frames, fps = extract_frames(video_path)

    # Load audio
    logger.info("Loading audio...")
    audio = load_audio(audio_path, target_sr=16000)

    # Detect faces
    logger.info("Detecting faces...")
    detector_method = face_detector or 'auto'
    detector = FaceDetector(method=detector_method)

    face_bboxes = []
    for i, frame in enumerate(frames):
        bbox = detector.get_largest_face(frame)
        if bbox is None:
            logger.warning(f"No face detected in frame {i}, using previous bbox")
            bbox = face_bboxes[-1] if face_bboxes else (0, 0, 100, 100)
        face_bboxes.append(bbox)

    # Smooth bboxes to reduce jitter
    face_bboxes = smooth_bbox_sequence(face_bboxes)

    # Prepare face frames
    logger.info("Preparing face regions...")
    face_frames = []
    for frame, bbox in zip(frames, face_bboxes):
        # Expand and square the bbox
        bbox_exp = detector.expand_bbox(bbox, scale=1.3, frame_shape=frame.shape[:2])
        bbox_square = detector.make_square_bbox(bbox_exp, frame_shape=frame.shape[:2])

        # Crop and prepare
        face = crop_face_region(frame, bbox_square)
        face_prep = prepare_face_for_wav2lip(face)
        face_frames.append(face_prep)

    # Compute mel spectrogram
    logger.info("Computing mel spectrogram...")
    mel = get_mel_spectrogram(audio)

    # Prepare mel chunks for each frame
    logger.info("Preparing mel spectrogram chunks...")
    mel_chunks = []
    mel_step_size = 16  # From Wav2Lip

    for i in range(len(frames)):
        # Calculate mel chunk indices
        start_idx = i * mel_step_size
        end_idx = start_idx + mel_step_size * 2  # Get some context

        if end_idx > mel.shape[1]:
            # Pad if we go beyond the mel spectrogram
            pad_width = end_idx - mel.shape[1]
            mel_chunk = mel[:, start_idx:]
            if start_idx >= mel.shape[1]:
                # If we're completely beyond, use zeros
                mel_chunk = np.zeros((80, mel_step_size * 2))
            else:
                mel_chunk = np.pad(mel_chunk, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_chunk = mel[:, start_idx:end_idx]

        mel_chunks.append(mel_chunk)

    # Initialize synthesizer
    logger.info("Loading Wav2Lip model...")
    model_path = get_model_path('wav2lip')
    synthesizer = Wav2LipSynthesizer(str(model_path))

    # Generate lip-synced faces
    logger.info("Generating lip-synced frames...")
    generated_faces = synthesizer.generate(face_frames, mel_chunks)

    # Paste faces back onto frames
    logger.info("Pasting faces back onto frames...")
    output_frames = []
    for frame, gen_face, bbox in zip(frames, generated_faces, face_bboxes):
        # Convert generated face back to BGR for pasting
        import cv2
        gen_face_bgr = cv2.cvtColor(gen_face, cv2.COLOR_RGB2BGR)

        # Resize generated face to match the expanded bbox
        x, y, w, h = bbox
        if gen_face_bgr.shape[:2] != (h, w):
            gen_face_bgr = cv2.resize(gen_face_bgr, (w, h))

        # Paste face back onto frame
        output_frame = paste_face_on_frame(frame, gen_face_bgr, bbox)
        output_frames.append(output_frame)

    # Create output video
    logger.info("Creating final video...")
    create_video_from_frames(output_frames, output_path, fps, audio_path)

    logger.info(f"Lip-sync complete: {output_path}")
    return output_path