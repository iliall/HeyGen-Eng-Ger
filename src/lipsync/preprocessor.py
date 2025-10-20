"""Video and audio preprocessing for Wav2Lip."""

import cv2
import numpy as np
import librosa
from pathlib import Path
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


def load_audio(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """
    Load and resample audio for Wav2Lip.

    Wav2Lip expects 16kHz mono audio.

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (default: 16000 Hz)

    Returns:
        Audio waveform as numpy array
    """
    try:
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        return audio
    except Exception as e:
        raise RuntimeError(f"Failed to load audio from {audio_path}: {e}")


def extract_frames(video_path: str) -> Tuple[List[np.ndarray], float]:
    """
    Extract all frames from video.

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (frames list, fps)
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    logger.info(f"Extracted {len(frames)} frames at {fps} FPS")
    return frames, fps


def get_video_info(video_path: str) -> dict:
    """
    Get video metadata.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video information
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    }

    cap.release()
    return info


def crop_face_region(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop face region from frame.

    Args:
        frame: Video frame
        bbox: Bounding box (x, y, w, h)

    Returns:
        Cropped face region
    """
    x, y, w, h = bbox
    return frame[y:y+h, x:x+w]


def resize_frame(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize frame to target size.

    Args:
        frame: Input frame
        target_size: (width, height)

    Returns:
        Resized frame
    """
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)


def prepare_face_for_wav2lip(face: np.ndarray) -> np.ndarray:
    """
    Prepare face image for Wav2Lip model.

    Wav2Lip expects 96x96 RGB images with values in [0, 1].

    Args:
        face: Face region (any size, BGR format)

    Returns:
        Preprocessed face (96, 96, 3) in RGB, float32, range [0, 1]
    """
    # Resize to 96x96 (Wav2Lip input size)
    face_resized = cv2.resize(face, (96, 96), interpolation=cv2.INTER_LANCZOS4)

    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    face_normalized = face_rgb.astype(np.float32) / 255.0

    return face_normalized


def prepare_audio_chunk(audio: np.ndarray, start_frame: int, fps: float,
                       mel_step_size: int = 16) -> np.ndarray:
    """
    Prepare audio chunk for a specific frame.

    Args:
        audio: Full audio waveform (16kHz)
        start_frame: Starting frame number
        fps: Video frames per second
        mel_step_size: Number of audio frames per video frame

    Returns:
        Audio chunk for this frame
    """
    # Calculate audio window for this frame
    # Wav2Lip uses mel spectrogram with step_size=16
    # Each video frame corresponds to mel_step_size audio frames

    sr = 16000  # Wav2Lip expects 16kHz audio
    hop_length = 80  # Librosa mel spectrogram hop length

    # Calculate sample indices
    start_sample = int(start_frame * sr / fps)
    end_sample = int((start_frame + 1) * sr / fps)

    # Add some padding for mel spectrogram context
    context_frames = mel_step_size
    context_samples = context_frames * hop_length

    start_sample = max(0, start_sample - context_samples)
    end_sample = min(len(audio), end_sample + context_samples)

    return audio[start_sample:end_sample]


def create_video_from_frames(frames: List[np.ndarray], output_path: str,
                            fps: float, audio_path: str = None) -> None:
    """
    Create video file from frames.

    Args:
        frames: List of video frames (BGR format)
        output_path: Path for output video
        fps: Frames per second
        audio_path: Optional audio file to merge
    """
    if not frames:
        raise ValueError("No frames to write")

    height, width = frames[0].shape[:2]

    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()

    logger.info(f"Created video: {output_path}")

    # Merge with audio using ffmpeg if audio path provided
    if audio_path:
        merge_audio_with_video(output_path, audio_path)


def merge_audio_with_video(video_path: str, audio_path: str) -> None:
    """
    Merge audio with video using ffmpeg.

    Args:
        video_path: Path to video file (will be overwritten)
        audio_path: Path to audio file
    """
    import subprocess

    temp_output = str(Path(video_path).with_suffix('.temp.mp4'))

    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-strict', 'experimental',
        '-shortest',
        temp_output
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)

        # Replace original with temp
        Path(temp_output).replace(video_path)
        logger.info(f"Merged audio with video: {video_path}")

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e.stderr.decode()}")
        raise RuntimeError(f"Failed to merge audio with video: {e}")
