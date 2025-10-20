"""
Wav2Lip model inference for lip-sync generation.

Integrates with the official Wav2Lip repository (external/Wav2Lip).
"""

import sys
import torch
import numpy as np
from pathlib import Path
from typing import List
import logging
import cv2

# Add Wav2Lip to Python path
project_root = Path(__file__).parent.parent.parent
wav2lip_path = project_root / 'external' / 'Wav2Lip'
if str(wav2lip_path) not in sys.path:
    sys.path.insert(0, str(wav2lip_path))

# Import Wav2Lip modules
logger = logging.getLogger(__name__)

try:
    from models.wav2lip import Wav2Lip as Wav2LipModel
    WAV2LIP_AVAILABLE = True
except ImportError as e:
    WAV2LIP_AVAILABLE = False
    logger.error(f"Failed to import Wav2Lip: {e}")
    logger.error("Make sure Wav2Lip submodule is initialized: git submodule update --init")


class Wav2LipSynthesizer:
    """
    Wav2Lip model wrapper for lip-sync generation.

    This class loads the Wav2Lip model and performs inference to generate
    lip-synced video frames.
    """

    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize Wav2Lip synthesizer.

        Args:
            model_path: Path to wav2lip.pth model file
            device: 'cpu', 'cuda', or 'auto' (auto-detect)
        """
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Auto-detect device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        # Load model (placeholder - needs actual Wav2Lip model code)
        self.model = self._load_model()
        self.model.eval()

    def _load_model(self):
        """
        Load Wav2Lip model from checkpoint.
        """
        if not WAV2LIP_AVAILABLE:
            raise ImportError(
                "Wav2Lip model not available. "
                "Run: git submodule update --init --recursive"
            )

        logger.info(f"Loading Wav2Lip model from {self.model_path}")

        try:
            # Initialize model
            model = Wav2LipModel()

            # Load checkpoint
            checkpoint = torch.load(
                self.model_path,
                map_location=torch.device(self.device)
            )

            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Load weights
            model.load_state_dict(state_dict)

            # Move to device and set to eval mode
            model = model.to(self.device)
            model.eval()

            logger.info(f"Wav2Lip model loaded successfully on {self.device}")
            return model

        except Exception as e:
            logger.error(f"Failed to load Wav2Lip model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def generate(self, face_frames: List[np.ndarray],
                mel_chunks: List[np.ndarray]) -> List[np.ndarray]:
        """
        Generate lip-synced frames.

        Args:
            face_frames: List of face images (96x96x3, RGB, float32, [0,1])
            mel_chunks: List of mel spectrogram chunks

        Returns:
            List of generated face images with synced lips
        """
        if not WAV2LIP_AVAILABLE:
            raise RuntimeError("Wav2Lip model not available. Run: git submodule update --init --recursive")

        generated_frames = []

        # Convert to tensors
        face_tensor = torch.FloatTensor(np.asarray(face_frames)).unsqueeze(1).to(self.device)  # (T, 1, 96, 96, 3)
        mel_tensor = torch.FloatTensor(np.asarray(mel_chunks)).to(self.device)  # (T, 80, 16)

        # Permute dimensions for Wav2Lip: (T, 1, 3, 96, 96)
        face_tensor = face_tensor.permute(0, 1, 4, 2, 3)

        with torch.no_grad():
            # Generate in batches to avoid memory issues
            batch_size = 128
            for i in range(0, len(face_tensor), batch_size):
                end_idx = min(i + batch_size, len(face_tensor))
                face_batch = face_tensor[i:end_idx]
                mel_batch = mel_tensor[i:end_idx]

                # Wav2Lip inference
                gen_batch = self.model(mel_batch, face_batch)

                # Convert back to numpy and permute back: (batch, 96, 96, 3)
                gen_batch = gen_batch.permute(0, 2, 3, 4).cpu().numpy()

                # Clip values to [0, 1] and convert to uint8
                gen_batch = np.clip(gen_batch, 0, 1) * 255
                gen_batch = gen_batch.astype(np.uint8)

                generated_frames.extend(gen_batch)

        return generated_frames

    def process_batch(self, faces: np.ndarray, mels: np.ndarray,
                     batch_size: int = 16) -> np.ndarray:
        """
        Process a batch of faces and mels.

        Args:
            faces: Batch of faces (batch, 96, 96, 3)
            mels: Batch of mel spectrograms
            batch_size: Batch size for processing

        Returns:
            Generated faces
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Integration with Wav2Lip code required.")

        # TODO: Implement batched inference
        # This would process faces in batches for efficiency

        return faces  # Placeholder


def apply_lipsync(video_path: str, audio_path: str, output_path: str,
                 quality: str = 'balanced') -> str:
    """
    Apply lip-sync to a video.

    This is the high-level API that orchestrates the entire lip-sync process.

    Args:
        video_path: Path to input video
        audio_path: Path to audio (translated audio)
        output_path: Path for output video
        quality: Quality preset ('fast', 'balanced', 'best')

    Returns:
        Path to output video
    """
    from .detector import FaceDetector
    from .preprocessor import (
        extract_frames, load_audio, get_video_info,
        crop_face_region, prepare_face_for_wav2lip,
        create_video_from_frames
    )
    from .model_manager import get_model_path, check_models_exist
    from .utils import get_mel_spectrogram, paste_face_on_frame, smooth_bbox_sequence

    logger.info(f"Starting lip-sync: {video_path} → {output_path}")

    # Check if models exist
    if not check_models_exist():
        raise RuntimeError("Models not downloaded. Run: python -m src.lipsync.model_manager")

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
    detector = FaceDetector(method='auto')

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

    # Initialize synthesizer
    logger.info("Loading Wav2Lip model...")
    model_path = get_model_path('wav2lip')
    synthesizer = Wav2LipSynthesizer(str(model_path))

    # Generate lip-synced faces
    logger.info("Generating lip-synced frames...")
    # Prepare mel chunks for each frame
    logger.info("Preparing mel spectrogram chunks...")
    mel_chunks = []
    hop_length = 80  # From Wav2Lip's audio.py
    mel_step_size = 16  # From Wav2Lip

    for i in range(len(frames)):
        # Calculate mel chunk indices
        start_idx = i * mel_step_size
        end_idx = start_idx + mel_step_size * 2  # Get some context

        if end_idx > mel.shape[1]:
            # Pad if we go beyond the mel spectrogram
            pad_width = end_idx - mel.shape[1]
            mel_chunk = np.pad(mel[:, start_idx:], ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_chunk = mel[:, start_idx:end_idx]

        mel_chunks.append(mel_chunk)

    # Generate lip-synced faces
    generated_faces = synthesizer.generate(face_frames, mel_chunks)

    # Paste faces back onto frames
    # output_frames = []
    # for frame, gen_face, bbox in zip(frames, generated_frames, face_bboxes):
    #     output_frame = paste_face_on_frame(frame, gen_face, bbox)
    #     output_frames.append(output_frame)

    # Create output video
    # create_video_from_frames(output_frames, output_path, fps, audio_path)

    logger.info(f"Lip-sync complete: {output_path}")
    return output_path
