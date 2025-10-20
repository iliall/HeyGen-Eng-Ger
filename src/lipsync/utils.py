"""Utility functions for lip-sync processing."""

import numpy as np
import cv2
from typing import Tuple


def get_mel_spectrogram(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Compute mel spectrogram from audio.

    This is what Wav2Lip uses as audio representation.

    Args:
        audio: Audio waveform (mono, 16kHz)
        sr: Sample rate (default: 16000)

    Returns:
        Mel spectrogram
    """
    import librosa

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=800,
        hop_length=200,
        n_mels=80
    )

    # Convert to log scale
    mel_db = librosa.power_to_db(mel, ref=np.max)

    return mel_db


def blend_faces(original: np.ndarray, generated: np.ndarray,
                alpha: float = 0.8) -> np.ndarray:
    """
    Blend original and generated faces.

    Helps reduce quality loss from Wav2Lip.

    Args:
        original: Original face image
        generated: Wav2Lip generated face
        alpha: Blending factor (1.0 = all generated, 0.0 = all original)

    Returns:
        Blended face
    """
    if original.shape != generated.shape:
        generated = cv2.resize(generated, (original.shape[1], original.shape[0]))

    blended = cv2.addWeighted(generated, alpha, original, 1 - alpha, 0)
    return blended


def smooth_bbox_sequence(bboxes: list, window_size: int = 5) -> list:
    """
    Smooth bounding box sequence to reduce jitter.

    Args:
        bboxes: List of bounding boxes [(x, y, w, h), ...]
        window_size: Size of smoothing window

    Returns:
        Smoothed bounding boxes
    """
    if len(bboxes) < window_size:
        return bboxes

    smoothed = []

    for i in range(len(bboxes)):
        # Get window around current frame
        start = max(0, i - window_size // 2)
        end = min(len(bboxes), i + window_size // 2 + 1)

        window = bboxes[start:end]

        # Average coordinates
        x = int(np.mean([b[0] for b in window]))
        y = int(np.mean([b[1] for b in window]))
        w = int(np.mean([b[2] for b in window]))
        h = int(np.mean([b[3] for b in window]))

        smoothed.append((x, y, w, h))

    return smoothed


def calculate_iou(bbox1: Tuple[int, int, int, int],
                 bbox2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Used for tracking face across frames.

    Args:
        bbox1: First bounding box (x, y, w, h)
        bbox2: Second bounding box (x, y, w, h)

    Returns:
        IoU score [0, 1]
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Convert to (x1, y1, x2, y2) format
    box1 = (x1, y1, x1 + w1, y1 + h1)
    box2 = (x2, y2, x2 + w2, y2 + h2)

    # Calculate intersection
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Calculate union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def paste_face_on_frame(frame: np.ndarray, face: np.ndarray,
                        bbox: Tuple[int, int, int, int],
                        feather: int = 5) -> np.ndarray:
    """
    Paste generated face back onto original frame with feathering.

    Args:
        frame: Original frame
        face: Generated face (should match bbox size)
        bbox: Bounding box (x, y, w, h)
        feather: Feathering size for smooth blending

    Returns:
        Frame with face pasted
    """
    x, y, w, h = bbox

    # Resize face to match bbox
    if face.shape[:2] != (h, w):
        face = cv2.resize(face, (w, h))

    # Create a copy to modify
    result = frame.copy()

    if feather > 0:
        # Create feathered mask
        mask = np.zeros((h, w), dtype=np.float32)
        mask[feather:-feather, feather:-feather] = 1.0

        # Apply Gaussian blur for smooth transition
        mask = cv2.GaussianBlur(mask, (feather * 2 + 1, feather * 2 + 1), 0)
        mask = mask[:, :, np.newaxis]  # Add channel dimension

        # Blend using mask
        result[y:y+h, x:x+w] = (
            face * mask + result[y:y+h, x:x+w] * (1 - mask)
        ).astype(np.uint8)
    else:
        # Direct paste without feathering
        result[y:y+h, x:x+w] = face

    return result
