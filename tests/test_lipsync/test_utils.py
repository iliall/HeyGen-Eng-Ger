"""Tests for lip-sync utility functions."""

import pytest
import numpy as np
import cv2
from src.lipsync.utils import (
    get_mel_spectrogram, blend_faces, smooth_bbox_sequence,
    calculate_iou, paste_face_on_frame
)


class TestLipsyncUtils:
    """Test cases for lip-sync utility functions."""

    def test_get_mel_spectrogram(self):
        """Test mel spectrogram computation."""
        # Generate simple test audio (sine wave)
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

        mel = get_mel_spectrogram(audio, sr)

        # Check shape and properties
        assert mel.shape[0] == 80  # n_mels
        assert mel.shape[1] > 0    # time frames
        assert mel.dtype == np.float32 or mel.dtype == np.float64

    def test_blend_faces(self):
        """Test face blending."""
        # Create two simple test faces
        face1 = np.random.randint(0, 256, (96, 96, 3), dtype=np.uint8)
        face2 = np.random.randint(0, 256, (96, 96, 3), dtype=np.uint8)

        # Test blending
        blended = blend_faces(face1, face2, alpha=0.5)
        assert blended.shape == face1.shape
        assert blended.dtype == face1.dtype

        # Test with different sizes
        face3 = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        blended2 = blend_faces(face1, face3, alpha=0.8)
        assert blended2.shape == face1.shape  # Should resize to match first face

        # Test alpha values
        blended_full1 = blend_faces(face1, face2, alpha=1.0)
        blended_full0 = blend_faces(face1, face2, alpha=0.0)
        assert np.array_equal(blended_full1, face2)
        assert np.array_equal(blended_full0, face1)

    def test_smooth_bbox_sequence(self):
        """Test bounding box sequence smoothing."""
        # Create a sequence of bboxes with some jitter
        bboxes = [
            (100, 100, 50, 50),
            (102, 98, 52, 48),   # Small jitter
            (99, 101, 51, 49),
            (101, 99, 49, 51),
            (100, 100, 50, 50),
        ]

        smoothed = smooth_bbox_sequence(bboxes, window_size=3)

        assert len(smoothed) == len(bboxes)
        assert all(len(bbox) == 4 for bbox in smoothed)

        # Smoothed values should be closer to average
        for bbox in smoothed:
            assert all(isinstance(coord, int) for coord in bbox)

        # Test with sequence shorter than window
        short_bboxes = [(100, 100, 50, 50), (101, 101, 51, 51)]
        smoothed_short = smooth_bbox_sequence(short_bboxes, window_size=5)
        assert smoothed_short == short_bboxes

    def test_calculate_iou(self):
        """Test IoU calculation."""
        # Test identical boxes
        bbox1 = (100, 100, 50, 50)
        bbox2 = (100, 100, 50, 50)
        iou = calculate_iou(bbox1, bbox2)
        assert iou == 1.0

        # Test non-overlapping boxes
        bbox3 = (200, 200, 50, 50)
        iou = calculate_iou(bbox1, bbox3)
        assert iou == 0.0

        # Test partially overlapping boxes
        bbox4 = (125, 125, 50, 50)  # Overlaps partially
        iou = calculate_iou(bbox1, bbox4)
        assert 0.0 < iou < 1.0

        # Test with zero area box
        bbox5 = (100, 100, 0, 50)
        iou = calculate_iou(bbox1, bbox5)
        assert iou == 0.0

    def test_paste_face_on_frame(self):
        """Test pasting face back onto frame."""
        # Create test frame and face
        frame = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        face = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        bbox = (75, 75, 50, 50)  # (x, y, w, h)

        # Test direct paste
        result = paste_face_on_frame(frame, face, bbox, feather=0)
        assert result.shape == frame.shape
        assert result.dtype == frame.dtype

        # Check that face was pasted in correct location
        pasted_region = result[75:125, 75:125]
        assert np.array_equal(pasted_region, face)

        # Test with feathering
        result_feathered = paste_face_on_frame(frame, face, bbox, feather=5)
        assert result_feathered.shape == frame.shape

        # Test with face resize needed
        face_wrong_size = np.random.randint(0, 256, (30, 30, 3), dtype=np.uint8)
        result_resized = paste_face_on_frame(frame, face_wrong_size, bbox, feather=0)
        assert result_resized.shape == frame.shape

        # Test bbox at edges
        bbox_edge = (0, 0, 50, 50)
        result_edge = paste_face_on_frame(frame, face, bbox_edge, feather=0)
        assert result_edge.shape == frame.shape

    def test_blend_faces_different_dtypes(self):
        """Test face blending with different data types."""
        face1 = np.random.randint(0, 256, (96, 96, 3), dtype=np.uint8)
        face2 = np.random.randint(0, 256, (96, 96, 3), dtype=np.uint8)

        # Test with float32
        face1_float = face1.astype(np.float32) / 255.0
        face2_float = face2.astype(np.float32) / 255.0

        blended = blend_faces(face1_float, face2_float, alpha=0.5)
        assert blended.dtype == np.float32
        assert 0.0 <= blended.max() <= 1.0

    def test_smooth_bbox_sequence_stability(self):
        """Test that smoothing produces stable results."""
        # Create sequence with noise
        np.random.seed(42)
        base_bbox = (100, 100, 50, 50)
        noisy_bboxes = []
        for i in range(20):
            noise = np.random.randint(-3, 4, 4)
            bbox = tuple(base_bbox[j] + noise[j] for j in range(4))
            noisy_bboxes.append(bbox)

        smoothed = smooth_bbox_sequence(noisy_bboxes, window_size=5)

        # Smoothed sequence should have less variance
        original_variance = np.var(noisy_bboxes, axis=0)
        smoothed_variance = np.var(smoothed, axis=0)

        assert all(smoothed_var <= orig_var for smoothed_var, orig_var in zip(smoothed_variance, original_variance))