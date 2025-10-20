"""Tests for face detection functionality."""

import pytest
import numpy as np
import cv2
from src.lipsync.detector import FaceDetector


class TestFaceDetector:
    """Test cases for FaceDetector class."""

    def test_init_auto(self):
        """Test automatic detector initialization."""
        detector = FaceDetector(method='auto')
        assert detector.method in ['retinaface', 'haar']
        assert detector.detector is not None

    def test_init_haar(self):
        """Test Haar Cascade detector initialization."""
        detector = FaceDetector(method='haar')
        assert detector.method == 'haar'
        assert detector.detector is not None

    @pytest.mark.skipif(
        not pytest.importorskip("batch_face", reason="batch-face not available"),
        reason="RetinaFace requires batch-face package"
    )
    def test_init_retinaface(self):
        """Test RetinaFace detector initialization."""
        detector = FaceDetector(method='retinaface')
        assert detector.method == 'retinaface'
        assert detector.detector is not None

    def test_invalid_method(self):
        """Test invalid detection method raises error."""
        with pytest.raises(ValueError):
            FaceDetector(method='invalid')

    def test_detect_faces_synthetic(self):
        """Test face detection on synthetic image."""
        # Create a simple synthetic face-like image
        img = np.zeros((200, 200, 3), dtype=np.uint8)

        # Add some face-like features (very basic)
        # Oval shape for face
        cv2.ellipse(img, (100, 100), (40, 50), 0, 0, 360, (255, 255, 255), -1)
        # Eyes
        cv2.circle(img, (85, 90), 5, (0, 0, 0), -1)
        cv2.circle(img, (115, 90), 5, (0, 0, 0), -1)
        # Mouth
        cv2.ellipse(img, (100, 120), (15, 8), 0, 0, 180, (0, 0, 0), 2)

        detector = FaceDetector(method='haar')
        faces = detector.detect_faces(img)

        # Should detect at least one face (though detection quality varies)
        assert isinstance(faces, list)
        for face in faces:
            assert len(face) == 4  # (x, y, w, h)
            assert all(isinstance(coord, int) for coord in face)

    def test_get_largest_face(self):
        """Test getting largest face from multiple detections."""
        detector = FaceDetector(method='haar')

        # Create test image
        img = np.zeros((300, 300, 3), dtype=np.uint8)

        # Add two face-like regions of different sizes
        cv2.ellipse(img, (100, 100), (30, 40), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(img, (200, 200), (40, 50), 0, 0, 360, (255, 255, 255), -1)

        largest_face = detector.get_largest_face(img)

        if largest_face is not None:
            assert len(largest_face) == 4
            assert all(isinstance(coord, int) for coord in largest_face)

    def test_expand_bbox(self):
        """Test bounding box expansion."""
        detector = FaceDetector(method='haar')
        bbox = (50, 50, 100, 100)  # (x, y, w, h)

        # Test expansion
        expanded = detector.expand_bbox(bbox, scale=1.5)
        assert len(expanded) == 4

        # Expanded bbox should be larger
        assert expanded[2] > bbox[2]  # width larger
        assert expanded[3] > bbox[3]  # height larger

        # Center should remain roughly the same
        orig_cx = bbox[0] + bbox[2] // 2
        orig_cy = bbox[1] + bbox[3] // 2
        exp_cx = expanded[0] + expanded[2] // 2
        exp_cy = expanded[1] + expanded[3] // 2
        assert abs(orig_cx - exp_cx) <= 1
        assert abs(orig_cy - exp_cy) <= 1

    def test_make_square_bbox(self):
        """Test converting bbox to square."""
        detector = FaceDetector(method='haar')

        # Test rectangular bbox (wider than tall)
        bbox = (50, 50, 100, 80)
        square = detector.make_square_bbox(bbox)

        assert len(square) == 4
        assert square[2] == square[3]  # width == height (square)
        assert square[2] == 100  # Should use larger dimension

        # Center should remain the same
        orig_cx = bbox[0] + bbox[2] // 2
        orig_cy = bbox[1] + bbox[3] // 2
        sq_cx = square[0] + square[2] // 2
        sq_cy = square[1] + square[3] // 2
        assert orig_cx == sq_cx
        assert orig_cy == sq_cy

    def test_expand_bbox_frame_clipping(self):
        """Test bbox expansion with frame boundary clipping."""
        detector = FaceDetector(method='haar')
        frame_shape = (200, 300)  # (height, width)

        # Test bbox near edge
        bbox = (250, 150, 100, 100)  # Near right edge
        expanded = detector.expand_bbox(bbox, scale=1.3, frame_shape=frame_shape)

        # Should not go beyond frame boundaries
        assert expanded[0] >= 0  # x >= 0
        assert expanded[1] >= 0  # y >= 0
        assert expanded[0] + expanded[2] <= frame_shape[1]  # x + w <= width
        assert expanded[1] + expanded[3] <= frame_shape[0]  # y + h <= height