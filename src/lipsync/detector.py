"""Face detection for lip-sync processing."""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

try:
    from batch_face import RetinaFace
    BATCH_FACE_AVAILABLE = True
except ImportError:
    BATCH_FACE_AVAILABLE = False
    logger.warning("batch_face not available, using OpenCV Haar Cascades")


class FaceDetector:
    """Detect and track faces in video frames."""

    def __init__(self, method: str = 'auto'):
        """
        Initialize face detector.

        Args:
            method: Detection method ('auto', 'retinaface', 'haar')
                   'auto' tries RetinaFace first, falls back to Haar
        """
        self.method = method
        self.detector = None

        if method == 'auto':
            if BATCH_FACE_AVAILABLE:
                self._init_retinaface()
            else:
                self._init_haar()
        elif method == 'retinaface':
            self._init_retinaface()
        elif method == 'haar':
            self._init_haar()
        else:
            raise ValueError(f"Unknown detection method: {method}")

    def _init_retinaface(self):
        """Initialize RetinaFace detector."""
        if not BATCH_FACE_AVAILABLE:
            raise ImportError("batch_face is not installed")

        try:
            self.detector = RetinaFace()
            self.method = 'retinaface'
            logger.info("Using RetinaFace for face detection")
        except Exception as e:
            logger.warning(f"Failed to initialize RetinaFace: {e}, falling back to Haar")
            self._init_haar()

    def _init_haar(self):
        """Initialize OpenCV Haar Cascade detector."""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)

        if self.detector.empty():
            raise RuntimeError("Failed to load Haar Cascade classifier")

        self.method = 'haar'
        logger.info("Using Haar Cascade for face detection")

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a single frame.

        Args:
            frame: Video frame as numpy array (BGR format)

        Returns:
            List of bounding boxes [(x, y, w, h), ...]
            Sorted by size (largest first)
        """
        if self.method == 'retinaface':
            return self._detect_retinaface(frame)
        elif self.method == 'haar':
            return self._detect_haar(frame)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _detect_retinaface(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using RetinaFace."""
        try:
            # RetinaFace expects RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.detector(rgb_frame)

            if faces is None or len(faces) == 0:
                return []

            # Convert to (x, y, w, h) format
            boxes = []
            for face in faces:
                bbox = face[:4]  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, bbox)
                w = x2 - x1
                h = y2 - y1
                boxes.append((x1, y1, w, h))

            # Sort by area (largest first)
            boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
            return boxes

        except Exception as e:
            logger.error(f"RetinaFace detection failed: {e}")
            return []

    def _detect_haar(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using Haar Cascade."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            return []

        # Sort by area (largest first)
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]

    def get_largest_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Get the largest face in the frame.

        Args:
            frame: Video frame

        Returns:
            Bounding box (x, y, w, h) or None if no face detected
        """
        faces = self.detect_faces(frame)
        return faces[0] if faces else None

    def expand_bbox(self, bbox: Tuple[int, int, int, int],
                    scale: float = 1.3,
                    frame_shape: Tuple[int, int] = None) -> Tuple[int, int, int, int]:
        """
        Expand bounding box by a scale factor.

        Useful for including more context around the face.

        Args:
            bbox: Original bounding box (x, y, w, h)
            scale: Scale factor (1.3 = 30% larger)
            frame_shape: (height, width) to clip to frame bounds

        Returns:
            Expanded bounding box (x, y, w, h)
        """
        x, y, w, h = bbox

        # Calculate center
        cx = x + w // 2
        cy = y + h // 2

        # Scale dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Calculate new top-left
        new_x = cx - new_w // 2
        new_y = cy - new_h // 2

        # Clip to frame bounds if provided
        if frame_shape is not None:
            height, width = frame_shape
            new_x = max(0, min(new_x, width - new_w))
            new_y = max(0, min(new_y, height - new_h))
            new_w = min(new_w, width - new_x)
            new_h = min(new_h, height - new_y)

        return (new_x, new_y, new_w, new_h)

    def make_square_bbox(self, bbox: Tuple[int, int, int, int],
                        frame_shape: Tuple[int, int] = None) -> Tuple[int, int, int, int]:
        """
        Convert bounding box to square (required for Wav2Lip).

        Args:
            bbox: Original bounding box (x, y, w, h)
            frame_shape: (height, width) to clip to frame bounds

        Returns:
            Square bounding box (x, y, size, size)
        """
        x, y, w, h = bbox

        # Use larger dimension as size
        size = max(w, h)

        # Center the square on the face
        cx = x + w // 2
        cy = y + h // 2

        new_x = cx - size // 2
        new_y = cy - size // 2

        # Clip to frame bounds if provided
        if frame_shape is not None:
            height, width = frame_shape
            new_x = max(0, min(new_x, width - size))
            new_y = max(0, min(new_y, height - size))
            size = min(size, width - new_x, height - new_y)

        return (new_x, new_y, size, size)
