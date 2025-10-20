"""Integration tests for the lip-sync pipeline."""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from src.lipsync.detector import FaceDetector
from src.lipsync.preprocessor import (
    load_audio, extract_frames, get_video_info,
    crop_face_region, prepare_face_for_wav2lip
)
from src.lipsync.utils import get_mel_spectrogram, smooth_bbox_sequence
from src.lipsync.model_manager import get_model_path, check_models_exist


class TestLipsyncIntegration:
    """Integration tests for the complete lip-sync pipeline."""

    @pytest.fixture
    def temp_video_file(self):
        """Create a temporary video file for testing."""
        # Create a simple synthetic video
        import cv2

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            temp_path = f.name

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, 25.0, (320, 240))

        # Generate simple frames with a moving circle
        for i in range(50):  # 2 seconds at 25 fps
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            # Add a circle that moves (simulating face motion)
            center_x = 160 + int(30 * np.sin(i * 0.1))
            center_y = 120 + int(20 * np.cos(i * 0.1))
            cv2.circle(frame, (center_x, center_y), 40, (255, 255, 255), -1)
            # Add some features to make it more face-like
            cv2.circle(frame, (center_x - 10, center_y - 10), 5, (0, 0, 0), -1)  # eye
            cv2.circle(frame, (center_x + 10, center_y - 10), 5, (0, 0, 0), -1)  # eye
            cv2.ellipse(frame, (center_x, center_y + 15), (15, 8), 0, 0, 180, (0, 0, 0), 2)  # mouth
            out.write(frame)

        out.release()
        yield temp_path

        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def temp_audio_file(self):
        """Create a temporary audio file for testing."""
        import librosa

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name

        # Generate simple test audio (sine wave)
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

        # Save as WAV
        librosa.output.write_wav(temp_path, audio, sr)
        yield temp_path

        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_get_video_info(self, temp_video_file):
        """Test video info extraction."""
        info = get_video_info(temp_video_file)

        assert 'fps' in info
        assert 'frame_count' in info
        assert 'width' in info
        assert 'height' in info
        assert 'duration' in info

        assert info['width'] == 320
        assert info['height'] == 240
        assert info['fps'] == 25.0
        assert info['frame_count'] == 50
        assert abs(info['duration'] - 2.0) < 0.1

    def test_extract_frames(self, temp_video_file):
        """Test frame extraction."""
        frames, fps = extract_frames(temp_video_file)

        assert len(frames) == 50
        assert fps == 25.0
        assert all(frame.shape == (240, 320, 3) for frame in frames)
        assert all(frame.dtype == np.uint8 for frame in frames)

    def test_load_audio(self, temp_audio_file):
        """Test audio loading."""
        audio = load_audio(temp_audio_file, target_sr=16000)

        assert len(audio) == 32000  # 2 seconds * 16000 Hz
        assert audio.dtype == np.float32 or audio.dtype == np.float64

    def test_face_detection_pipeline(self, temp_video_file):
        """Test complete face detection pipeline."""
        # Extract frames
        frames, fps = extract_frames(temp_video_file)

        # Initialize detector
        detector = FaceDetector(method='haar')

        # Detect faces in all frames
        face_bboxes = []
        for i, frame in enumerate(frames):
            bbox = detector.get_largest_face(frame)
            if bbox is None:
                pytest.skip(f"No face detected in frame {i}")
            face_bboxes.append(bbox)

        assert len(face_bboxes) == len(frames)

        # Smooth bboxes
        smoothed_bboxes = smooth_bbox_sequence(face_bboxes)
        assert len(smoothed_bboxes) == len(face_bboxes)

        # Test bbox expansion and squaring
        for bbox in smoothed_bboxes:
            expanded = detector.expand_bbox(bbox, scale=1.3, frame_shape=frames[0].shape[:2])
            square = detector.make_square_bbox(expanded, frame_shape=frames[0].shape[:2])

            assert len(expanded) == 4
            assert len(square) == 4
            assert square[2] == square[3]  # Should be square

    def test_face_preprocessing_pipeline(self, temp_video_file):
        """Test face preprocessing pipeline."""
        frames, fps = extract_frames(temp_video_file)
        detector = FaceDetector(method='haar')

        # Get first frame with detected face
        for frame in frames:
            bbox = detector.get_largest_face(frame)
            if bbox is not None:
                # Expand and square bbox
                bbox_exp = detector.expand_bbox(bbox, scale=1.3, frame_shape=frame.shape[:2])
                bbox_square = detector.make_square_bbox(bbox_exp, frame_shape=frame.shape[:2])

                # Crop and prepare face
                face = crop_face_region(frame, bbox_square)
                face_prep = prepare_face_for_wav2lip(face)

                assert face.shape[:2] == bbox_square[2:]  # Cropped to bbox size
                assert face_prep.shape == (96, 96, 3)  # Wav2Lip input size
                assert face_prep.dtype == np.float32
                assert 0.0 <= face_prep.max() <= 1.0
                assert 0.0 <= face_prep.min() <= 1.0
                break
        else:
            pytest.skip("No faces detected in any frame")

    def test_mel_spectrogram_pipeline(self, temp_audio_file):
        """Test mel spectrogram computation pipeline."""
        # Load audio
        audio = load_audio(temp_audio_file, target_sr=16000)

        # Compute mel spectrogram
        mel = get_mel_spectrogram(audio)

        assert mel.shape[0] == 80  # n_mels
        assert mel.shape[1] > 0    # time frames
        assert mel.dtype in [np.float32, np.float64]

        # Test mel chunking (simulating what would happen in full pipeline)
        mel_step_size = 16
        num_frames = 50  # Simulating 2 seconds at 25 fps

        mel_chunks = []
        for i in range(num_frames):
            start_idx = i * mel_step_size
            end_idx = start_idx + mel_step_size * 2

            if end_idx > mel.shape[1]:
                pad_width = end_idx - mel.shape[1]
                mel_chunk = np.pad(mel[:, start_idx:], ((0, 0), (0, pad_width)), mode='constant')
            else:
                mel_chunk = mel[:, start_idx:end_idx]

            mel_chunks.append(mel_chunk)

        assert len(mel_chunks) == num_frames
        assert all(chunk.shape[0] == 80 for chunk in mel_chunks)

    def test_model_paths(self):
        """Test model path management."""
        # Test that model paths are correctly generated
        wav2lip_path = get_model_path('wav2lip')
        s3fd_path = get_model_path('s3fd')

        assert wav2lip_path.name == 'wav2lip.pth'
        assert s3fd_path.name == 's3fd.pth'
        assert wav2lip_path.parent == s3fd_path.parent

        # Test invalid model name
        with pytest.raises(ValueError):
            get_model_path('invalid_model')

    @pytest.mark.skipif(not check_models_exist(), reason="Models not downloaded")
    def test_full_pipeline_components(self, temp_video_file, temp_audio_file):
        """Test all components working together (requires downloaded models)."""
        # Extract frames
        frames, fps = extract_frames(temp_video_file)
        audio = load_audio(temp_audio_file, target_sr=16000)

        # Detect faces
        detector = FaceDetector(method='haar')
        face_bboxes = []
        for frame in frames:
            bbox = detector.get_largest_face(frame)
            if bbox is None:
                bbox = face_bboxes[-1] if face_bboxes else (0, 0, 100, 100)
            face_bboxes.append(bbox)

        # Smooth bboxes
        face_bboxes = smooth_bbox_sequence(face_bboxes)

        # Prepare face frames
        face_frames = []
        for frame, bbox in zip(frames, face_bboxes):
            bbox_exp = detector.expand_bbox(bbox, scale=1.3, frame_shape=frame.shape[:2])
            bbox_square = detector.make_square_bbox(bbox_exp, frame_shape=frame.shape[:2])
            face = crop_face_region(frame, bbox_square)
            face_prep = prepare_face_for_wav2lip(face)
            face_frames.append(face_prep)

        # Compute mel spectrogram and chunks
        mel = get_mel_spectrogram(audio)
        mel_step_size = 16

        mel_chunks = []
        for i in range(len(frames)):
            start_idx = i * mel_step_size
            end_idx = start_idx + mel_step_size * 2

            if end_idx > mel.shape[1]:
                pad_width = end_idx - mel.shape[1]
                mel_chunk = np.pad(mel[:, start_idx:], ((0, 0), (0, pad_width)), mode='constant')
            else:
                mel_chunk = mel[:, start_idx:end_idx]

            mel_chunks.append(mel_chunk)

        # Verify all components are working
        assert len(face_frames) == len(frames)
        assert len(mel_chunks) == len(frames)
        assert all(face.shape == (96, 96, 3) for face in face_frames)
        assert all(chunk.shape[0] == 80 for chunk in mel_chunks)

    def test_error_handling(self):
        """Test error handling in the pipeline."""
        # Test with non-existent video file
        with pytest.raises(RuntimeError):
            get_video_info('non_existent_file.mp4')

        # Test with non-existent audio file
        with pytest.raises(RuntimeError):
            load_audio('non_existent_file.wav')