"""
Lip-sync module for video translation.

This module provides visual lip-sync using Wav2Lip to adjust mouth movements
to match translated audio.
"""

from .model_manager import (
    download_model,
    download_all_models,
    check_models_exist,
    get_model_path
)
from .api import apply_lipsync
from .synthesizer import Wav2LipSynthesizer
from .detector import FaceDetector
from .preprocessor import load_audio, extract_frames, get_video_info

__all__ = [
    # Model management
    'download_model',
    'download_all_models',
    'check_models_exist',
    'get_model_path',
    # Lip-sync
    'apply_lipsync',
    'Wav2LipSynthesizer',
    'FaceDetector',
    # Preprocessing
    'load_audio',
    'extract_frames',
    'get_video_info',
]
