"""Audio source separation for preserving background audio."""

import os
import logging
from pathlib import Path
from typing import Tuple, Optional
import tempfile
import shutil
import numpy as np

logger = logging.getLogger(__name__)

# Check if demucs is available
try:
    import torch
    import torchaudio
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False
    logger.warning("Demucs not available. Install with: pip install demucs")


def apply_simple_noise_gate(audio_tensor, sr: int, threshold: float = 0.01) -> torch.Tensor:
    """
    Apply a simple noise gate to reduce low-level noise and artifacts.

    Args:
        audio_tensor: Audio tensor [channels, samples]
        sr: Sample rate
        threshold: Noise gate threshold (0.0-1.0)

    Returns:
        Denoised audio tensor
    """
    # Import scipy only when needed to avoid dependency issues
    try:
        from scipy import signal
        use_smoothing = True
    except ImportError:
        use_smoothing = False

    # Convert to numpy for processing
    audio_np = audio_tensor.numpy()

    # Calculate RMS energy for each channel
    if len(audio_np.shape) == 2:
        # Stereo: process each channel separately
        for ch in range(audio_np.shape[0]):
            channel = audio_np[ch]
            # Simple gate: reduce (not zero) audio below threshold
            gate_mask = np.where(np.abs(channel) > threshold, 1.0, 0.1)
            audio_np[ch] = channel * gate_mask
    else:
        # Mono
        gate_mask = np.where(np.abs(audio_np) > threshold, 1.0, 0.1)
        audio_np = audio_np * gate_mask

    # Apply gentle smoothing to reduce hard cuts
    if use_smoothing:
        window_size = int(sr * 0.01)  # 10ms window
        if window_size > 0:
            smooth_window = np.ones(window_size) / window_size
            if len(audio_np.shape) == 2:
                for ch in range(audio_np.shape[0]):
                    audio_np[ch] = signal.convolve(audio_np[ch], smooth_window, mode='same')
            else:
                audio_np = signal.convolve(audio_np, smooth_window, mode='same')

    return torch.from_numpy(audio_np)


class AudioSeparator:
    """Separate audio into vocals and background using Demucs."""

    def __init__(self, model_name: str = "htdemucs", use_enhancement: bool = True):
        """
        Initialize audio separator.

        Args:
            model_name: Demucs model to use:
                - "htdemucs" (default): Hybrid Transformer Demucs, best quality
                - "htdemucs_ft": Fine-tuned version, even better
                - "mdx_extra": Alternative model
            use_enhancement: Apply noise reduction and quality improvements to background
        """
        if not DEMUCS_AVAILABLE:
            raise ImportError(
                "Demucs is not installed. Install with: pip install demucs"
            )

        self.model_name = model_name
        self.use_enhancement = use_enhancement
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Note: MPS (Apple Silicon) has channel limitations, so we use CPU for Demucs
        # This ensures compatibility at the cost of speed

        logger.info(f"Audio separator initialized with model: {model_name}, device: {self.device}, enhancement: {use_enhancement}")

    def _load_model(self):
        """Lazy load the Demucs model."""
        if self.model is None:
            logger.info(f"Loading Demucs model: {self.model_name}")
            self.model = get_model(self.model_name)
            self.model.to(self.device)
            logger.info("Model loaded successfully")

    def separate(
        self,
        audio_path: str,
        output_dir: str,
        shifts: int = 1,
        overlap: float = 0.25
    ) -> Tuple[str, str]:
        """
        Separate audio into vocals and background (music/ambient).

        Args:
            audio_path: Path to input audio file
            output_dir: Directory to save separated audio files
            shifts: Number of random shifts for separation (higher=better, slower)
                   Default: 1 (fast), recommended: 5-10 (best quality)
            overlap: Overlap between chunks (0.25 = 25%)

        Returns:
            Tuple of (vocals_path, background_path)
        """
        self._load_model()

        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Separating audio: {audio_path.name}")
        logger.info(f"Using {shifts} shifts with {overlap} overlap")

        # Load audio using soundfile (avoids torchcodec dependency)
        import subprocess
        import soundfile as sf

        logger.info("Loading audio with soundfile...")

        # Always use soundfile for compatibility
        try:
            audio_data, sr = sf.read(str(audio_path), always_2d=True)
            # soundfile with always_2d=True returns [samples, channels]
            # We need [channels, samples] for PyTorch
            wav = torch.from_numpy(audio_data.T).float()
        except Exception as e:
            logger.warning(f"Soundfile loading failed: {e}")
            logger.info("Converting audio with FFmpeg first...")

            # Fallback: Convert with FFmpeg first
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name

            try:
                subprocess.run([
                    'ffmpeg', '-i', str(audio_path),
                    '-ar', '44100',  # Standard sample rate
                    '-ac', '2',      # Stereo
                    '-y', tmp_path
                ], check=True, capture_output=True)

                # Load with soundfile
                audio_data, sr = sf.read(tmp_path, always_2d=True)
                # Convert to tensor [channels, samples]
                wav = torch.from_numpy(audio_data.T).float()
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        wav = wav.to(self.device)

        # Ensure stereo (Demucs expects 2 channels)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2]

        # Add batch dimension
        wav = wav.unsqueeze(0)

        # Apply model
        logger.info("Running source separation...")
        with torch.no_grad():
            sources = apply_model(
                self.model,
                wav,
                shifts=shifts,
                overlap=overlap,
                device=self.device
            )

        # Extract sources
        # Demucs htdemucs outputs 4 stems: drums, bass, other, vocals
        sources = sources[0]  # Remove batch dimension

        # Get source indices
        source_names = self.model.sources
        vocals_idx = source_names.index("vocals")

        # Extract vocals
        vocals = sources[vocals_idx].cpu()

        # Combine all non-vocal sources as background with better quality handling
        background_indices = [i for i in range(len(source_names)) if i != vocals_idx]

        # Instead of simple summation, use weighted average to prevent clipping
        background_sources = [sources[i] for i in background_indices]
        background = torch.stack(background_sources, dim=0).mean(dim=0).cpu()

        # Apply gentle noise reduction to reduce separation artifacts if enabled
        if self.use_enhancement:
            background = apply_simple_noise_gate(background, sr=sr, threshold=0.01)
            logger.info("  Applied background enhancement (noise gate)")
        else:
            logger.info("  Background enhancement disabled")

        # Save separated audio using soundfile (avoids torchcodec)
        import soundfile as sf

        vocals_path = output_dir / f"{audio_path.stem}_vocals.wav"
        background_path = output_dir / f"{audio_path.stem}_background.wav"

        logger.info(f"Saving vocals to: {vocals_path}")
        # Convert from [channels, samples] to [samples, channels] for soundfile
        vocals_np = vocals.numpy().T
        sf.write(str(vocals_path), vocals_np, sr)

        logger.info(f"Saving background to: {background_path}")
        background_np = background.numpy().T
        sf.write(str(background_path), background_np, sr)

        logger.info("Audio separation complete")

        return str(vocals_path), str(background_path)


def separate_audio(
    audio_path: str,
    output_dir: str,
    model: str = "htdemucs",
    quality: str = "balanced",
    use_background_enhancement: bool = True
) -> Optional[Tuple[str, str]]:
    """
    Convenience function to separate audio into vocals and background.

    Args:
        audio_path: Path to input audio file
        output_dir: Directory to save separated files
        model: Demucs model name (default: "htdemucs")
        quality: Quality preset:
            - "fast": 1 shift (quick separation)
            - "balanced": 5 shifts (good quality/speed tradeoff)
            - "best": 10 shifts (best quality, slowest)
        use_background_enhancement: Apply noise reduction and quality improvements

    Returns:
        Tuple of (vocals_path, background_path) or None if demucs unavailable
    """
    if not DEMUCS_AVAILABLE:
        logger.warning("Demucs not available, skipping audio separation")
        return None

    # Quality presets
    shifts_map = {
        "fast": 1,
        "balanced": 5,
        "best": 10
    }
    shifts = shifts_map.get(quality, 5)

    separator = AudioSeparator(model_name=model, use_enhancement=use_background_enhancement)
    return separator.separate(
        audio_path=audio_path,
        output_dir=output_dir,
        shifts=shifts
    )


def is_separation_available() -> bool:
    """Check if audio separation is available."""
    return DEMUCS_AVAILABLE
