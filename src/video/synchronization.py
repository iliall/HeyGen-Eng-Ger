from pydub import AudioSegment
from pathlib import Path
from typing import List, Dict, Tuple
import subprocess


def get_audio_duration(audio_path: str) -> float:
    """
    Get duration of audio file in seconds.

    Args:
        audio_path: Path to audio file

    Returns:
        Duration in seconds
    """
    audio = AudioSegment.from_file(audio_path)
    return len(audio) / 1000.0


def calculate_duration_mismatch(original_duration: float, new_duration: float) -> Dict:
    """
    Calculate timing mismatch between original and new audio.

    Args:
        original_duration: Original audio duration in seconds
        new_duration: New audio duration in seconds

    Returns:
        Dictionary with mismatch info
    """
    difference = new_duration - original_duration
    percentage = (difference / original_duration) * 100 if original_duration > 0 else 0

    return {
        'original_duration': original_duration,
        'new_duration': new_duration,
        'difference': difference,
        'percentage': percentage,
        'needs_adjustment': abs(percentage) > 5.0  # Adjust if >5% difference
    }


def time_stretch_audio(audio_path: str, output_path: str, target_duration: float) -> str:
    """
    Stretch or compress audio to match target duration using rubberband.

    Args:
        audio_path: Path to input audio file
        output_path: Path for output audio file
        target_duration: Target duration in seconds

    Returns:
        Path to time-stretched audio file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    current_duration = get_audio_duration(audio_path)
    time_ratio = target_duration / current_duration

    # Use rubberband for time-stretching (preserves pitch)
    cmd = [
        'rubberband',
        '-t', str(time_ratio),  # Time ratio
        audio_path,
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError:
        raise RuntimeError("rubberband not found. Install with: brew install rubberband")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"rubberband failed: {e.stderr.decode()}")

    return str(output_path)


def adjust_segment_timing(segments: List[Dict], original_duration: float,
                         new_duration: float) -> List[Dict]:
    """
    Adjust segment timestamps proportionally based on duration change.

    Args:
        segments: List of segments with start/end times
        original_duration: Original total duration
        new_duration: New total duration

    Returns:
        List of segments with adjusted timing
    """
    if original_duration == 0:
        return segments

    time_ratio = new_duration / original_duration
    adjusted_segments = []

    for segment in segments:
        adjusted = segment.copy()
        adjusted['start'] = segment['start'] * time_ratio
        adjusted['end'] = segment['end'] * time_ratio
        adjusted['original_start'] = segment['start']
        adjusted['original_end'] = segment['end']
        adjusted_segments.append(adjusted)

    return adjusted_segments


def calculate_speed_factor(original_duration: float, new_duration: float) -> float:
    """
    Calculate video speed adjustment factor.

    Args:
        original_duration: Original duration in seconds
        new_duration: New duration in seconds

    Returns:
        Speed factor (1.0 = normal, <1.0 = slower, >1.0 = faster)
    """
    return original_duration / new_duration
