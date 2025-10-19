from pydub import AudioSegment
from pathlib import Path
from typing import List, Dict
import subprocess


def time_stretch_segment(audio_path: str, output_path: str, target_duration: float) -> str:
    """
    Time-stretch audio segment to match target duration.

    Args:
        audio_path: Path to input audio file
        output_path: Path for output audio file
        target_duration: Target duration in seconds

    Returns:
        Path to time-stretched audio file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Get current duration
    audio = AudioSegment.from_file(audio_path)
    current_duration = len(audio) / 1000.0

    # Calculate time ratio
    time_ratio = target_duration / current_duration

    # Use rubberband for time-stretching (preserves pitch)
    cmd = [
        'rubberband',
        '-t', str(time_ratio),
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


def merge_time_aligned_segments(audio_files: List[str], segments: List[Dict],
                                output_path: str) -> str:
    """
    Merge audio segments with time-stretching to match original timing.

    Args:
        audio_files: List of synthesized audio file paths
        segments: List of segments with original timing info
        output_path: Path for merged output file

    Returns:
        Path to merged audio file
    """
    if len(audio_files) != len(segments):
        raise ValueError("Number of audio files must match number of segments")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Create temp directory for stretched segments
    temp_dir = output_file.parent / "stretched_segments"
    temp_dir.mkdir(exist_ok=True)

    stretched_files = []

    # Time-stretch each segment to match original duration
    for i, (audio_file, segment) in enumerate(zip(audio_files, segments)):
        original_duration = segment['end'] - segment['start']
        stretched_path = temp_dir / f"stretched_{i:04d}.wav"

        time_stretch_segment(audio_file, str(stretched_path), original_duration)
        stretched_files.append(str(stretched_path))

    # Concatenate stretched segments
    combined = AudioSegment.empty()
    for stretched_file in stretched_files:
        segment_audio = AudioSegment.from_file(stretched_file)
        combined += segment_audio

    # Export merged audio
    combined.export(output_path, format="wav")

    return output_path
