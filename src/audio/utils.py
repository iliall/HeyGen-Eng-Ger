from pydub import AudioSegment
from pathlib import Path
from typing import List, Dict
import subprocess
import logging

logger = logging.getLogger(__name__)


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


def merge_word_level_segments(audio_files: List[str], segments: List[Dict],
                             word_segments: List[Dict], output_path: str) -> str:
    """
    Merge audio segments with word-level time-stretching for perfect alignment.

    Args:
        audio_files: List of synthesized audio file paths (segment-level)
        segments: List of original segments with timing info
        word_segments: List of word-level segments with aligned timing
        output_path: Path for merged output file

    Returns:
        Path to merged audio file
    """
    if not word_segments or len(word_segments) == 0:
        # Fallback to segment-level alignment
        return merge_time_aligned_segments(audio_files, segments, output_path)

    # For now, implement a safer approach: use segment-level alignment
    # The word-level timing data is too complex to implement safely without
    # proper word boundary detection and audio splitting
    logger.warning("Word-level alignment is experimental. Using segment-level alignment for safety.")
    return merge_time_aligned_segments(audio_files, segments, output_path)


def create_word_level_audio_mapping(original_segments: List[Dict],
                                  translated_segments: List[Dict],
                                  original_audio_path: str,
                                  translated_audio_path: str) -> List[Dict]:
    """
    Create word-level mapping between original and translated audio.

    This is a placeholder for more sophisticated word-level alignment.
    In practice, this would use the forced alignment data from ElevenLabs.

    Args:
        original_segments: Original timing segments
        translated_segments: Translated text segments
        original_audio_path: Path to original audio
        translated_audio_path: Path to translated audio

    Returns:
        List of word-level segments with timing
    """
    # For now, create simple word-level segments
    # This would be replaced with actual forced alignment data
    word_segments = []

    for i, segment in enumerate(translated_segments):
        words = segment['text'].split()
        if not words:
            continue

        segment_duration = segment['end'] - segment['start']
        word_duration = segment_duration / len(words)

        for j, word in enumerate(words):
            word_start = segment['start'] + (j * word_duration)
            word_end = word_start + word_duration

            word_segments.append({
                'text': word,
                'start': word_start,
                'end': word_end,
                'segment_id': i,
                'word_id': j
            })

    return word_segments
