import ffmpeg
from pathlib import Path
from typing import Optional


def extract_audio(video_path: str, output_path: Optional[str] = None,
                  sample_rate: int = 44100, audio_format: str = "wav") -> str:
    """
    Extract audio stream from video file.

    Args:
        video_path: Path to input video file
        output_path: Path for output audio file (optional)
        sample_rate: Audio sample rate in Hz
        audio_format: Output audio format (wav, mp3, etc.)

    Returns:
        Path to extracted audio file
    """
    video_file = Path(video_path)

    if output_path is None:
        output_path = video_file.parent / f"{video_file.stem}_audio.{audio_format}"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    stream = ffmpeg.input(str(video_path))
    stream = ffmpeg.output(stream.audio, str(output_path),
                          acodec='pcm_s16le' if audio_format == 'wav' else None,
                          ar=sample_rate,
                          ac=1)

    ffmpeg.run(stream, overwrite_output=True, quiet=True)

    return str(output_path)


def get_video_duration(video_path: str) -> float:
    """
    Get duration of video file in seconds.

    Args:
        video_path: Path to video file

    Returns:
        Duration in seconds
    """
    probe = ffmpeg.probe(video_path)
    duration = float(probe['format']['duration'])
    return duration


def get_audio_info(video_path: str) -> dict:
    """
    Get audio stream information from video file.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with audio stream info (codec, sample_rate, channels)
    """
    probe = ffmpeg.probe(video_path)

    audio_streams = [s for s in probe['streams'] if s['codec_type'] == 'audio']

    if not audio_streams:
        raise ValueError(f"No audio stream found in {video_path}")

    audio_stream = audio_streams[0]

    return {
        'codec': audio_stream.get('codec_name'),
        'sample_rate': int(audio_stream.get('sample_rate', 0)),
        'channels': audio_stream.get('channels'),
        'duration': float(audio_stream.get('duration', 0))
    }