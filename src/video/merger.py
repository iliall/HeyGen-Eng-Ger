import ffmpeg
from pathlib import Path
from typing import Optional


def remove_audio(video_path: str, output_path: Optional[str] = None) -> str:
    """
    Remove audio from video file.

    Args:
        video_path: Path to input video file
        output_path: Path for output video file (optional)

    Returns:
        Path to video file without audio
    """
    video_file = Path(video_path)

    if output_path is None:
        output_path = video_file.parent / f"{video_file.stem}_no_audio{video_file.suffix}"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    stream = ffmpeg.input(str(video_path))
    stream = ffmpeg.output(stream.video, str(output_path), vcodec='copy')
    ffmpeg.run(stream, overwrite_output=True, quiet=True)

    return str(output_path)


def replace_audio(video_path: str, audio_path: str, output_path: str,
                 video_codec: str = "copy", audio_codec: str = "aac") -> str:
    """
    Replace video's audio track with new audio.

    Args:
        video_path: Path to input video file
        audio_path: Path to new audio file
        output_path: Path for output video file
        video_codec: Video codec (use 'copy' to avoid re-encoding)
        audio_codec: Audio codec for output

    Returns:
        Path to output video file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    video_stream = ffmpeg.input(str(video_path)).video
    audio_stream = ffmpeg.input(str(audio_path)).audio

    stream = ffmpeg.output(
        video_stream,
        audio_stream,
        str(output_path),
        vcodec=video_codec,
        acodec=audio_codec,
        shortest=None
    )

    ffmpeg.run(stream, overwrite_output=True, quiet=True)

    return str(output_path)


def merge_audio_video(video_path: str, audio_path: str, output_path: str) -> str:
    """
    Merge audio and video into single file.

    Args:
        video_path: Path to video file (can have or not have audio)
        audio_path: Path to audio file
        output_path: Path for output video file

    Returns:
        Path to merged video file
    """
    return replace_audio(video_path, audio_path, output_path)
