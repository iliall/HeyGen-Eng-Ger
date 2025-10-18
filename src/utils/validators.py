from pathlib import Path


SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
SUPPORTED_AUDIO_FORMATS = ['.mp3', '.wav', '.m4a', '.flac']


def validate_video_file(path: str) -> bool:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    if file_path.suffix.lower() not in SUPPORTED_VIDEO_FORMATS:
        raise ValueError(f"Unsupported video format: {file_path.suffix}")

    return True


def validate_audio_file(path: str) -> bool:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if file_path.suffix.lower() not in SUPPORTED_AUDIO_FORMATS:
        raise ValueError(f"Unsupported audio format: {file_path.suffix}")

    return True


def validate_output_dir(path: str) -> bool:
    dir_path = Path(path)

    if dir_path.exists() and not dir_path.is_dir():
        raise ValueError(f"Output path exists but is not a directory: {path}")

    return True