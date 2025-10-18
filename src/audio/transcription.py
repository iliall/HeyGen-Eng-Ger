import whisper
from typing import Dict, List, Optional
from pathlib import Path
import json


def transcribe_audio(audio_path: str, model_size: str = "base",
                     language: str = "en") -> Dict:
    """
    Transcribe audio file to text using Whisper.

    Args:
        audio_path: Path to audio file
        model_size: Whisper model size (tiny, base, small, medium, large)
        language: Source language code

    Returns:
        Dictionary containing transcription and segments with timestamps
    """
    model = whisper.load_model(model_size)

    result = model.transcribe(
        audio_path,
        language=language,
        task="transcribe",
        verbose=False
    )

    return result


def get_segments(transcription_result: Dict) -> List[Dict]:
    """
    Extract segments with timestamps from transcription result.

    Args:
        transcription_result: Result from transcribe_audio()

    Returns:
        List of segments with text, start, and end times
    """
    segments = []

    for segment in transcription_result.get('segments', []):
        segments.append({
            'id': segment.get('id'),
            'start': segment.get('start'),
            'end': segment.get('end'),
            'text': segment.get('text').strip()
        })

    return segments


def save_transcription(transcription_result: Dict, output_path: str) -> None:
    """
    Save transcription result to JSON file.

    Args:
        transcription_result: Result from transcribe_audio()
        output_path: Path for output JSON file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(transcription_result, f, ensure_ascii=False, indent=2)


def load_transcription(json_path: str) -> Dict:
    """
    Load transcription from JSON file.

    Args:
        json_path: Path to JSON file

    Returns:
        Transcription dictionary
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)