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


def merge_segments(segments: List[Dict], min_words: int = 5) -> List[Dict]:
    """
    Merge segments that have too few words.

    This improves audio quality by reducing the number of time-stretching
    operations. Small segments with few words sound unnatural when stretched.

    Simple algorithm: If segment has <= min_words, merge with next segment.
    Keep merging forward until we have > min_words.

    Args:
        segments: List of segments with 'start', 'end', 'text'
        min_words: Minimum words per segment (default: 5)

    Returns:
        List of merged segments with updated IDs

    Example:
        Before: [{"text": "Hello", "start": 0.0, "end": 0.5},
                 {"text": "world", "start": 0.6, "end": 1.0}]
        After:  [{"text": "Hello world", "start": 0.0, "end": 1.0}]
    """
    if not segments:
        return []

    merged = []
    i = 0

    while i < len(segments):
        current = segments[i].copy()

        # Keep merging with next segments until we have > min_words
        while len(current['text'].split()) <= min_words and i + 1 < len(segments):
            i += 1
            next_seg = segments[i]
            current['text'] = current['text'].strip() + " " + next_seg['text'].strip()
            current['end'] = next_seg['end']

        merged.append(current)
        i += 1

    # Update IDs to be sequential
    for idx, seg in enumerate(merged):
        seg['id'] = idx

    return merged