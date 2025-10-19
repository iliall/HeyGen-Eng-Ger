from elevenlabs import VoiceSettings, Voice, clone, generate, save, set_api_key
from pathlib import Path
from typing import List, Dict, Optional
import os
import requests
import json
from pydub import AudioSegment


def setup_elevenlabs():
    """Initialize ElevenLabs API with key from environment."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY not found in environment")
    set_api_key(api_key)


def prepare_voice_samples(audio_path: str, segments: List[Dict],
                         output_dir: str, max_samples: int = 3) -> List[str]:
    """
    Extract voice samples from audio for cloning.

    Selects the longest segments to get a good voice sample.

    Args:
        audio_path: Path to original audio file
        segments: List of transcription segments with timing
        output_dir: Directory to save voice samples
        max_samples: Maximum number of samples to extract

    Returns:
        List of paths to voice sample files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    audio = AudioSegment.from_file(audio_path)

    # Sort segments by duration (longest first)
    sorted_segments = sorted(segments, key=lambda s: s['end'] - s['start'], reverse=True)

    # Take the longest segments (up to max_samples)
    selected_segments = sorted_segments[:max_samples]

    sample_files = []

    for i, segment in enumerate(selected_segments):
        start_ms = int(segment['start'] * 1000)
        end_ms = int(segment['end'] * 1000)

        sample = audio[start_ms:end_ms]

        sample_path = output_path / f"voice_sample_{i:02d}.mp3"
        sample.export(str(sample_path), format="mp3")
        sample_files.append(str(sample_path))

    return sample_files


def clone_voice(name: str, audio_files: List[str], description: str = "") -> str:
    """
    Clone a voice from audio samples.

    Args:
        name: Name for the cloned voice
        audio_files: List of paths to audio files for voice cloning
        description: Description of the voice

    Returns:
        Voice ID of cloned voice
    """
    setup_elevenlabs()

    voice = clone(
        name=name,
        description=description,
        files=audio_files
    )

    return voice.voice_id


def synthesize_speech(text: str, voice_id: str, output_path: str,
                     model: str = "eleven_multilingual_v2",
                     stability: float = 0.5,
                     similarity_boost: float = 0.8,
                     style: float = 0.4,
                     use_speaker_boost: bool = True,
                     output_format: str = "mp3_44100_192") -> str:
    """
    Generate speech from text using cloned voice.

    Args:
        text: Text to convert to speech
        voice_id: ID of voice to use
        output_path: Path for output audio file
        model: ElevenLabs model to use
        stability: Voice stability (0-1). Lower = more emotional, Higher = more consistent
        similarity_boost: How closely to match cloned voice (0-1)
        style: Style exaggeration (0-1). Amplifies original speaker's style
        use_speaker_boost: Boost similarity to original speaker
        output_format: Audio output format (mp3_44100_192 = 192kbps, highest quality)

    Returns:
        Path to generated audio file
    """
    setup_elevenlabs()

    voice_settings = VoiceSettings(
        stability=stability,
        similarity_boost=similarity_boost,
        style=style,
        use_speaker_boost=use_speaker_boost
    )

    voice = Voice(
        voice_id=voice_id,
        settings=voice_settings
    )

    audio = generate(
        text=text,
        voice=voice,
        model=model,
        output_format=output_format
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    save(audio, str(output_path))

    return str(output_path)


def synthesize_segments(segments: List[Dict], voice_id: str,
                       output_dir: str, model: str = "eleven_multilingual_v2",
                       stability: float = 0.5,
                       similarity_boost: float = 0.8,
                       style: float = 0.4,
                       use_speaker_boost: bool = True,
                       output_format: str = "mp3_44100_192") -> List[str]:
    """
    Generate speech for each segment.

    Args:
        segments: List of segments with translated text
        voice_id: ID of voice to use
        output_dir: Directory for output audio files
        model: ElevenLabs model to use
        stability: Voice stability (0-1)
        similarity_boost: How closely to match cloned voice (0-1)
        style: Style exaggeration (0-1)
        use_speaker_boost: Boost similarity to original speaker
        output_format: Audio output format (mp3_44100_192 = highest quality)

    Returns:
        List of paths to generated audio files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    audio_files = []

    for i, segment in enumerate(segments):
        file_path = output_path / f"segment_{i:04d}.mp3"
        synthesize_speech(
            text=segment['text'],
            voice_id=voice_id,
            output_path=str(file_path),
            model=model,
            stability=stability,
            similarity_boost=similarity_boost,
            style=style,
            use_speaker_boost=use_speaker_boost,
            output_format=output_format
        )
        audio_files.append(str(file_path))

    return audio_files


def merge_audio_segments(audio_files: List[str], output_path: str) -> str:
    """
    Merge multiple audio files into single file.

    Args:
        audio_files: List of audio file paths
        output_path: Path for merged output file

    Returns:
        Path to merged audio file
    """
    combined = AudioSegment.empty()

    for audio_file in audio_files:
        segment = AudioSegment.from_file(audio_file)
        combined += segment

    combined.export(output_path, format="wav")

    return output_path


def get_forced_alignment(audio_path: str, text: str) -> Dict:
    """
    Get forced alignment data for audio and text using ElevenLabs API.

    Args:
        audio_path: Path to audio file
        text: Transcript text to align with audio

    Returns:
        Dictionary containing alignment data with words and characters
        {
            "characters": [{"text": "string", "start": 0.0, "end": 0.0}],
            "words": [{"text": "string", "start": 0.0, "end": 0.0, "loss": 0.0}],
            "loss": 0.0
        }
    """
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY not found in environment")

    url = "https://api.elevenlabs.io/v1/forced-alignment"
    headers = {"xi-api-key": api_key}

    try:
        with open(audio_path, 'rb') as audio_file:
            files = {
                'text': (None, text),
                'file': (audio_path, audio_file, 'audio/mpeg')
            }

            response = requests.post(url, headers=headers, files=files)
            response.raise_for_status()

            return response.json()

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Forced alignment API call failed: {e}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse alignment response: {e}")


def align_translated_words(original_alignment: Dict, translated_text: str,
                          original_text: str) -> List[Dict]:
    """
    Map translated words to original timing using word alignment.

    Args:
        original_alignment: Alignment data from original audio
        translated_text: German translated text
        original_text: English original text

    Returns:
        List of aligned word segments with timing
        [{"text": "word", "start": 0.0, "end": 0.0, "original_word": "word"}, ...]
    """
    original_words = original_alignment.get('words', [])

    # Simple word count mapping - this is a basic implementation
    # In a more sophisticated version, we could use translation alignment tools
    original_word_list = [w['text'] for w in original_words]
    translated_word_list = translated_text.split()

    if len(translated_word_list) != len(original_word_list):
        # Fallback: use proportional timing if word counts don't match
        return align_by_proportional_timing(original_words, translated_text)

    aligned_words = []
    for i, (orig_word, trans_word) in enumerate(zip(original_word_list, translated_word_list)):
        if i < len(original_words):
            aligned_words.append({
                'text': trans_word,
                'start': original_words[i]['start'],
                'end': original_words[i]['end'],
                'original_word': orig_word
            })

    return aligned_words


def align_by_proportional_timing(original_words: List[Dict], translated_text: str) -> List[Dict]:
    """
    Fallback alignment using proportional timing when word counts don't match.

    Args:
        original_words: List of original word timing data
        translated_text: Translated text

    Returns:
        List of aligned word segments with proportional timing
    """
    translated_words = translated_text.split()

    if not original_words:
        return []

    total_duration = original_words[-1]['end'] - original_words[0]['start']
    start_time = original_words[0]['start']

    aligned_words = []
    total_original_chars = sum(len(w['text']) + 1 for w in original_words)  # +1 for spaces
    total_translated_chars = len(translated_text)

    current_time = start_time
    char_position = 0

    for word in translated_words:
        word_chars = len(word) + 1  # +1 for space
        proportion = word_chars / total_translated_chars
        word_duration = proportion * total_duration

        aligned_words.append({
            'text': word,
            'start': current_time,
            'end': current_time + word_duration,
            'original_word': ''
        })

        current_time += word_duration

    return aligned_words


def create_word_level_segments(aligned_words: List[Dict]) -> List[Dict]:
    """
    Convert aligned words to segment format for processing.

    Args:
        aligned_words: List of aligned word data

    Returns:
        List of segments in standard format
        [{"id": 0, "start": 0.0, "end": 0.5, "text": "word"}, ...]
    """
    segments = []
    for i, word in enumerate(aligned_words):
        segments.append({
            'id': i,
            'start': word['start'],
            'end': word['end'],
            'text': word['text']
        })

    return segments