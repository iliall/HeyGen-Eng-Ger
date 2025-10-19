from elevenlabs import VoiceSettings, Voice, clone, generate, save, set_api_key
from pathlib import Path
from typing import List, Dict, Optional
import os
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