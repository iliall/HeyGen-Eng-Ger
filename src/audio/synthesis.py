from elevenlabs import VoiceSettings, clone, generate, save, set_api_key
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
                     model: str = "eleven_multilingual_v2") -> str:
    """
    Generate speech from text using cloned voice.

    Args:
        text: Text to convert to speech
        voice_id: ID of voice to use
        output_path: Path for output audio file
        model: ElevenLabs model to use

    Returns:
        Path to generated audio file
    """
    setup_elevenlabs()

    audio = generate(
        text=text,
        voice=voice_id,
        model=model
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    save(audio, str(output_path))

    return str(output_path)


def synthesize_segments(segments: List[Dict], voice_id: str,
                       output_dir: str, model: str = "eleven_multilingual_v2") -> List[str]:
    """
    Generate speech for each segment.

    Args:
        segments: List of segments with translated text
        voice_id: ID of voice to use
        output_dir: Directory for output audio files
        model: ElevenLabs model to use

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
            model=model
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