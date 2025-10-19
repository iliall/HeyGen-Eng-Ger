import pytest
from pathlib import Path
import os
from dotenv import load_dotenv
from src.audio.synthesis import synthesize_speech, synthesize_segments, merge_audio_segments

load_dotenv()


class TestAudioSynthesis:
    @pytest.fixture
    def output_dir(self):
        return "data/temp/synthesis_test"

    @pytest.fixture
    def sample_segments(self):
        return [
            {
                'id': 0,
                'start': 0.0,
                'end': 5.0,
                'text': 'Guten Morgen'
            },
            {
                'id': 1,
                'start': 5.0,
                'end': 10.0,
                'text': 'Wie geht es dir?'
            }
        ]

    @pytest.mark.skipif(
        not os.getenv("ELEVENLABS_API_KEY"),
        reason="ELEVENLABS_API_KEY not set"
    )
    def test_synthesize_speech(self, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = f"{output_dir}/test_speech.mp3"

        # Using pre-made voice "Rachel" to save API credits
        voice_id = "21m00Tcm4TlvDq8ikWAM"
        text = "Guten Morgen"

        result = synthesize_speech(
            text=text,
            voice_id=voice_id,
            output_path=output_path,
            model="eleven_multilingual_v2"
        )

        assert result == output_path
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

    @pytest.mark.skipif(
        not os.getenv("ELEVENLABS_API_KEY"),
        reason="ELEVENLABS_API_KEY not set"
    )
    def test_synthesize_segments(self, sample_segments, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Using pre-made voice "Rachel"
        voice_id = "21m00Tcm4TlvDq8ikWAM"

        audio_files = synthesize_segments(
            segments=sample_segments,
            voice_id=voice_id,
            output_dir=output_dir
        )

        assert len(audio_files) == len(sample_segments)

        for audio_file in audio_files:
            assert Path(audio_file).exists()
            assert Path(audio_file).stat().st_size > 0

    @pytest.mark.skipif(
        not os.getenv("ELEVENLABS_API_KEY"),
        reason="ELEVENLABS_API_KEY not set"
    )
    def test_merge_audio_segments(self, sample_segments, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        voice_id = "21m00Tcm4TlvDq8ikWAM"

        # Generate segments
        audio_files = synthesize_segments(
            segments=sample_segments,
            voice_id=voice_id,
            output_dir=output_dir
        )

        # Merge them
        merged_path = f"{output_dir}/merged.wav"
        result = merge_audio_segments(audio_files, merged_path)

        assert Path(result).exists()
        assert Path(result).stat().st_size > 0
        # Merged file should be larger than individual segments
        assert Path(result).stat().st_size > Path(audio_files[0]).stat().st_size