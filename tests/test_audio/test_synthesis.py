import pytest
from pathlib import Path
import os
from dotenv import load_dotenv
from src.audio.synthesis import synthesize_speech, synthesize_segments, merge_audio_segments, prepare_voice_samples, clone_voice

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

    def test_prepare_voice_samples(self, output_dir):
        audio_path = "data/temp/Tanzania_audio.wav"
        if not Path(audio_path).exists():
            pytest.skip("Tanzania audio not available")

        segments = [
            {'id': 0, 'start': 0.0, 'end': 3.5, 'text': 'Short'},
            {'id': 1, 'start': 3.5, 'end': 10.0, 'text': 'Longer segment'},
            {'id': 2, 'start': 10.0, 'end': 20.0, 'text': 'Longest segment'},
            {'id': 3, 'start': 20.0, 'end': 22.0, 'text': 'Short again'},
        ]

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        voice_samples_dir = f"{output_dir}/voice_samples"

        samples = prepare_voice_samples(
            audio_path=audio_path,
            segments=segments,
            output_dir=voice_samples_dir,
            max_samples=3
        )

        assert len(samples) == 3
        for sample in samples:
            assert Path(sample).exists()
            assert Path(sample).stat().st_size > 0

    @pytest.mark.skipif(
        not os.getenv("ELEVENLABS_API_KEY"),
        reason="ELEVENLABS_API_KEY not set"
    )
    def test_clone_voice(self, output_dir):
        audio_path = "data/temp/Tanzania_audio.wav"
        if not Path(audio_path).exists():
            pytest.skip("Tanzania audio not available")

        segments = [
            {'id': 0, 'start': 0.0, 'end': 5.0, 'text': 'First'},
            {'id': 1, 'start': 5.0, 'end': 10.0, 'text': 'Second'},
        ]

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        voice_samples_dir = f"{output_dir}/voice_samples_clone"

        samples = prepare_voice_samples(
            audio_path=audio_path,
            segments=segments,
            output_dir=voice_samples_dir,
            max_samples=2
        )

        voice_id = clone_voice(
            name="test_voice_clone",
            audio_files=samples,
            description="Test voice cloning"
        )

        assert voice_id is not None
        assert isinstance(voice_id, str)
        assert len(voice_id) > 0