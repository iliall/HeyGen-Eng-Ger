import pytest
from pathlib import Path
from src.audio.transcription import transcribe_audio, get_segments, save_transcription, load_transcription


class TestAudioTranscription:
    @pytest.fixture
    def audio_file(self):
        return "data/temp/Tanzania_audio.wav"

    @pytest.fixture
    def output_json(self):
        return "data/temp/transcription.json"

    def test_transcribe_audio(self, audio_file):
        result = transcribe_audio(audio_file, model_size="base", language="en")

        assert result is not None
        assert 'text' in result
        assert 'segments' in result
        assert len(result['text']) > 0
        assert len(result['segments']) > 0

    def test_get_segments(self, audio_file):
        transcription = transcribe_audio(audio_file, model_size="base", language="en")
        segments = get_segments(transcription)

        assert len(segments) > 0

        for segment in segments:
            assert 'id' in segment
            assert 'start' in segment
            assert 'end' in segment
            assert 'text' in segment
            assert segment['start'] < segment['end']

    def test_save_and_load_transcription(self, audio_file, output_json):
        transcription = transcribe_audio(audio_file, model_size="base", language="en")

        save_transcription(transcription, output_json)
        assert Path(output_json).exists()

        loaded = load_transcription(output_json)
        assert loaded['text'] == transcription['text']
        assert len(loaded['segments']) == len(transcription['segments'])