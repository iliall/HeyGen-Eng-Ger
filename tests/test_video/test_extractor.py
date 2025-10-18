import pytest
from pathlib import Path
from src.video.extractor import extract_audio, get_video_duration, get_audio_info


class TestVideoExtractor:
    @pytest.fixture
    def sample_video(self):
        return "data/input/Tanzania.mp4"

    @pytest.fixture
    def output_path(self):
        return "data/temp/test_audio.wav"

    def test_get_video_duration(self, sample_video):
        duration = get_video_duration(sample_video)
        assert duration > 0
        assert isinstance(duration, float)

    def test_get_audio_info(self, sample_video):
        audio_info = get_audio_info(sample_video)

        assert 'codec' in audio_info
        assert 'sample_rate' in audio_info
        assert 'channels' in audio_info
        assert 'duration' in audio_info

        assert audio_info['sample_rate'] > 0
        assert audio_info['channels'] > 0

    def test_extract_audio(self, sample_video, output_path):
        extracted_path = extract_audio(sample_video, output_path)

        assert extracted_path == output_path
        assert Path(extracted_path).exists()
        assert Path(extracted_path).stat().st_size > 0

    def test_extract_audio_default_output(self, sample_video):
        extracted_path = extract_audio(sample_video)

        assert Path(extracted_path).exists()
        assert extracted_path.endswith('.wav')
        assert Path(extracted_path).stat().st_size > 0