import pytest
from pathlib import Path
from src.video.synchronization import (
    get_audio_duration,
    calculate_duration_mismatch,
    adjust_segment_timing,
    calculate_speed_factor
)


class TestSynchronization:
    @pytest.fixture
    def sample_audio(self):
        audio_path = "data/temp/Tanzania_audio.wav"
        if not Path(audio_path).exists():
            from src.video.extractor import extract_audio
            extract_audio("data/input/Tanzania.mp4", audio_path)
        return audio_path

    @pytest.fixture
    def sample_segments(self):
        return [
            {'id': 0, 'start': 0.0, 'end': 5.0, 'text': 'First segment'},
            {'id': 1, 'start': 5.0, 'end': 10.0, 'text': 'Second segment'},
            {'id': 2, 'start': 10.0, 'end': 15.0, 'text': 'Third segment'},
        ]

    def test_get_audio_duration(self, sample_audio):
        duration = get_audio_duration(sample_audio)

        assert duration > 0
        assert isinstance(duration, float)
        assert 59 < duration < 62  # Tanzania video is ~60 seconds

    def test_calculate_duration_mismatch_no_change(self):
        result = calculate_duration_mismatch(60.0, 60.0)

        assert result['original_duration'] == 60.0
        assert result['new_duration'] == 60.0
        assert result['difference'] == 0.0
        assert result['percentage'] == 0.0
        assert result['needs_adjustment'] == False

    def test_calculate_duration_mismatch_small_change(self):
        result = calculate_duration_mismatch(60.0, 62.0)

        assert result['difference'] == 2.0
        assert abs(result['percentage'] - 3.33) < 0.01
        assert result['needs_adjustment'] == False  # <5%

    def test_calculate_duration_mismatch_large_change(self):
        result = calculate_duration_mismatch(60.0, 70.0)

        assert result['difference'] == 10.0
        assert abs(result['percentage'] - 16.67) < 0.01
        assert result['needs_adjustment'] == True  # >5%

    def test_adjust_segment_timing_no_change(self, sample_segments):
        adjusted = adjust_segment_timing(sample_segments, 15.0, 15.0)

        assert len(adjusted) == len(sample_segments)
        for i, seg in enumerate(adjusted):
            assert seg['start'] == sample_segments[i]['start']
            assert seg['end'] == sample_segments[i]['end']

    def test_adjust_segment_timing_stretched(self, sample_segments):
        # Audio is 20% longer
        adjusted = adjust_segment_timing(sample_segments, 15.0, 18.0)

        assert len(adjusted) == len(sample_segments)

        # First segment: 0-5s should become 0-6s (20% longer)
        assert adjusted[0]['start'] == 0.0
        assert adjusted[0]['end'] == 6.0
        assert adjusted[0]['original_start'] == 0.0
        assert adjusted[0]['original_end'] == 5.0

        # Second segment: 5-10s should become 6-12s
        assert adjusted[1]['start'] == 6.0
        assert adjusted[1]['end'] == 12.0

    def test_calculate_speed_factor_normal(self):
        factor = calculate_speed_factor(60.0, 60.0)
        assert factor == 1.0

    def test_calculate_speed_factor_slower(self):
        # New audio is longer, need to slow down video
        factor = calculate_speed_factor(60.0, 70.0)
        assert abs(factor - 0.857) < 0.001

    def test_calculate_speed_factor_faster(self):
        # New audio is shorter, need to speed up video
        factor = calculate_speed_factor(60.0, 50.0)
        assert factor == 1.2
