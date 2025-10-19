import pytest
from pathlib import Path
from src.audio.utils import time_stretch_segment, merge_time_aligned_segments
from src.video.synchronization import get_audio_duration
from pydub import AudioSegment


class TestAudioUtils:
    @pytest.fixture
    def sample_audio_segment(self):
        segments_dir = Path("data/temp/segments")
        if segments_dir.exists():
            audio_files = sorted(segments_dir.glob("segment_*.mp3"))
            if audio_files:
                return str(audio_files[0])

        from src.video.extractor import extract_audio
        audio_path = "data/temp/Tanzania_audio.wav"
        if not Path(audio_path).exists():
            extract_audio("data/input/Tanzania.mp4", audio_path)
        return audio_path

    @pytest.fixture
    def sample_segments_with_timing(self):
        return [
            {'id': 0, 'start': 0.0, 'end': 3.0, 'text': 'First segment', 'translated_text': 'Erstes Segment'},
            {'id': 1, 'start': 3.0, 'end': 6.0, 'text': 'Second segment', 'translated_text': 'Zweites Segment'},
            {'id': 2, 'start': 6.0, 'end': 10.0, 'text': 'Third segment', 'translated_text': 'Drittes Segment'},
        ]

    @pytest.fixture
    def output_dir(self):
        output = Path("data/temp/utils_test")
        output.mkdir(parents=True, exist_ok=True)
        return str(output)

    def test_time_stretch_segment_longer(self, sample_audio_segment, output_dir):
        output_path = f"{output_dir}/stretched_longer.wav"
        target_duration = 5.0

        result = time_stretch_segment(sample_audio_segment, output_path, target_duration)

        assert Path(result).exists()
        actual_duration = get_audio_duration(result)
        assert abs(actual_duration - target_duration) < 0.1

    def test_time_stretch_segment_shorter(self, sample_audio_segment, output_dir):
        output_path = f"{output_dir}/stretched_shorter.wav"
        target_duration = 2.0

        result = time_stretch_segment(sample_audio_segment, output_path, target_duration)

        assert Path(result).exists()
        actual_duration = get_audio_duration(result)
        assert abs(actual_duration - target_duration) < 0.1

    def test_time_stretch_segment_no_change(self, sample_audio_segment, output_dir):
        original_duration = get_audio_duration(sample_audio_segment)
        output_path = f"{output_dir}/stretched_same.wav"

        result = time_stretch_segment(sample_audio_segment, output_path, original_duration)

        assert Path(result).exists()
        actual_duration = get_audio_duration(result)
        assert abs(actual_duration - original_duration) < 0.1

    def test_merge_time_aligned_segments_basic(self, output_dir):
        segments_dir = Path("data/temp/segments")
        if not segments_dir.exists() or not any(segments_dir.glob("segment_*.mp3")):
            pytest.skip("No synthesized segments found. Run full pipeline first.")

        audio_files = sorted([str(f) for f in segments_dir.glob("segment_*.mp3")])[:3]

        segments = [
            {'id': 0, 'start': 0.0, 'end': 2.0, 'text': 'First'},
            {'id': 1, 'start': 2.0, 'end': 4.5, 'text': 'Second'},
            {'id': 2, 'start': 4.5, 'end': 7.0, 'text': 'Third'},
        ]

        output_path = f"{output_dir}/merged_aligned.wav"
        result = merge_time_aligned_segments(audio_files, segments, output_path)

        assert Path(result).exists()

        total_duration = get_audio_duration(result)
        expected_duration = segments[-1]['end']
        assert abs(total_duration - expected_duration) < 0.5

    def test_merge_time_aligned_segments_mismatch_count(self, output_dir):
        audio_files = ["file1.mp3", "file2.mp3"]
        segments = [
            {'id': 0, 'start': 0.0, 'end': 2.0},
        ]
        output_path = f"{output_dir}/merged_mismatch.wav"

        with pytest.raises(ValueError, match="Number of audio files must match"):
            merge_time_aligned_segments(audio_files, segments, output_path)

    def test_time_stretch_preserves_content(self, sample_audio_segment, output_dir):
        output_path = f"{output_dir}/stretched_preserve.wav"
        target_duration = 4.0

        original_audio = AudioSegment.from_file(sample_audio_segment)
        original_sample_rate = original_audio.frame_rate

        time_stretch_segment(sample_audio_segment, output_path, target_duration)

        stretched_audio = AudioSegment.from_file(output_path)
        stretched_sample_rate = stretched_audio.frame_rate

        assert stretched_sample_rate == original_sample_rate
        assert stretched_audio.channels == original_audio.channels
