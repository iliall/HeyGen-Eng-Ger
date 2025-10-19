import pytest
from pathlib import Path
from src.video.merger import remove_audio, replace_audio, merge_audio_video
from src.video.extractor import get_video_duration, get_audio_info


class TestVideoMerger:
    @pytest.fixture
    def sample_video(self):
        return "data/input/Tanzania.mp4"

    @pytest.fixture
    def sample_audio(self):
        return "data/temp/Tanzania_audio.wav"

    @pytest.fixture
    def output_dir(self):
        return "data/temp/merger_test"

    def test_remove_audio(self, sample_video, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = f"{output_dir}/no_audio.mp4"

        result = remove_audio(sample_video, output_path)

        assert Path(result).exists()
        assert result == output_path

        # Verify video has no audio
        try:
            audio_info = get_audio_info(result)
            # If we get here, video still has audio (shouldn't happen)
            pytest.fail("Video should not have audio track")
        except:
            # Expected - no audio track found
            pass

    def test_replace_audio(self, sample_video, sample_audio, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = f"{output_dir}/replaced_audio.mp4"

        # First extract audio if it doesn't exist
        if not Path(sample_audio).exists():
            from src.video.extractor import extract_audio
            extract_audio(sample_video, sample_audio)

        result = replace_audio(sample_video, sample_audio, output_path)

        assert Path(result).exists()
        assert result == output_path

        # Verify output has audio
        audio_info = get_audio_info(result)
        assert audio_info['codec'] is not None
        assert audio_info['sample_rate'] > 0

    def test_merge_audio_video(self, sample_video, sample_audio, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = f"{output_dir}/merged.mp4"

        # First extract audio if it doesn't exist
        if not Path(sample_audio).exists():
            from src.video.extractor import extract_audio
            extract_audio(sample_video, sample_audio)

        result = merge_audio_video(sample_video, sample_audio, output_path)

        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

        # Verify video duration is preserved
        original_duration = get_video_duration(sample_video)
        merged_duration = get_video_duration(result)

        # Duration should be roughly the same (within 1 second)
        assert abs(merged_duration - original_duration) < 1.0
