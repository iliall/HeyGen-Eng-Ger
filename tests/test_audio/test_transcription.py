import pytest
from pathlib import Path
from src.audio.transcription import transcribe_audio, get_segments, save_transcription, load_transcription, merge_segments


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


class TestSegmentMerging:
    """Test segment merging functionality."""

    def test_merge_segments_empty_list(self):
        """Test merging with empty segment list."""
        result = merge_segments([])
        assert result == []

    def test_merge_segments_single_segment(self):
        """Test merging with single segment."""
        segments = [{"id": 0, "start": 0.0, "end": 1.0, "text": "Hello world"}]
        result = merge_segments(segments)
        assert len(result) == 1
        assert result[0]['text'] == "Hello world"

    def test_merge_short_segments(self):
        """Test merging segments with <= 5 words."""
        segments = [
            {"id": 0, "start": 0.0, "end": 0.5, "text": "Hi"},
            {"id": 1, "start": 0.6, "end": 1.0, "text": "there"},
            {"id": 2, "start": 1.5, "end": 3.0, "text": "friend"}
        ]
        result = merge_segments(segments, min_words=5)

        # All three should merge because each has <= 5 words
        assert len(result) == 1
        assert result[0]['text'] == "Hi there friend"
        assert result[0]['start'] == 0.0
        assert result[0]['end'] == 3.0

    def test_merge_until_enough_words(self):
        """Test merging continues until segment has > 5 words."""
        segments = [
            {"id": 0, "start": 0.0, "end": 1.0, "text": "One"},
            {"id": 1, "start": 1.0, "end": 2.0, "text": "two"},
            {"id": 2, "start": 2.0, "end": 3.0, "text": "three"},
            {"id": 3, "start": 3.0, "end": 4.0, "text": "four"},
            {"id": 4, "start": 4.0, "end": 5.0, "text": "five"},
            {"id": 5, "start": 5.0, "end": 6.0, "text": "six"},
            {"id": 6, "start": 6.0, "end": 7.0, "text": "This has many words now"}
        ]
        result = merge_segments(segments, min_words=5)

        # First 6 segments merge to get "One two three four five six" (6 words > 5)
        # Last segment already has > 5 words
        assert len(result) == 2
        assert result[0]['text'] == "One two three four five six"
        assert result[1]['text'] == "This has many words now"

    def test_no_merge_long_segments(self):
        """Test that segments with > 5 words don't merge."""
        segments = [
            {"id": 0, "start": 0.0, "end": 2.0, "text": "This is a segment with many words"},
            {"id": 1, "start": 3.0, "end": 5.0, "text": "This is another segment with many words"}
        ]
        result = merge_segments(segments, min_words=5)

        # Should not merge because both have > 5 words
        assert len(result) == 2
        assert result[0]['text'] == "This is a segment with many words"
        assert result[1]['text'] == "This is another segment with many words"

    def test_update_ids_sequential(self):
        """Test that merged segments have sequential IDs."""
        segments = [
            {"id": 0, "start": 0.0, "end": 1.0, "text": "One"},
            {"id": 1, "start": 1.0, "end": 2.0, "text": "two"},
            {"id": 2, "start": 2.0, "end": 3.0, "text": "This has many words in it"}
        ]
        result = merge_segments(segments, min_words=5)

        for i, segment in enumerate(result):
            assert segment['id'] == i

    def test_custom_min_words(self):
        """Test segment merging with custom min_words parameter."""
        segments = [
            {"id": 0, "start": 0.0, "end": 1.0, "text": "One two"},
            {"id": 1, "start": 1.0, "end": 2.0, "text": "Three four"}
        ]

        # With min_words=3, both segments have 2 words (≤ 3), so merge
        result = merge_segments(segments, min_words=3)
        assert len(result) == 1
        assert result[0]['text'] == "One two Three four"

        # With min_words=2, first segment has 2 words (≤ 2) but still merges
        result = merge_segments(segments, min_words=2)
        assert len(result) == 1

    def test_preserve_timing(self):
        """Test that timing is preserved correctly after merging."""
        segments = [
            {"id": 0, "start": 1.5, "end": 2.5, "text": "Hello"},
            {"id": 1, "start": 2.6, "end": 3.5, "text": "world"}
        ]
        result = merge_segments(segments, min_words=5)

        assert len(result) == 1
        assert result[0]['start'] == 1.5  # Start from first segment
        assert result[0]['end'] == 3.5    # End from last merged segment

    def test_last_segment_short(self):
        """Test when last segment has <= 5 words (edge case)."""
        segments = [
            {"id": 0, "start": 0.0, "end": 1.0, "text": "This is a long segment with many words"},
            {"id": 1, "start": 1.0, "end": 2.0, "text": "Short"}
        ]
        result = merge_segments(segments, min_words=5)

        # First segment has > 5 words, stays separate
        # Last segment has 1 word but no next segment to merge with
        assert len(result) == 2
        assert result[0]['text'] == "This is a long segment with many words"
        assert result[1]['text'] == "Short"