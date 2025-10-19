import pytest
import requests
from unittest.mock import patch, mock_open, MagicMock
from src.audio.synthesis import (
    get_forced_alignment,
    align_translated_words,
    align_by_proportional_timing,
    create_word_level_segments
)


class TestForcedAlignment:
    """Test ElevenLabs forced alignment functionality."""

    @pytest.fixture
    def mock_alignment_response(self):
        """Mock forced alignment API response."""
        return {
            "characters": [
                {"text": "H", "start": 0.0, "end": 0.1},
                {"text": "e", "start": 0.1, "end": 0.2},
                {"text": "l", "start": 0.2, "end": 0.3},
                {"text": "l", "start": 0.3, "end": 0.4},
                {"text": "o", "start": 0.4, "end": 0.5},
                {"text": " ", "start": 0.5, "end": 0.6},
                {"text": "w", "start": 0.6, "end": 0.7},
                {"text": "o", "start": 0.7, "end": 0.8},
                {"text": "r", "start": 0.8, "end": 0.9},
                {"text": "l", "start": 0.9, "end": 1.0},
                {"text": "d", "start": 1.0, "end": 1.1}
            ],
            "words": [
                {"text": "Hello", "start": 0.0, "end": 0.5, "loss": 0.01},
                {"text": "world", "start": 0.6, "end": 1.1, "loss": 0.02}
            ],
            "loss": 0.015
        }

    @patch('src.audio.synthesis.os.getenv')
    @patch('src.audio.synthesis.requests.post')
    @patch('builtins.open', new_callable=mock_open, read_data=b"fake audio data")
    def test_get_forced_alignment_success(self, mock_file, mock_post, mock_getenv, mock_alignment_response):
        """Test successful forced alignment API call."""
        # Setup mocks
        mock_getenv.return_value = "test_api_key"
        mock_response = MagicMock()
        mock_response.json.return_value = mock_alignment_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Test function
        result = get_forced_alignment("test_audio.mp3", "Hello world")

        # Verify API call
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args

        assert args[0] == "https://api.elevenlabs.io/v1/forced-alignment"
        assert kwargs['headers']['xi-api-key'] == "test_api_key"
        assert 'text' in kwargs['files']
        assert 'file' in kwargs['files']

        # Verify result
        assert result == mock_alignment_response
        assert len(result['words']) == 2
        assert result['words'][0]['text'] == "Hello"

    @patch('src.audio.synthesis.os.getenv')
    def test_get_forced_alignment_no_api_key(self, mock_getenv):
        """Test forced alignment fails without API key."""
        mock_getenv.return_value = None

        with pytest.raises(ValueError, match="ELEVENLABS_API_KEY not found"):
            get_forced_alignment("test_audio.mp3", "Hello world")

    @patch('src.audio.synthesis.os.getenv')
    @patch('src.audio.synthesis.requests.post')
    @patch('builtins.open', new_callable=mock_open, read_data=b"fake audio data")
    def test_get_forced_alignment_api_error(self, mock_file, mock_post, mock_getenv):
        """Test forced alignment handles API errors gracefully."""
        mock_getenv.return_value = "test_api_key"
        mock_post.side_effect = requests.exceptions.RequestException("API Error")

        with pytest.raises(RuntimeError, match="Forced alignment API call failed"):
            get_forced_alignment("test_audio.mp3", "Hello world")


class TestWordAlignment:
    """Test word alignment functionality."""

    @pytest.fixture
    def sample_alignment(self):
        """Sample alignment data for testing."""
        return {
            "words": [
                {"text": "Hello", "start": 0.0, "end": 0.5, "loss": 0.01},
                {"text": "world", "start": 0.6, "end": 1.1, "loss": 0.02},
                {"text": "this", "start": 1.2, "end": 1.6, "loss": 0.01},
                {"text": "is", "start": 1.7, "end": 1.9, "loss": 0.01},
                {"text": "test", "start": 2.0, "end": 2.5, "loss": 0.02}
            ]
        }

    def test_align_translated_words_equal_count(self, sample_alignment):
        """Test word alignment when word counts match."""
        original_text = "Hello world this is test"
        translated_text = "Hallo Welt das ist Test"

        result = align_translated_words(sample_alignment, translated_text, original_text)

        assert len(result) == 5
        assert result[0]['text'] == "Hallo"
        assert result[0]['start'] == 0.0
        assert result[0]['end'] == 0.5
        assert result[0]['original_word'] == "Hello"

        assert result[4]['text'] == "Test"
        assert result[4]['start'] == 2.0
        assert result[4]['end'] == 2.5

    def test_align_translated_words_different_count(self, sample_alignment):
        """Test word alignment fallback when word counts don't match."""
        original_text = "Hello world this is test"
        translated_text = "Hallo Welt Test"  # 3 words vs 5 original

        result = align_translated_words(sample_alignment, translated_text, original_text)

        assert len(result) == 3
        assert result[0]['text'] == "Hallo"
        assert result[1]['text'] == "Welt"
        assert result[2]['text'] == "Test"

    def test_align_by_proportional_timing(self, sample_alignment):
        """Test proportional timing alignment."""
        translated_text = "Hallo Welt Test"

        result = align_by_proportional_timing(sample_alignment['words'], translated_text)

        assert len(result) == 3
        assert result[0]['text'] == "Hallo"
        assert result[1]['text'] == "Welt"
        assert result[2]['text'] == "Test"

        # Check timing is proportional
        assert result[0]['start'] == 0.0
        assert result[2]['end'] <= 2.8  # Allow some tolerance for proportional calculation

    def test_align_by_proportional_timing_empty_words(self):
        """Test proportional timing with empty original words."""
        result = align_by_proportional_timing([], "Hallo Welt")
        assert result == []

    def test_create_word_level_segments(self):
        """Test creation of word-level segments."""
        aligned_words = [
            {"text": "Hallo", "start": 0.0, "end": 0.5, "original_word": "Hello"},
            {"text": "Welt", "start": 0.6, "end": 1.1, "original_word": "world"}
        ]

        result = create_word_level_segments(aligned_words)

        assert len(result) == 2
        assert result[0]['id'] == 0
        assert result[0]['text'] == "Hallo"
        assert result[0]['start'] == 0.0
        assert result[0]['end'] == 0.5

        assert result[1]['id'] == 1
        assert result[1]['text'] == "Welt"
        assert result[1]['start'] == 0.6
        assert result[1]['end'] == 1.1


class TestWordLevelIntegration:
    """Test integration scenarios for word-level alignment."""

    def test_complete_alignment_workflow(self):
        """Test complete word alignment workflow."""
        # Mock alignment data
        alignment_data = {
            "words": [
                {"text": "Tanzania", "start": 0.0, "end": 0.8, "loss": 0.01},
                {"text": "home", "start": 0.9, "end": 1.2, "loss": 0.02},
                {"text": "wildlife", "start": 1.3, "end": 2.0, "loss": 0.01}
            ]
        }

        original_text = "Tanzania home wildlife"
        translated_text = "Tansania Heimat Tierwelt"

        # Step 1: Align words
        aligned_words = align_translated_words(alignment_data, translated_text, original_text)

        # Step 2: Create segments
        segments = create_word_level_segments(aligned_words)

        # Verify workflow
        assert len(aligned_words) == 3
        assert aligned_words[0]['text'] == "Tansania"
        assert aligned_words[0]['original_word'] == "Tanzania"

        assert len(segments) == 3
        assert segments[0]['text'] == "Tansania"
        assert segments[0]['id'] == 0

    def test_alignment_with_complex_text(self):
        """Test alignment with more complex text."""
        alignment_data = {
            "words": [
                {"text": "The", "start": 0.0, "end": 0.1, "loss": 0.01},
                {"text": "Great", "start": 0.2, "end": 0.6, "loss": 0.01},
                {"text": "Migration", "start": 0.7, "end": 1.4, "loss": 0.02},
                {"text": "is", "start": 1.5, "end": 1.7, "loss": 0.01},
                {"text": "amazing", "start": 1.8, "end": 2.5, "loss": 0.01}
            ]
        }

        original_text = "The Great Migration is amazing"
        translated_text = "Die Große Völkerwanderung ist erstaunlich"

        result = align_translated_words(alignment_data, translated_text, original_text)

        # Should fall back to proportional timing due to word count mismatch
        # Note: German compounds "Große" and "Völkerwanderung" might be treated as single words
        assert len(result) >= 5  # At least 5 segments
        assert "Die" in result[0]['text']
        assert any("erstaunlich" in r['text'] for r in result)


if __name__ == "__main__":
    pytest.main([__file__])