import pytest
from src.audio.translation import translate_text, translate_segments, get_full_translation


class TestTranslation:
    @pytest.fixture
    def sample_segments(self):
        return [
            {
                'id': 0,
                'start': 0.0,
                'end': 4.68,
                'text': 'Tanzania, home to some of the most breathtaking wildlife on Earth.'
            },
            {
                'id': 1,
                'start': 4.68,
                'end': 9.68,
                'text': 'Here in the heart of East Africa, the Great Serengeti National Park.'
            },
            {
                'id': 2,
                'start': 9.68,
                'end': 13.84,
                'text': 'Lions, the kings of the savanna, stalk their prey.'
            }
        ]

    def test_translate_text_google(self):
        text = "Hello, how are you?"
        translated = translate_text(text, source_lang="en", target_lang="de", service="google")

        assert translated is not None
        assert len(translated) > 0
        assert translated != text
        assert isinstance(translated, str)

    def test_translate_segments(self, sample_segments):
        translated = translate_segments(sample_segments, source_lang="en", target_lang="de", service="google")

        assert len(translated) == len(sample_segments)

        for i, segment in enumerate(translated):
            assert 'id' in segment
            assert 'start' in segment
            assert 'end' in segment
            assert 'text' in segment
            assert 'original_text' in segment

            assert segment['id'] == sample_segments[i]['id']
            assert segment['start'] == sample_segments[i]['start']
            assert segment['end'] == sample_segments[i]['end']

            assert segment['original_text'] == sample_segments[i]['text']
            assert segment['text'] != sample_segments[i]['text']
            assert len(segment['text']) > 0

    def test_translate_segments_preserves_timing(self, sample_segments):
        translated = translate_segments(sample_segments, source_lang="en", target_lang="de")

        for i in range(len(translated)):
            assert translated[i]['start'] == sample_segments[i]['start']
            assert translated[i]['end'] == sample_segments[i]['end']

    def test_get_full_translation(self, sample_segments):
        translated = translate_segments(sample_segments, source_lang="en", target_lang="de")
        full_text = get_full_translation(translated)

        assert isinstance(full_text, str)
        assert len(full_text) > 0

        for segment in translated:
            assert segment['text'] in full_text

    def test_translate_text_different_languages(self):
        text = "Good morning"

        german = translate_text(text, source_lang="en", target_lang="de")
        assert german != text

        spanish = translate_text(text, source_lang="en", target_lang="es")
        assert spanish != text
        assert spanish != german