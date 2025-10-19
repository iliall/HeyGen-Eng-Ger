import pytest
from pathlib import Path
import tempfile
from src.audio.srt_parser import (
    srt_timestamp_to_seconds,
    seconds_to_srt_timestamp,
    strip_html_tags,
    parse_srt_file,
    validate_srt_segments,
    save_segments_as_srt
)


class TestSrtTimestamps:
    def test_srt_timestamp_to_seconds_basic(self):
        assert srt_timestamp_to_seconds("00:00:04,680") == 4.68
        assert srt_timestamp_to_seconds("00:00:00,000") == 0.0
        assert srt_timestamp_to_seconds("00:01:30,250") == 90.25

    def test_srt_timestamp_to_seconds_hours(self):
        assert srt_timestamp_to_seconds("01:00:00,000") == 3600.0
        assert srt_timestamp_to_seconds("02:30:15,500") == 9015.5

    def test_srt_timestamp_to_seconds_with_period(self):
        # Some .srt files use period instead of comma
        assert srt_timestamp_to_seconds("00:00:04.680") == 4.68

    def test_srt_timestamp_to_seconds_invalid(self):
        with pytest.raises(ValueError):
            srt_timestamp_to_seconds("invalid")
        with pytest.raises(ValueError):
            srt_timestamp_to_seconds("00:00")

    def test_seconds_to_srt_timestamp(self):
        assert seconds_to_srt_timestamp(4.68) == "00:00:04,680"
        assert seconds_to_srt_timestamp(0.0) == "00:00:00,000"
        assert seconds_to_srt_timestamp(90.25) == "00:01:30,250"
        assert seconds_to_srt_timestamp(3600.0) == "01:00:00,000"

    def test_timestamp_round_trip(self):
        # Test that conversion is reversible
        original = "00:01:30,250"
        seconds = srt_timestamp_to_seconds(original)
        result = seconds_to_srt_timestamp(seconds)
        assert result == original


class TestHtmlStripping:
    def test_strip_html_tags_basic(self):
        assert strip_html_tags("<i>Hello</i>") == "Hello"
        assert strip_html_tags("<b>Bold text</b>") == "Bold text"

    def test_strip_html_tags_multiple(self):
        assert strip_html_tags("<i>Italic</i> and <b>bold</b>") == "Italic and bold"

    def test_strip_html_tags_no_tags(self):
        assert strip_html_tags("Plain text") == "Plain text"

    def test_strip_html_tags_complex(self):
        text = '<font color="red">Red text</font>'
        assert strip_html_tags(text) == "Red text"


class TestSrtParser:
    @pytest.fixture
    def temp_srt_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def valid_srt(self, temp_srt_dir):
        srt_content = """1
00:00:00,000 --> 00:00:04,680
Tanzania, home to some of the most breathtaking wildlife

2
00:00:04,680 --> 00:00:09,680
Here in the heart of East Africa

3
00:00:09,680 --> 00:00:13,840
The Great Migration
"""
        srt_file = temp_srt_dir / "test.srt"
        srt_file.write_text(srt_content, encoding='utf-8')
        return str(srt_file)

    @pytest.fixture
    def multiline_srt(self, temp_srt_dir):
        srt_content = """1
00:00:00,000 --> 00:00:05,000
This is line one
and line two
"""
        srt_file = temp_srt_dir / "multiline.srt"
        srt_file.write_text(srt_content, encoding='utf-8')
        return str(srt_file)

    @pytest.fixture
    def html_srt(self, temp_srt_dir):
        srt_content = """1
00:00:00,000 --> 00:00:05,000
<i>Italic text</i> and <b>bold text</b>
"""
        srt_file = temp_srt_dir / "html.srt"
        srt_file.write_text(srt_content, encoding='utf-8')
        return str(srt_file)

    def test_parse_valid_srt(self, valid_srt):
        segments = parse_srt_file(valid_srt)

        assert len(segments) == 3

        assert segments[0]['id'] == 0
        assert segments[0]['start'] == 0.0
        assert segments[0]['end'] == 4.68
        assert "Tanzania" in segments[0]['text']

        assert segments[1]['id'] == 1
        assert segments[1]['start'] == 4.68
        assert segments[1]['end'] == 9.68

        assert segments[2]['id'] == 2
        assert segments[2]['start'] == 9.68
        assert segments[2]['end'] == 13.84

    def test_parse_multiline_srt(self, multiline_srt):
        segments = parse_srt_file(multiline_srt)

        assert len(segments) == 1
        # Multi-line text should be joined with space
        assert "line one and line two" in segments[0]['text']

    def test_parse_html_srt(self, html_srt):
        segments = parse_srt_file(html_srt)

        assert len(segments) == 1
        # HTML tags should be stripped
        assert "<i>" not in segments[0]['text']
        assert "<b>" not in segments[0]['text']
        assert "Italic text and bold text" == segments[0]['text']

    def test_parse_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            parse_srt_file("nonexistent.srt")

    def test_parse_empty_srt(self, temp_srt_dir):
        empty_file = temp_srt_dir / "empty.srt"
        empty_file.write_text("", encoding='utf-8')

        with pytest.raises(ValueError, match="No valid subtitles found"):
            parse_srt_file(str(empty_file))

    def test_parse_malformed_srt(self, temp_srt_dir):
        malformed_content = """1
This is not a valid timestamp
Some text
"""
        malformed_file = temp_srt_dir / "malformed.srt"
        malformed_file.write_text(malformed_content, encoding='utf-8')

        with pytest.raises(ValueError, match="No valid subtitles found"):
            parse_srt_file(str(malformed_file))


class TestValidation:
    def test_validate_valid_segments(self):
        segments = [
            {'id': 0, 'start': 0.0, 'end': 5.0, 'text': 'First'},
            {'id': 1, 'start': 5.0, 'end': 10.0, 'text': 'Second'},
        ]

        result = validate_srt_segments(segments)

        assert result['valid'] is True
        assert len(result['errors']) == 0

    def test_validate_overlapping_segments(self):
        segments = [
            {'id': 0, 'start': 0.0, 'end': 5.0, 'text': 'First'},
            {'id': 1, 'start': 4.5, 'end': 10.0, 'text': 'Second'},  # Overlaps
        ]

        result = validate_srt_segments(segments)

        assert len(result['warnings']) > 0
        assert any('Overlaps' in w for w in result['warnings'])

    def test_validate_negative_timestamp(self):
        segments = [
            {'id': 0, 'start': -1.0, 'end': 5.0, 'text': 'First'},
        ]

        result = validate_srt_segments(segments)

        assert result['valid'] is False
        assert len(result['errors']) > 0
        assert any('Negative timestamp' in e for e in result['errors'])

    def test_validate_start_after_end(self):
        segments = [
            {'id': 0, 'start': 5.0, 'end': 2.0, 'text': 'First'},
        ]

        result = validate_srt_segments(segments)

        assert result['valid'] is False
        assert len(result['errors']) > 0

    def test_validate_empty_text(self):
        segments = [
            {'id': 0, 'start': 0.0, 'end': 5.0, 'text': '   '},
        ]

        result = validate_srt_segments(segments)

        assert len(result['warnings']) > 0
        assert any('Empty text' in w for w in result['warnings'])


class TestSrtSaving:
    @pytest.fixture
    def temp_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_save_segments_as_srt(self, temp_output_dir):
        segments = [
            {'id': 0, 'start': 0.0, 'end': 4.68, 'text': 'First subtitle'},
            {'id': 1, 'start': 4.68, 'end': 9.68, 'text': 'Second subtitle'},
        ]

        output_path = temp_output_dir / "output.srt"
        save_segments_as_srt(segments, str(output_path))

        assert output_path.exists()

        # Read back and verify
        parsed = parse_srt_file(str(output_path))

        assert len(parsed) == 2
        assert parsed[0]['start'] == 0.0
        assert parsed[0]['end'] == 4.68
        assert parsed[0]['text'] == 'First subtitle'

    def test_save_and_reload_round_trip(self, temp_output_dir):
        original_segments = [
            {'id': 0, 'start': 0.0, 'end': 5.0, 'text': 'Test subtitle one'},
            {'id': 1, 'start': 5.0, 'end': 10.5, 'text': 'Test subtitle two'},
            {'id': 2, 'start': 10.5, 'end': 15.25, 'text': 'Test subtitle three'},
        ]

        output_path = temp_output_dir / "roundtrip.srt"
        save_segments_as_srt(original_segments, str(output_path))

        # Read back
        loaded_segments = parse_srt_file(str(output_path))

        assert len(loaded_segments) == len(original_segments)
        for orig, loaded in zip(original_segments, loaded_segments):
            assert abs(orig['start'] - loaded['start']) < 0.001
            assert abs(orig['end'] - loaded['end']) < 0.001
            assert orig['text'] == loaded['text']


class TestRealWorldSrt:
    def test_parse_tanzania_srt_if_exists(self):
        # Test with actual Tanzania.srt if it exists
        srt_path = Path("data/input/Tanzania-caption.srt")
        if not srt_path.exists():
            pytest.skip("Tanzania.srt not found")

        segments = parse_srt_file(str(srt_path))

        assert len(segments) > 0
        assert all('start' in seg for seg in segments)
        assert all('end' in seg for seg in segments)
        assert all('text' in seg for seg in segments)

        # Validate timing
        result = validate_srt_segments(segments)
        assert result['valid'] is True
