import re
from typing import List, Dict, Optional
from pathlib import Path


def srt_timestamp_to_seconds(timestamp: str) -> float:
    """
    Convert .srt timestamp to seconds.

    Args:
        timestamp: Timestamp in format "HH:MM:SS,mmm" (e.g., "00:00:04,680")

    Returns:
        Time in seconds as float (e.g., 4.68)
    """
    # Handle both comma and period as decimal separator
    timestamp = timestamp.replace(',', '.')

    # Parse HH:MM:SS.mmm
    parts = timestamp.split(':')
    if len(parts) != 3:
        raise ValueError(f"Invalid timestamp format: {timestamp}")

    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])

    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds


def seconds_to_srt_timestamp(seconds: float) -> str:
    """
    Convert seconds to .srt timestamp format.

    Args:
        seconds: Time in seconds (e.g., 4.68)

    Returns:
        Timestamp in format "HH:MM:SS,mmm" (e.g., "00:00:04,680")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')


def strip_html_tags(text: str) -> str:
    """
    Remove HTML tags from subtitle text.

    Args:
        text: Text potentially containing HTML tags (e.g., "<i>Hello</i>")

    Returns:
        Clean text without HTML tags (e.g., "Hello")
    """
    # Remove HTML tags
    clean_text = re.sub(r'<[^>]+>', '', text)
    return clean_text


def parse_srt_file(srt_path: str) -> List[Dict]:
    """
    Parse .srt file and convert to internal segment format.

    Args:
        srt_path: Path to .srt subtitle file

    Returns:
        List of segments in internal format:
        [
            {'id': 0, 'start': 0.0, 'end': 4.68, 'text': '...'},
            {'id': 1, 'start': 4.68, 'end': 9.68, 'text': '...'},
            ...
        ]

    Raises:
        FileNotFoundError: If .srt file doesn't exist
        ValueError: If .srt format is invalid
    """
    srt_file = Path(srt_path)
    if not srt_file.exists():
        raise FileNotFoundError(f"Subtitle file not found: {srt_path}")

    with open(srt_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by double newline (subtitle entries are separated by blank lines)
    entries = re.split(r'\n\s*\n', content.strip())

    segments = []

    for entry in entries:
        if not entry.strip():
            continue

        lines = entry.strip().split('\n')

        if len(lines) < 3:
            # Skip malformed entries
            continue

        # Line 0: Index (e.g., "1")
        try:
            index = int(lines[0].strip())
        except ValueError:
            # Skip if index is not a number
            continue

        # Line 1: Timestamp (e.g., "00:00:00,000 --> 00:00:04,680")
        timestamp_line = lines[1].strip()
        timestamp_match = re.match(r'(\S+)\s+-->\s+(\S+)', timestamp_line)

        if not timestamp_match:
            # Skip if timestamp format is invalid
            continue

        start_time_str = timestamp_match.group(1)
        end_time_str = timestamp_match.group(2)

        try:
            start_time = srt_timestamp_to_seconds(start_time_str)
            end_time = srt_timestamp_to_seconds(end_time_str)
        except ValueError:
            # Skip if timestamp conversion fails
            continue

        # Lines 2+: Subtitle text (can be multi-line)
        text_lines = lines[2:]
        text = ' '.join(line.strip() for line in text_lines if line.strip())

        # Clean HTML tags
        text = strip_html_tags(text)

        if not text:
            # Skip empty text
            continue

        # Create segment in internal format (0-based indexing)
        segment = {
            'id': len(segments),  # Use 0-based index
            'start': start_time,
            'end': end_time,
            'text': text
        }

        segments.append(segment)

    if not segments:
        raise ValueError(f"No valid subtitles found in {srt_path}")

    return segments


def validate_srt_segments(segments: List[Dict]) -> Dict[str, List[str]]:
    """
    Validate that segments are properly formatted.

    Checks:
    - No overlapping timestamps
    - Chronological order
    - No negative timestamps
    - Text is not empty

    Args:
        segments: List of segments to validate

    Returns:
        Dictionary with validation results:
        {
            'valid': True/False,
            'warnings': ['warning 1', 'warning 2', ...],
            'errors': ['error 1', 'error 2', ...]
        }
    """
    warnings = []
    errors = []

    for i, segment in enumerate(segments):
        # Check required fields
        if 'start' not in segment or 'end' not in segment or 'text' not in segment:
            errors.append(f"Segment {i}: Missing required fields")
            continue

        # Check negative timestamps
        if segment['start'] < 0 or segment['end'] < 0:
            errors.append(f"Segment {i}: Negative timestamp (start={segment['start']}, end={segment['end']})")

        # Check start < end
        if segment['start'] >= segment['end']:
            errors.append(f"Segment {i}: Start time >= end time ({segment['start']} >= {segment['end']})")

        # Check empty text
        if not segment['text'].strip():
            warnings.append(f"Segment {i}: Empty text")

        # Check chronological order with previous segment
        if i > 0:
            prev_segment = segments[i - 1]
            if segment['start'] < prev_segment['end']:
                warnings.append(
                    f"Segment {i}: Overlaps with previous segment "
                    f"({segment['start']} < {prev_segment['end']})"
                )
            elif segment['start'] < prev_segment['start']:
                errors.append(
                    f"Segment {i}: Not in chronological order "
                    f"({segment['start']} < {prev_segment['start']})"
                )

    return {
        'valid': len(errors) == 0,
        'warnings': warnings,
        'errors': errors
    }


def save_segments_as_srt(segments: List[Dict], output_path: str) -> None:
    """
    Save internal segments to .srt format.

    Args:
        segments: Internal segment format
        output_path: Path to save .srt file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments):
            # Index (1-based)
            f.write(f"{i + 1}\n")

            # Timestamp
            start_str = seconds_to_srt_timestamp(segment['start'])
            end_str = seconds_to_srt_timestamp(segment['end'])
            f.write(f"{start_str} --> {end_str}\n")

            # Text
            f.write(f"{segment['text']}\n")

            # Blank line between entries
            f.write("\n")
