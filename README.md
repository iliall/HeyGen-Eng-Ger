# Video Translation: English to German

## Overview
This project translates videos from English to German while preserving the speaker's voice identity, tone, and quality. The pipeline supports both audio transcription and SRT subtitle input, with optional voice cloning for perfect voice preservation.

## Quick Start

### Prerequisites
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install system dependencies (macOS)
brew install ffmpeg rubberband

# Set up environment variables
cp .env.example .env
# Add your ELEVENLABS_API_KEY to .env
```

### Basic Usage

#### 1. **Simple Translation** (Default Voice)
```bash
python -m src.main input.mp4 -o output_de.mp4
```

#### 2. **Translation with Voice Cloning** (Recommended)
```bash
python -m src.main input.mp4 -o output_de.mp4 --clone-voice
```

#### 3. **Use Existing SRT Subtitles**
```bash
python -m src.main input.mp4 --srt-input subtitles.srt -o output_de.mp4 --clone-voice
```

#### 4. **Generate Translated Subtitles**
```bash
python -m src.main input.mp4 -o output_de.mp4 --save-srt --clone-voice
```

#### 5. **Advanced: Forced Alignment (Experimental)**
```bash
python -m src.main input.mp4 --srt-input subtitles.srt -o output_de.mp4 --clone-voice --word-level-timing
```

## Complete Command Reference

```bash
python -m src.main INPUT_VIDEO [OPTIONS]

Required Arguments:
  INPUT_VIDEO                 Path to input video file

Options:
  -o, --output PATH           Output video path
  --source-lang TEXT          Source language code (default: en)
  --target-lang TEXT          Target language code (default: de)

Voice Options:
  --voice-id TEXT             ElevenLabs voice ID (default: 21m00Tcm4TlvDq8ikWAM)
                              Ignored if --clone-voice is used
  --clone-voice               Clone voice from original video (recommended)
  --voice-name TEXT           Name for cloned voice (optional)

Voice Quality Settings:
  --stability FLOAT           Voice stability 0-1 (default: 0.5, lower=more emotional)
  --similarity-boost FLOAT    Similarity to cloned voice 0-1 (default: 0.8)
  --style FLOAT               Style exaggeration 0-1 (default: 0.4)

Processing Options:
  --whisper-model TEXT        Whisper model size: tiny/base/small/medium/large (default: base)
  --translation-service TEXT  Translation service: google/deepl (default: google)
  --temp-dir PATH             Temporary files directory (default: data/temp)
  --keep-temp                 Keep temporary files after processing
  --save-transcription        Save transcription to JSON file

SRT Subtitle Options:
  --srt-input PATH            Use SRT file instead of audio transcription
  --save-srt                  Save translated subtitles as SRT file
  --word-level-timing         Use forced alignment for word-level timing (experimental)
```

## Output Files

The tool generates the following files:

```
data/output/
├── video_de.mp4              # Translated video (main output)
├── video_de.srt              # Translated subtitles (if --save-srt)
└── video_transcription.json  # Original transcription (if --save-transcription)

data/temp/                    # Temporary files (if --keep-temp)
├── video_audio.wav           # Extracted audio
├── segments/                 # Individual audio segments
├── voice_samples/            # Voice cloning samples
└── video_de_audio.wav        # Merged translated audio
```

## Testing

### Run Test Suite
```bash
pytest tests/ -v
```

### Test with Sample Video
```bash
# Test with included Tanzania video
python -m src.main data/input/Tanzania.mp4 \
  --srt-input data/input/Tanzania-caption.srt \
  -o data/output/test_output.mp4 \
  --clone-voice
```

## Limitations

### Current Known Limitations:

1. **Background Audio Not Preserved**
   - All original audio is replaced with synthesized voice
   - Background music, ambient sounds, and sound effects are lost
   - Output contains only the translated voice with no atmosphere

2. **Single Speaker Only**
   - System assumes one speaker throughout the video
   - Multiple speakers will be cloned as a single voice
   - No speaker diarization (identifying who is speaking when)

3. **No Visual Lip-Sync**
   - Only audio timing is adjusted (time-stretching)
   - Mouth movements still show original language
   - Noticeable in close-up shots of speakers

4. **Voice Cloning Quality Depends on Source**
   - Requires clean voice samples (3 longest segments used)
   - Background noise in source affects clone quality
   - Very short videos (< 30s) may not have enough voice data
   - **Workaround**: Use `--stability` and `--similarity-boost` to fine-tune

5. **Time-Stretching Limitations**
   - Extreme speed changes (>50%) can sound unnatural
   - German often has different sentence lengths than English
   - Segment merging helps but doesn't eliminate all issues
   - **Best Practice**: Use SRT input for better timing control

6. **Processing Time & Cost**
   - ElevenLabs API costs per character synthesized
   - Longer videos require more API calls and processing time
   - Voice cloning adds ~30s setup time per video
   - **Estimate**: ~2-5 minutes per minute of video (depending on model size)

7. **Whisper Transcription Accuracy**
   - Fast speech, accents, or technical terms may be misheard
   - Background noise reduces transcription quality
   - **Workaround**: Use `--srt-input` with human-corrected subtitles for best results
   - Larger models (`--whisper-model large`) are more accurate but slower

## API Keys Required

Add to your `.env` file:
```
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
```

Optionally for DeepL:
```
DEEPL_API_KEY=your_deepl_api_key_here
```
