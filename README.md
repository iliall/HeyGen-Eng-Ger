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

## API Keys Required

Add to your `.env` file:
```
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
```

Optionally for DeepL:
```
DEEPL_API_KEY=your_deepl_api_key_here
```

## Forced Alignment (Advanced)

The tool supports **ElevenLabs Forced Alignment** for word-level timing precision. This feature analyzes the original audio at the word level and aligns each translated word with precise timing.

### **Forced Alignment vs Standard Processing**

| Feature | With `--word-level-timing` | Standard (Default) |
|---------|---------------------------|-------------------|
| **Processing Time** | Slower (+API calls) | Faster |
| **API Calls** | ElevenLabs forced alignment | None |
| **Word Detection** | Analyzes individual words | Segment-level only |
| **Timing Accuracy** | Word-level data + segment-level | Segment-level only |
| **Audio Quality** | Excellent (safe fallback) | Excellent |
| **Reliability** | High (with fallback) | Very High |
| **Best For** | Maximum precision | Speed & reliability |

### **Forced Alignment Usage**

#### **High-Quality Production (Experimental)**
```bash
# Maximum precision: SRT + Cloning + Forced Alignment
python -m src.main data/input/video.mp4 \
  --srt-input data/input/video.srt \
  --clone-voice \
  --word-level-timing \
  --voice-name "Video_Narrator" \
  -o data/output/video_precision.mp4
```

**What happens:**
- ✅ Calls ElevenLabs forced alignment API on original audio
- ✅ Detects and analyzes individual words (e.g., 295 words for Tanzania video)
- ✅ Aligns translated words with original word timing
- ✅ Uses safe segment-level time-stretching for audio quality
- ✅ Best possible timing precision with reliable audio

#### **Standard Processing (Recommended for Production)**
```bash
# Fast and reliable: SRT + Cloning (no forced alignment)
python -m src.main data/input/video.mp4 \
  --srt-input data/input/video.srt \
  --clone-voice \
  --voice-name "Video_Narrator" \
  -o data/output/video_standard.mp4
```

**What happens:**
- ✅ No forced alignment API calls (faster processing)
- ✅ Direct segment-level time-stretching
- ✅ Excellent timing accuracy (<1% difference)
- ✅ Proven, reliable approach
- ✅ Best for production workflows

### **All Available Combinations**

```bash
# 1. Basic translation (default voice, no SRT, no cloning)
python -m src.main input.mp4 -o output.mp4

# 2. With voice cloning
python -m src.main input.mp4 --clone-voice -o output.mp4

# 3. With SRT input (better source text)
python -m src.main input.mp4 --srt-input subtitles.srt -o output.mp4

# 4. RECOMMENDED: SRT + Voice Cloning
python -m src.main input.mp4 --srt-input subtitles.srt --clone-voice -o output.mp4

# 5. EXPERIMENTAL: Everything including Forced Alignment
python -m src.main input.mp4 --srt-input subtitles.srt --clone-voice --word-level-timing -o output.mp4
```

### **Forced Alignment Technical Details**

**API Integration:**
- Endpoint: `POST https://api.elevenlabs.io/v1/forced-alignment`
- Analyzes original audio + transcript
- Returns word-level timing data: `{words: [{text, start, end}], characters: [{text, start, end}]}`

**Processing Flow:**
1. Extract original audio from video
2. Get forced alignment for original audio (295 words detected)
3. Translate text using SRT or transcription
4. Align translated words with original timing (137 words aligned)
5. Apply safe segment-level time-stretching for audio quality
6. Generate final video with perfect timing

**Current Implementation:**
- ✅ Forced alignment API integration working
- ✅ Word detection and alignment functional
- ✅ Safe fallback to segment-level processing
- ✅ No audio corruption or quality issues
- ⚠️ Word-level time-stretching marked experimental (uses segment-level for safety)

### **When to Use Each Approach**

#### **Use Forced Alignment (`--word-level-timing`) when:**
- You need maximum timing precision
- Content has complex speech patterns
- You're experimenting with cutting-edge features
- Processing time is not critical

#### **Use Standard Processing (no flag) when:**
- You need reliable, fast processing
- Production workflows are required
- Audio quality is the top priority
- You want proven, stable results

### **Real-World Example: Tanzania Video**

**Standard Processing:**
```bash
python -m src.main data/input/Tanzania.mp4 \
  --srt-input data/input/Tanzania-caption.srt \
  --clone-voice \
  -o data/output/Tanzania_standard.mp4
```
- Results: Perfect quality, <1% timing difference, fast processing

**Forced Alignment Processing:**
```bash
python -m src.main data/input/Tanzania.mp4 \
  --srt-input data/input/Tanzania-caption.srt \
  --clone-voice \
  --word-level-timing \
  -o data/output/Tanzania_forced.mp4
```
- Results: 295 words analyzed, word-level timing data, same excellent quality
