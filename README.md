# Video Translation: English to German

## Overview
This project translates videos from English to German while preserving the speaker's voice identity, tone, and quality. The pipeline supports both audio transcription and SRT subtitle input, with optional voice cloning for perfect voice preservation.

## Pipeline Overview

The translation pipeline consists of the following main steps:

### Step 1: Audio Extraction
- Extracts audio from the input video using FFmpeg
- Saves as high-quality WAV file (44.1kHz) for processing
- **Implementation**: `src/video/extractor.py`

### Step 1.5: Voice Separation (Optional)
- **When enabled**: Separates audio into vocals and background using Demucs AI model
  - **Vocals**: Isolated speech/voice (used for transcription and voice cloning)
  - **Background**: Music, ambient sounds, and noise (currently not preserved in output)
- **Benefits**:
  - Cleaner voice samples for more accurate transcription
  - Better voice cloning quality (no background interference)
  - Reduces background noise in transcription
- **Skip with**: `--no-background` flag (48% faster, recommended if video has no background audio)
- **Requires**: `demucs` package (installed via requirements.txt)
- **Implementation**: `src/audio/separation.py`

### Step 2: Transcription
- **Option A**: Transcribe audio using OpenAI Whisper (local)
  - Uses separated vocals (if available) for cleaner transcription
  - Supports multiple model sizes (tiny, base, small, medium, large)
  - Automatic segment merging: segments with ≤5 words are merged with the next segment
  - Provides word-level timestamps for each segment
- **Option B**: Load existing SRT subtitle file
  - Parses SRT format with timestamp validation
  - Strips HTML tags and formatting
- **Implementation**: `src/audio/transcription.py`, `src/audio/srt_parser.py`

### Step 3: Translation
- Translates English text to German
- Supports Google Translate (default) and DeepL
- Preserves segment structure and timing information
- **Implementation**: `src/audio/translation.py`

### Step 4: Voice Synthesis
- **Option A**: Use pre-configured ElevenLabs voice
- **Option B**: Clone voice from original speaker (recommended)
  - Uses separated vocals (if available) for cleaner voice samples
  - Extracts 3 longest audio segments as voice samples
  - Creates instant voice clone via ElevenLabs API
- Generates German audio segment-by-segment
- Voice quality settings: stability=0.5, similarity_boost=0.8, style=0.4
- **Implementation**: `src/audio/synthesis.py`

### Step 5: Audio Time-Stretching
- **Default Method**: Segment-level time-stretching
  - Each translated segment is time-stretched to match original duration
  - Uses rubberband for pitch-preserving speed adjustment
  - Achieves <1% duration mismatch for proper lip-sync
- **Optional Method**: Word-level forced alignment (experimental)
  - Extracts word-level timestamps from original audio via ElevenLabs API
  - Aligns translated German words to original English timing
  - Provides frame-accurate synchronization
  - Enable with `--word-level-timing` flag
- **Implementation**: `src/audio/utils.py`, `src/audio/synthesis.py` (forced alignment)

### Step 6: Audio Merging
- Combines all time-stretched segments into single audio file
- Maintains original timing and silence gaps
- **Implementation**: `src/audio/utils.py`

### Step 7: Video Merging
- Replaces original audio track with translated audio
- Preserves original video stream without re-encoding
- Uses FFmpeg for efficient stream replacement
- **Implementation**: `src/video/merger.py`

## Setup

### Requirements

- **Python**: 3.8 or higher
- **Operating System**: macOS, Linux, or Windows
- **API Keys**: ElevenLabs API key (required for voice synthesis)

### Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/iliall/HeyGen-Eng-Ger.git
cd HeyGen-Eng-Ger
```

#### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

#### 3. Install System Dependencies

**macOS:**
```bash
brew install ffmpeg rubberband
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg rubberband-cli
```

**Windows:**
- Download and install [FFmpeg](https://ffmpeg.org/download.html)
- Download and install [Rubberband](https://breakfastquay.com/rubberband/)
- Add both to your system PATH

#### 4. Install Python Dependencies
```bash
pip install -r requirements.txt
```

**Note:** This will install ~2GB of dependencies including:
- `demucs` (voice separation model)
- `openai-whisper` (speech recognition)
- `torch` (deep learning framework)
- `elevenlabs` (voice synthesis API)

#### 5. Configure Environment Variables

Create a `.env` file in the project root:
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```bash
# Required
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here

# Optional (for DeepL translation)
DEEPL_API_KEY=your_deepl_api_key_here
```

**Getting an ElevenLabs API Key:**
1. Sign up at [elevenlabs.io](https://elevenlabs.io)
2. Go to your [Profile Settings](https://elevenlabs.io/app/settings)
3. Copy your API key from the "API Key" section

#### 6. Verify Installation

Test the setup with a sample video:
```bash
python -m src.main data/input/Tanzania.mp4 -o data/output/test.mp4 --clone-voice
```

If successful, you should see output in `data/output/test.mp4`.

---

## Testing with Sample Videos

The repository includes sample videos in `data/input/` for testing. Here are recommended test commands:

### Quick Test (Fastest)
Basic translation with pre-made voice, no background separation:
```bash
python -m src.main data/input/Tanzania.mp4 -o data/output/tanzania_quick.mp4 --no-background
```

### Standard Test (Recommended for Tanzania)
Voice cloning with SRT subtitles, no background separation:
```bash
python -m src.main data/input/Tanzania.mp4 \
  --srt-input data/input/Tanzania-caption.srt \
  -o data/output/tanzania_standard.mp4 \
  --clone-voice \
  --no-background
```

### Full Features Test
Force alignment for frame-accurate timing:
```bash
python -m src.main data/input/Tanzania.mp4 \
  --srt-input data/input/Tanzania-caption.srt \
  -o data/output/tanzania_word_level.mp4 \
  --clone-voice \
  --no-background \
  --word-level-timing
```

### Background Separation Test (Tanzania_2)
Test voice separation with background audio (slower, for videos with music/ambient sounds):
```bash
python -m src.main data/input/Tanzania_2.mp4 \
  -o data/output/tanzania2_with_separation.mp4 \
  --clone-voice --word-level-timing
```
**Note:** Tanzania_2 has background audio, so voice separation runs automatically to extract clean vocals for better transcription and cloning quality. Takes ~3-5 minutes (slower than `--no-background`).

**Notes:**
- Tanzania video has **no background audio**, so `--no-background` is recommended (48% faster)
- Using `--srt-input` with existing subtitles gives better timing than Whisper auto-transcription
- `--clone-voice` significantly improves quality by matching the original speaker's voice

---

## Usage

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

#### 5. **Fast Translation** (No Background Audio)
```bash
python -m src.main input.mp4 -o output_de.mp4 --clone-voice --no-background
# 48% faster - for videos without background music or ambient sounds
```

#### 6. **Advanced: Forced Alignment (Experimental)**
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

Audio Separation Options:
  --no-background             Skip voice separation (48% faster, assumes no background audio)
  --background-enhancement/--no-background-enhancement  Apply noise reduction to separated background audio (default: True)

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
├── separation/               # Voice separation output (if available)
│   ├── video_audio_vocals.wav     # Isolated vocals (clean)
│   └── video_audio_background.wav  # Background audio (music, ambient sounds)
└── video_de_audio.wav        # Merged translated audio
```

## Example Files

This repository includes example input and output files for demonstration:

### Input Files
- **Video**: `data/input/Tanzania.mp4` - Sample video with English narration about Tanzania
- **Subtitles**: `data/input/Tanzania-caption.srt` - Original English subtitles

### Output Files
- **Translated Video**: `data/output/test_translation.mp4` - German translation with cloned voice
- Additional output files are generated when running the pipeline with appropriate flags

## Testing

### Run Test Suite
```bash
pytest tests/ -v
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
   - **Note**: Check out the `lipsync` branch for a partial Wav2Lip implementation

4. **Voice Cloning Quality Depends on Source**
   - Requires clean voice samples (3 longest segments used)
   - Background noise in source affects clone quality
   - Very short videos (< 30s) may not have enough voice data
   - **Workaround**: Voice separation (enabled by default) automatically removes background noise for better cloning quality. Also adjust `--stability` and `--similarity-boost` to fine-tune

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
   - **Workaround**: Voice separation (enabled by default) removes background noise for better transcription. For best results, use `--srt-input` with human-corrected subtitles or larger models (`--whisper-model large`)
