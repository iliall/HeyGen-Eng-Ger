"""
Complete Modal-based lip-sync processing solution.
Handles model setup, downloading, and video processing in a single script.
"""

import modal
import os
from pathlib import Path

# Define the Modal image with all dependencies
image = modal.Image.debian_slim().apt_install(
    "git",
    "ffmpeg",
    "libgl1-mesa-glx",
    "libglib2.0-0",
    "libsm6",
    "libxext6",
    "libxrender-dev"
).pip_install([
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "torchaudio>=2.0.0",
    "opencv-python>=4.8.0",
    "librosa>=0.10.0",
    "batch-face>=1.5.0",
    "numpy>=1.26.0",
    "scipy>=1.11.4",
    "tqdm",
    "requests",
    "ffmpeg-python",
    "click",
    "pyyaml",
    "python-dotenv",
    "openai-whisper",
    "deep-translator",
    "elevenlabs",
    "gdown",
]).run_commands(
    "git clone https://github.com/Rudrabha/Wav2Lip.git /Wav2Lip"
)

# Create Modal volume for models
models_volume = modal.Volume.from_name("lipsync-models", create_if_missing=True)

# Create the Modal app
app = modal.App("lipsync-modal", image=image)


@app.function(
    gpu="T4",
    timeout=1800,
    volumes={"/models": models_volume},
)
def setup_models():
    """Download and setup lip-sync models on Modal."""
    import gdown
    import torch
    from pathlib import Path

    print("Setting up lip-sync models on Modal...")

    models_dir = Path("/models/lipsync")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Model URLs
    models = {
        "wav2lip.pth": "https://drive.google.com/uc?id=15RsBIQ1D3l2hkQ4h9I3LrqPYSKvxqckm",
        "s3fd.pth": "https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/s3fd.pth"
    }

    downloaded_models = {}

    for model_name, url in models.items():
        model_path = models_dir / model_name
        if model_path.exists():
            size_mb = model_path.stat().st_size / 1024 / 1024
            print(f"[OK] {model_name} already exists ({size_mb:.1f}MB)")
            downloaded_models[model_name] = str(model_path)
        else:
            print(f"Downloading {model_name}...")
            try:
                gdown.download(url, str(model_path), quiet=False)
                if model_path.exists():
                    size_mb = model_path.stat().st_size / 1024 / 1024
                    print(f"[OK] {model_name} downloaded successfully ({size_mb:.1f}MB)")
                    downloaded_models[model_name] = str(model_path)
                else:
                    print(f"[ERROR] Failed to download {model_name}")
            except Exception as e:
                print(f"[ERROR] Error downloading {model_name}: {e}")

    # Test model loading
    if "wav2lip.pth" in downloaded_models:
        try:
            checkpoint = torch.load(downloaded_models["wav2lip.pth"], map_location='cpu', weights_only=False)
            print(f"[OK] Wav2Lip model loads successfully (type: {type(checkpoint)})")
        except Exception as e:
            print(f"[ERROR] Wav2Lip model loading failed: {e}")

    return {
        "models_dir": str(models_dir),
        "downloaded_models": downloaded_models,
        "gpu_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


@app.function(
    gpu="T4",
    timeout=3600,  # 1 hour
    volumes={"/models": models_volume},
    secrets=[modal.Secret.from_name("elevenlabs")],
)
def process_lipsync_video(
    input_video_data: bytes,
    input_audio_data: bytes,
    quality: str = "balanced",
    face_detector: str = "auto"
):
    """Process lip-sync on Modal GPU."""
    import sys
    import torch
    import os
    from pathlib import Path
    import tempfile

    print("Modal Lip-Sync Processing")
    print("=" * 40)

    # Setup paths
    sys.path.insert(0, "/Wav2Lip")
    models_dir = Path("/models/lipsync")

    # Check models
    wav2lip_path = models_dir / "wav2lip.pth"
    s3fd_path = models_dir / "s3fd.pth"

    if not (wav2lip_path.exists() and s3fd_path.exists()):
        print("[ERROR] Models not available. Please run setup first.")
        return {"status": "failed", "reason": "models_missing"}

    print(f"GPU available: {'Yes' if torch.cuda.is_available() else 'No'}")
    print(f"Video data: {len(input_video_data) / 1024 / 1024:.1f} MB")
    print(f"Audio data: {len(input_audio_data) / 1024 / 1024:.1f} MB")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Write input files
            input_video_path = tmp / "input.mp4"
            input_audio_path = tmp / "audio.wav"
            output_video_path = tmp / "output.mp4"

            input_video_path.write_bytes(input_video_data)
            input_audio_path.write_bytes(input_audio_data)

            print(f"Temp input video: {input_video_path}")
            print(f"Temp input audio: {input_audio_path}")

            # Import local lipsync module
            sys.path.insert(0, "/root")
            from src.lipsync.synthesizer import apply_lipsync

            print("\nStarting lip-sync processing...")

            # Process lip-sync
            result_path = apply_lipsync(
                video_path=str(input_video_path),
                audio_path=str(input_audio_path),
                output_path=str(output_video_path),
                quality=quality
            )

            if os.path.exists(result_path):
                # Read result
                output_data = Path(result_path).read_bytes()
                size_mb = len(output_data) / 1024 / 1024

                print(f"\n[OK] Lip-sync processing completed successfully")
                print(f"Output video: {result_path}")
                print(f"File size: {size_mb:.1f}MB")

                return {
                    "status": "success",
                    "output_data": output_data,
                    "file_size_mb": size_mb,
                    "device": "cuda" if torch.cuda.is_available() else "cpu"
                }
            else:
                return {"status": "failed", "reason": "output_not_created"}

    except Exception as e:
        import traceback
        print(f"[ERROR] Lip-sync processing failed: {e}")
        print(traceback.format_exc())
        return {"status": "failed", "error": str(e)}


@app.function(
    gpu="T4",
    timeout=1800,
    volumes={"/models": models_volume},
)
def test_setup():
    """Test Modal lip-sync setup."""
    import sys
    import torch
    from pathlib import Path

    print("Testing Modal Lip-Sync Setup")
    print("=" * 35)

    sys.path.insert(0, "/Wav2Lip")
    models_dir = Path("/models/lipsync")

    results = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    if torch.cuda.is_available():
        results["gpu_name"] = torch.cuda.get_device_name(0)
        results["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"

    # Check models
    wav2lip_path = models_dir / "wav2lip.pth"
    s3fd_path = models_dir / "s3fd.pth"

    results["wav2lip_available"] = wav2lip_path.exists()
    results["s3fd_available"] = s3fd_path.exists()

    # Test imports
    try:
        import cv2
        import librosa
        results["dependencies"] = "success"
    except Exception as e:
        results["dependencies"] = f"failed: {e}"

    # Test Wav2Lip import
    try:
        from models.wav2lip import Wav2Lip
        results["wav2lip_import"] = "success"
    except Exception as e:
        results["wav2lip_import"] = f"failed: {e}"

    print("\nTest Results:")
    for key, value in results.items():
        status = "[OK]" if "success" in str(value) or value == True else "[FAIL]"
        print(f"   {status} {key}: {value}")

    return results


# Local entrypoints
@app.local_entrypoint()
def setup():
    """Setup lip-sync models on Modal."""
    print("Setting up lip-sync models on Modal...")
    print("Usage: modal run lipsync_modal.py::setup")


@app.local_entrypoint()
def test():
    """Test Modal lip-sync setup."""
    print("Testing Modal lip-sync setup...")
    print("Usage: modal run lipsync_modal.py::test")


@app.local_entrypoint()
def process(
    input_video: str,
    input_audio: str,
    output_video: str,
    quality: str = "balanced",
    face_detector: str = "auto"
):
    """Process lip-sync on Modal."""
    print("Starting lip-sync processing on Modal...")
    print("Usage: modal run lipsync_modal.py::process --input-video <video> --input-audio <audio> --output-video <output>")


if __name__ == "__main__":
    import sys
    import subprocess

    if len(sys.argv) < 2:
        print("Modal Lip-Sync Processing Tool")
        print("=" * 40)
        print()
        print("Usage:")
        print("  modal run lipsync_modal.py::setup")
        print("  modal run lipsync_modal.py::test")
        print("  modal run lipsync_modal.py::process --input-video video.mp4 --input-audio audio.wav --output-video output.mp4")
        print()
        print("Or use the convenience wrapper:")
        print("  python lipsync_modal.py setup")
        print("  python lipsync_modal.py test")
        print("  python lipsync_modal.py process video.mp4 audio.wav output.mp4 [quality] [face_detector]")
        print()
        print("Examples:")
        print("  modal run lipsync_modal.py::setup")
        print("  modal run lipsync_modal.py::test")
        print("  modal run lipsync_modal.py::process --input-video input.mp4 --input-audio translated.wav --output-video output.mp4 --quality balanced --face-detector auto")
        sys.exit(0)

    # Convenience wrapper - run commands via modal
    command = sys.argv[1]

    if command == "setup":
        print("Running setup on Modal...")
        subprocess.run(["modal", "run", "lipsync_modal.py::setup"])
    elif command == "test":
        print("Running test on Modal...")
        subprocess.run(["modal", "run", "lipsync_modal.py::test"])
    elif command == "process":
        if len(sys.argv) < 5:
            print("Usage: python lipsync_modal.py process <video> <audio> <output> [quality] [face_detector]")
            sys.exit(1)

        input_video = sys.argv[2]
        input_audio = sys.argv[3]
        output_video = sys.argv[4]
        quality = sys.argv[5] if len(sys.argv) > 5 else "balanced"
        face_detector = sys.argv[6] if len(sys.argv) > 6 else "auto"

        print("Starting lip-sync processing on Modal...")
        cmd = [
            "modal", "run", "lipsync_modal.py::process",
            "--input-video", input_video,
            "--input-audio", input_audio,
            "--output-video", output_video,
            "--quality", quality,
            "--face-detector", face_detector
        ]
        subprocess.run(cmd)
    else:
        print(f"Error: Unknown command '{command}'")
        print("Available commands: setup, test, process")
        sys.exit(1)