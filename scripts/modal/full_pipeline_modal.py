"""
Run complete video translation pipeline on Modal GPU.
"""

import modal

# Define Modal image with all dependencies
image = modal.Image.debian_slim().pip_install([
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "opencv-python>=4.8.0",
    "librosa>=0.10.0",
    "batch-face>=1.5.0",
    "numpy>=1.26.0",
    "scipy>=1.11.4",
    "ffmpeg-python",
    "click",
    "pyyaml",
    "python-dotenv",
    "openai-whisper",
    "deep-translator",
    "elevenlabs",
    "pydub",
    "requests",
    "tqdm",
    "typing-extensions",
]).apt_install("ffmpeg", "rubberband-cli").run_commands(
    "git clone https://github.com/Rudrabha/Wav2Lip.git /Wav2Lip"
)

# Create Modal volume for models
models_volume = modal.Volume.from_name("lipsync-models", create_if_missing=True)

# Create Modal app
app = modal.App("heygen-translation", image=image)


@app.function(
    gpu="T4",
    timeout=3600,  # 1 hour
    volumes={"/models": models_volume},
    secrets=[modal.Secret.from_name("elevenlabs")],
)
def translate_video(
    input_video_url: str,
    srt_input_url: str,
    target_lang: str = "de",
    enable_lipsync: bool = True,
):
    """
    Translate video with optional lipsync on Modal GPU.

    Args:
        input_video_url: URL or path to input video
        srt_input_url: URL or path to SRT file
        target_lang: Target language code
        enable_lipsync: Enable visual lip-sync

    Returns:
        Dict with output file paths
    """
    import sys
    import subprocess
    from pathlib import Path
    import tempfile

    print("HeyGen Video Translation on Modal GPU")
    print("=" * 50)

    # Setup paths
    sys.path.insert(0, "/Wav2Lip")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Download input files
        input_video = tmp / "input.mp4"
        srt_file = tmp / "input.srt"
        output_video = tmp / "output.mp4"

        # Use wget or curl to download files
        subprocess.run(["wget", "-O", str(input_video), input_video_url], check=True)
        subprocess.run(["wget", "-O", str(srt_file), srt_input_url], check=True)

        # Build command
        cmd = [
            "python", "-m", "src.main",
            str(input_video),
            "-o", str(output_video),
            "--srt-input", str(srt_file),
            "--target-lang", target_lang,
            "--clone-voice",
            "--keep-temp",
        ]

        if enable_lipsync:
            cmd.extend([
                "--enable-lipsync",
                "--lipsync-quality", "balanced",
            ])

        print(f"Running pipeline: {' '.join(cmd)}")

        # Run pipeline
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        if result.returncode != 0:
            raise RuntimeError(f"Pipeline failed with code {result.returncode}")

        # Read output file
        if output_video.exists():
            output_data = output_video.read_bytes()
            print(f"[OK] Translation complete! Output size: {len(output_data)/1024/1024:.1f}MB")

            return {
                "status": "success",
                "output_size_mb": len(output_data) / 1024 / 1024,
                "output_data": output_data,
            }
        else:
            raise RuntimeError("Output video not created")


@app.local_entrypoint()
def main(
    input_video: str,
    srt_input: str,
    output: str,
    target_lang: str = "de",
    enable_lipsync: bool = True,
):
    """
    Run translation on Modal and save output locally.

    Usage:
        modal run full_pipeline_modal.py \
            --input-video data/input/Tanzania.mp4 \
            --srt-input data/input/Tanzania-caption.srt \
            --output data/output/Tanzania_modal.mp4
    """

    # For now, files need to be uploaded to a URL
    # Simplified version - just show what would happen
    print(f"Input: {input_video}")
    print(f"SRT: {srt_input}")
    print(f"Output: {output}")
    print(f"Lipsync: {'Enabled' if enable_lipsync else 'Disabled'}")
    print("Alternative: Use the demo script to run locally with Modal GPU for lip-sync only.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Modal Full Pipeline")
        print("Note: Input files must be accessible via URL for Modal processing.")
