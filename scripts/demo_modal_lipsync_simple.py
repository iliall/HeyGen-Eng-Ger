"""
Simple demo to test Modal lipsync with Tanzania video.

Usage:
    modal run scripts/demo_modal_lipsync_simple.py
"""

import modal
from pathlib import Path
import sys

# Add modal scripts directory to path
modal_dir = Path(__file__).parent / "modal"
sys.path.insert(0, str(modal_dir))

from lipsync_modal import app, process_lipsync_video


@app.local_entrypoint()
def main():
    """Run lipsync demo on Modal GPU."""

    print("Modal Lipsync Demo - Tanzania Video")
    print("=" * 50)

    # File paths
    video_file = Path("data/input/Tanzania.mp4")
    audio_file = Path("data/temp/Tanzania_de_audio.wav")
    output_file = Path("data/output/Tanzania_modal_lipsync.mp4")

    # Verify inputs exist
    if not video_file.exists():
        print(f"[ERROR] Video not found: {video_file}")
        return

    if not audio_file.exists():
        print(f"[ERROR] Audio not found: {audio_file}")
        print(f"\nRun translation first to generate German audio:")
        print(f"  python -m src.main data/input/Tanzania.mp4 \\")
        print(f"    --srt-input data/input/Tanzania-caption.srt \\")
        print(f"    --clone-voice --keep-temp")
        return

    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Read input files
    print(f"Reading video: {video_file}")
    video_data = video_file.read_bytes()
    video_mb = len(video_data) / 1024 / 1024

    print(f"Reading audio: {audio_file}")
    audio_data = audio_file.read_bytes()
    audio_mb = len(audio_data) / 1024 / 1024

    print()
    print(f"Video size: {video_mb:.1f} MB")
    print(f"Audio size: {audio_mb:.1f} MB")
    print()

    # Process on Modal
    print("Uploading to Modal and starting GPU processing...")
    print("This may take a few minutes...")
    print()

    result = process_lipsync_video.remote(
        input_video_data=video_data,
        input_audio_data=audio_data,
        quality="fast"
    )

    print()
    print("Result:")
    print(f"  Status: {result.get('status')}")

    if result.get('status') == 'success':
        # Save output
        output_data = result.get('output_data')
        if output_data:
            print(f"Saving output to: {output_file}")
            output_file.write_bytes(output_data)

            final_mb = len(output_data) / 1024 / 1024
            print(f"[OK] Success! Output video saved ({final_mb:.1f} MB)")
            print(f"  Device used: {result.get('device')}")
            print()
            print(f"View the result:")
            print(f"  open {output_file}")
        else:
            print("[ERROR] No output data received")
    else:
        print(f"[ERROR] Failed: {result.get('error', result.get('reason', 'unknown'))}")


if __name__ == "__main__":
    main()
