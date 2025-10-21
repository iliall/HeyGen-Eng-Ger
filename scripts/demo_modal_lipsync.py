"""
Demo script to test Modal-based lipsync processing.

This script demonstrates running the lipsync pipeline on Modal GPU
using the Tanzania video and its German translated audio.
"""

import modal
from pathlib import Path

# Import the Modal app from lipsync_modal
import sys
sys.path.insert(0, str(Path(__file__).parent / "modal"))
from lipsync_modal import app, process_lipsync_video


@app.local_entrypoint()
def main(
    input_video: str = "data/input/Tanzania.mp4",
    input_audio: str = "data/temp/Tanzania_de_audio.wav",
    output_video: str = "data/output/Tanzania_modal_lipsync.mp4",
):
    """
    Run lipsync processing on Modal GPU.

    Args:
        input_video: Path to input video file
        input_audio: Path to translated audio file
        output_video: Path for output video file
    """
    import tempfile
    import shutil
    from pathlib import Path

    print("Modal Lipsync Demo")
    print("=" * 50)
    print(f"Input video: {input_video}")
    print(f"Translated audio: {input_audio}")
    print(f"Output video: {output_video}")
    print()

    # Verify input files exist
    video_path = Path(input_video)
    audio_path = Path(input_audio)

    # Create output directory
    output_path = Path(output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read input files
    with open(video_path, 'rb') as f:
        video_data = f.read()

    with open(audio_path, 'rb') as f:
        audio_data = f.read()

    print(f"   Video size: {len(video_data) / 1024 / 1024:.1f} MB")
    print(f"   Audio size: {len(audio_data) / 1024 / 1024:.1f} MB")
    print()

    # Create temporary files on Modal's side
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_video = Path(tmpdir) / "input.mp4"
        tmp_audio = Path(tmpdir) / "audio.wav"
        tmp_output = Path(tmpdir) / "output.mp4"

        # Write files
        tmp_video.write_bytes(video_data)
        tmp_audio.write_bytes(audio_data)

        # Process on Modal
        result = process_lipsync_video.remote(
            input_video_path=str(tmp_video),
            input_audio_path=str(tmp_audio),
            output_video_path=str(tmp_output),
            quality="fast",
            face_detector="auto"
        )

        print()
        print("Processing Result:")
        print(f"   Status: {result.get('status', 'unknown')}")

        if result.get('status') == 'success':
            print(f"   Output size: {result.get('file_size_mb', 0):.1f} MB")
            print(f"   Device: {result.get('device', 'unknown')}")
            print()

            # Download result
            if tmp_output.exists():
                print(f"Downloading result to {output_video}...")
                shutil.copy(tmp_output, output_path)

                final_size = output_path.stat().st_size / 1024 / 1024
                print(f"[OK] Success! Output video: {output_video} ({final_size:.1f} MB)")
            else:
                print("[ERROR] Output file not found")
        else:
            print(f"[ERROR] Processing failed: {result.get('error', 'unknown error')}")
            print(f"   Reason: {result.get('reason', 'unknown')}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Parse command line arguments
        input_video = sys.argv[1] if len(sys.argv) > 1 else "data/input/Tanzania.mp4"
        input_audio = sys.argv[2] if len(sys.argv) > 2 else "data/temp/Tanzania_de_audio.wav"
        output_video = sys.argv[3] if len(sys.argv) > 3 else "data/output/Tanzania_modal_lipsync.mp4"

        main(input_video, input_audio, output_video)
    else:
        # Use defaults
        main()
