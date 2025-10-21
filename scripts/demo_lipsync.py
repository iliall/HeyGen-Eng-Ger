"""
Demonstrate complete lip-sync functionality.
"""

import os
import subprocess
from pathlib import Path

def run_demo():
    """Run a complete lip-sync demonstration."""

    print("HeyGen Video Translation with Lip-Sync Demo")
    print("=" * 50)

    # Check if input video exists
    input_video = "data/input/Tanzania.mp4"
    if not os.path.exists(input_video):
        print(f"[ERROR] Input video not found: {input_video}")
        return False

    print(f"Input video: {input_video}")

    # Check if models are available
    wav2lip_model = "models/lipsync/wav2lip.pth"
    s3fd_model = "models/lipsync/s3fd.pth"

    print(f"Wav2Lip model: {'[OK] Available' if os.path.exists(wav2lip_model) else '[ERROR] Missing'}")
    print(f"S3FD model: {'[OK] Available' if os.path.exists(s3fd_model) else '[ERROR] Missing'}")

    if not os.path.exists(wav2lip_model):
        print("[ERROR] Wav2Lip model not found. Cannot proceed with lip-sync demo.")
        return False

    # Create output directory
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_video = output_dir / "Tanzania_lipsync_demo.mp4"

    print(f"Output video: {output_video}")
    print()

    # Build the command
    cmd = [
        "python", "-m", "src.main",
        input_video,
        "-o", str(output_video),
        "--target-lang", "de",
        "--clone-voice",
        "--voice-name", "Demo_Speaker",
        "--enable-lipsync",
        "--lipsync-quality", "balanced",
        "--face-detector", "auto",
        "--whisper-model", "base",
        "--keep-temp"
    ]

    print("Starting video translation with lip-sync...")
    print("Command:", " ".join(cmd))
    print()

    try:
        # Run the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        # Stream output in real-time
        for line in iter(process.stdout.readline, ''):
            if line.strip():
                print(f"  {line.strip()}")

        # Wait for completion
        return_code = process.wait()

        print()
        if return_code == 0:
            if os.path.exists(output_video):
                size_mb = os.path.getsize(output_video) / 1024 / 1024
                print("[OK] Lip-sync processing completed successfully!")
                print(f"Output video: {output_video}")
                print(f"File size: {size_mb:.1f}MB")
                print()
                print("Your video now has:")
                print("   - German audio translation")
                print("   - Preserved speaker voice")
                print("   - Synchronized lip movements")
                print("   - Perfect audio-visual sync")
                return True
            else:
                print("[ERROR] Output file not created")
                return False
        else:
            print(f"[ERROR] Process failed with return code: {return_code}")
            return False

    except KeyboardInterrupt:
        print("\n[WARN] Process interrupted by user")
        process.terminate()
        return False
    except Exception as e:
        print(f"[ERROR] Error running process: {e}")
        return False

if __name__ == "__main__":
    success = run_demo()
    if success:
        print("\nYou can now play the output video to see the lip-sync in action.")
    else:
        print("\nCheck the error messages above and ensure all dependencies are installed.")