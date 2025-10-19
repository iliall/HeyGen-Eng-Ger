import click
from pathlib import Path
from dotenv import load_dotenv
import json

from src.video.extractor import extract_audio, get_video_duration
from src.audio.transcription import transcribe_audio, get_segments
from src.audio.transcription import save_transcription as save_transcription_file
from src.audio.translation import translate_segments
from src.audio.synthesis import synthesize_segments, prepare_voice_samples, clone_voice as clone_voice_api
from src.audio.utils import merge_time_aligned_segments
from src.video.merger import merge_audio_video
from src.video.synchronization import get_audio_duration, calculate_duration_mismatch
from src.utils.logger import setup_logger
from src.utils.file_handler import ensure_dir, cleanup_temp_files, get_output_path

load_dotenv()


@click.command()
@click.argument('input_video', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output video path')
@click.option('--source-lang', default='en', help='Source language code (default: en)')
@click.option('--target-lang', default='de', help='Target language code (default: de)')
@click.option('--voice-id', default='21m00Tcm4TlvDq8ikWAM', help='ElevenLabs voice ID (ignored if --clone-voice is used)')
@click.option('--clone-voice', is_flag=True, help='Clone voice from original video')
@click.option('--voice-name', default=None, help='Name for cloned voice (default: speaker name from video)')
@click.option('--whisper-model', default='base', help='Whisper model size (tiny/base/small/medium/large)')
@click.option('--translation-service', default='google', help='Translation service (google/deepl)')
@click.option('--temp-dir', default='data/temp', help='Temporary files directory')
@click.option('--keep-temp', is_flag=True, help='Keep temporary files after processing')
@click.option('--save-transcription', is_flag=True, help='Save transcription to JSON file')
def translate_video(input_video, output, source_lang, target_lang, voice_id, clone_voice, voice_name,
                   whisper_model, translation_service, temp_dir, keep_temp, save_transcription):
    """
    Translate video from one language to another while preserving voice characteristics.

    Example:
        python -m src.main input.mp4 -o output.mp4 --target-lang de
    """
    logger = setup_logger('video_translation')

    logger.info(f"Starting video translation: {input_video}")
    logger.info(f"Source: {source_lang} -> Target: {target_lang}")

    # Setup paths
    input_path = Path(input_video)
    if output is None:
        output = get_output_path(str(input_path), input_path.parent, suffix=f"_{target_lang}")

    temp_path = Path(temp_dir)
    ensure_dir(temp_path)

    try:
        # Step 1: Extract audio from video
        logger.info("Step 1/6: Extracting audio from video...")
        audio_path = temp_path / f"{input_path.stem}_audio.wav"
        extract_audio(str(input_path), str(audio_path))
        original_duration = get_video_duration(str(input_path))
        logger.info(f"  Audio extracted: {audio_path} ({original_duration:.2f}s)")

        # Step 2: Transcribe audio to text
        logger.info(f"Step 2/6: Transcribing audio ({whisper_model} model)...")
        transcription = transcribe_audio(str(audio_path), model_size=whisper_model, language=source_lang)
        segments = get_segments(transcription)
        logger.info(f"  Transcribed {len(segments)} segments")

        if save_transcription:
            transcription_path = temp_path / f"{input_path.stem}_transcription.json"
            save_transcription_file(transcription, str(transcription_path))
            logger.info(f"  Transcription saved: {transcription_path}")

        # Step 3: Translate text to target language
        logger.info(f"Step 3/6: Translating text to {target_lang}...")
        translated_segments = translate_segments(
            segments,
            source_lang=source_lang,
            target_lang=target_lang,
            service=translation_service
        )
        logger.info(f"  Translated {len(translated_segments)} segments")

        # Step 3.5: Clone voice if requested
        if clone_voice:
            logger.info("Step 3.5/6: Cloning voice from original audio...")

            # Extract voice samples from longest segments
            voice_samples_dir = temp_path / "voice_samples"
            voice_samples = prepare_voice_samples(
                audio_path=str(audio_path),
                segments=segments,
                output_dir=str(voice_samples_dir),
                max_samples=3
            )
            logger.info(f"  Extracted {len(voice_samples)} voice samples")

            # Clone the voice
            if voice_name is None:
                voice_name = f"{input_path.stem}_voice"

            voice_id = clone_voice_api(
                name=voice_name,
                audio_files=voice_samples,
                description=f"Cloned voice from {input_path.name}"
            )
            logger.info(f"  Voice cloned successfully: {voice_id}")

        # Step 4: Synthesize translated audio
        logger.info(f"Step 4/6: Synthesizing {target_lang} audio...")
        logger.info(f"  Using voice ID: {voice_id}")
        segments_dir = temp_path / "segments"
        audio_files = synthesize_segments(
            segments=translated_segments,
            voice_id=voice_id,
            output_dir=str(segments_dir)
        )
        logger.info(f"  Generated {len(audio_files)} audio segments")

        # Step 5: Merge audio segments with time-stretching
        logger.info("Step 5/6: Merging audio segments with time alignment...")
        merged_audio_path = temp_path / f"{input_path.stem}_{target_lang}_audio.wav"
        merge_time_aligned_segments(audio_files, translated_segments, str(merged_audio_path))
        logger.info("  Time-stretched each segment to match original timing")

        # Check duration mismatch
        new_duration = get_audio_duration(str(merged_audio_path))
        mismatch = calculate_duration_mismatch(original_duration, new_duration)
        logger.info(f"  Merged audio: {merged_audio_path} ({new_duration:.2f}s)")
        logger.info(f"  Duration difference: {mismatch['difference']:.2f}s ({mismatch['percentage']:.1f}%)")

        if mismatch['needs_adjustment']:
            logger.warning(f"  ⚠ Large duration mismatch detected. Consider using time-stretching.")

        # Step 6: Replace video audio with translated audio
        logger.info("Step 6/6: Merging translated audio with video...")
        merge_audio_video(str(input_path), str(merged_audio_path), output)
        logger.info(f"  ✓ Translation complete: {output}")

        # Show summary
        logger.info("\n" + "="*60)
        logger.info("TRANSLATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Input:              {input_video}")
        logger.info(f"Output:             {output}")
        logger.info(f"Language:           {source_lang} → {target_lang}")
        logger.info(f"Segments:           {len(translated_segments)}")
        logger.info(f"Original duration:  {original_duration:.2f}s")
        logger.info(f"Translated duration: {new_duration:.2f}s")
        logger.info(f"Difference:         {mismatch['difference']:.2f}s ({mismatch['percentage']:.1f}%)")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Error during translation: {e}")
        raise

    finally:
        # Cleanup temporary files
        if not keep_temp:
            logger.info("Cleaning up temporary files...")
            cleanup_temp_files(str(temp_path))
        else:
            logger.info(f"Temporary files kept in: {temp_path}")


if __name__ == '__main__':
    translate_video()