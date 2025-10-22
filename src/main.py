import click
from pathlib import Path
from dotenv import load_dotenv
import json

from src.video.extractor import extract_audio, get_video_duration
from src.audio.transcription import transcribe_audio, get_segments
from src.audio.transcription import merge_segments as merge_segments_func
from src.audio.transcription import save_transcription as save_transcription_file
from src.audio.translation import translate_segments
from src.audio.synthesis import synthesize_segments, prepare_voice_samples, clone_voice as clone_voice_api, get_forced_alignment, align_translated_words, create_word_level_segments
from src.audio.utils import merge_time_aligned_segments, merge_word_level_segments
from src.audio.separation import separate_audio, is_separation_available
from src.video.merger import merge_audio_video
from src.video.synchronization import get_audio_duration, calculate_duration_mismatch
from src.audio.srt_parser import parse_srt_file, save_segments_as_srt, validate_srt_segments
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
@click.option('--stability', default=0.5, type=float, help='Voice stability 0-1 (default: 0.5, lower=more emotional)')
@click.option('--similarity-boost', default=0.8, type=float, help='Similarity to cloned voice 0-1 (default: 0.8)')
@click.option('--style', default=0.4, type=float, help='Style exaggeration 0-1 (default: 0.4)')
@click.option('--speaker-boost/--no-speaker-boost', default=True, help='Boost similarity to original speaker')
@click.option('--temp-dir', default='data/temp', help='Temporary files directory')
@click.option('--keep-temp', is_flag=True, help='Keep temporary files after processing')
@click.option('--save-transcription', is_flag=True, help='Save transcription to JSON file')
@click.option('--srt-input', type=click.Path(exists=True), help='Use SRT file instead of audio transcription')
@click.option('--save-srt', is_flag=True, help='Save translated subtitles as SRT file')
@click.option('--word-level-timing', is_flag=True, help='Use ElevenLabs forced alignment for word-level timing (experimental)')
@click.option('--no-background', is_flag=True, help='Skip voice separation (faster, assumes no background audio)')
@click.option('--background-enhancement/--no-background-enhancement', default=True, help='Apply noise reduction to separated background audio')
def translate_video(input_video, output, source_lang, target_lang, voice_id, clone_voice, voice_name,
                   whisper_model, translation_service, stability, similarity_boost, style, speaker_boost,
                   temp_dir, keep_temp, save_transcription, srt_input, save_srt, word_level_timing, no_background, background_enhancement):
    """
    Translate video from one language to another while preserving voice characteristics.

    Examples:
        python -m src.main input.mp4 -o output.mp4 --target-lang de
        python -m src.main input.mp4 --srt-input subtitles.srt --save-srt
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
        logger.info("Step 1/7: Extracting audio from video...")
        audio_path = temp_path / f"{input_path.stem}_audio.wav"
        extract_audio(str(input_path), str(audio_path))
        original_duration = get_video_duration(str(input_path))
        logger.info(f"  Audio extracted: {audio_path} ({original_duration:.2f}s)")

        # Step 1.5: Separate audio into vocals and background (optional)
        vocals_path = None
        background_path = None

        if no_background:
            logger.info("Step 1.5/7: Skipping voice separation (--no-background specified)")
            logger.info("  Using original audio directly (faster processing)")
        elif is_separation_available():
            try:
                logger.info("Step 1.5/7: Separating vocals and background audio...")
                separation_dir = temp_path / "separation"
                result = separate_audio(
                    audio_path=str(audio_path),
                    output_dir=str(separation_dir),
                    quality="balanced",
                    use_background_enhancement=background_enhancement
                )
                if result:
                    vocals_path, background_path = result
                    logger.info(f"  Vocals extracted: {Path(vocals_path).name}")
                    logger.info(f"  Background extracted: {Path(background_path).name}")
                else:
                    logger.warning("  Audio separation returned no result, continuing without background preservation")
            except Exception as e:
                logger.warning(f"  Audio separation failed: {e}")
                logger.warning("  Continuing without background preservation...")
        else:
            logger.info("Step 1.5/7: Audio separation not available (install demucs to preserve background)")

        # Step 2: Transcribe audio to text or parse SRT
        if srt_input:
            logger.info(f"Step 2/7: Parsing SRT file: {srt_input}")
            segments = parse_srt_file(srt_input)

            # Validate SRT segments
            validation = validate_srt_segments(segments)
            if not validation['valid']:
                logger.error(f"  SRT validation failed: {validation['errors']}")
                raise ValueError("Invalid SRT file format")
            if validation['warnings']:
                logger.warning(f"  SRT warnings: {validation['warnings']}")

            logger.info(f"  Parsed {len(segments)} segments from SRT")
        else:
            # Use clean vocals for transcription if available (better quality)
            transcription_audio = vocals_path if vocals_path else str(audio_path)
            audio_source = "clean vocals" if vocals_path else "original audio"
            logger.info(f"Step 2/7: Transcribing {audio_source} ({whisper_model} model)...")
            transcription = transcribe_audio(transcription_audio, model_size=whisper_model, language=source_lang)
            segments = get_segments(transcription)
            logger.info(f"  Transcribed {len(segments)} segments from {audio_source}")

            # Always merge segments with <= 5 words for better audio quality
            if len(segments) > 1:
                original_count = len(segments)
                segments = merge_segments_func(segments, min_words=5)
                merged_count = len(segments)
                logger.info(f"  Merged segments: {original_count} → {merged_count}")

            if save_transcription:
                transcription_path = temp_path / f"{input_path.stem}_transcription.json"
                save_transcription_file(transcription, str(transcription_path))
                logger.info(f"  Transcription saved: {transcription_path}")

        # Step 3: Translate text to target language
        logger.info(f"Step 3/7: Translating text to {target_lang}...")
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

            # Use clean vocals for voice cloning if available (better quality)
            cloning_audio_path = vocals_path if vocals_path else str(audio_path)
            audio_source = "clean vocals" if vocals_path else "original audio"
            logger.info(f"  Using {audio_source} for voice cloning")

            # Extract voice samples from longest segments
            voice_samples_dir = temp_path / "voice_samples"
            voice_samples = prepare_voice_samples(
                audio_path=cloning_audio_path,
                segments=segments,
                output_dir=str(voice_samples_dir),
                max_samples=3
            )
            logger.info(f"  Extracted {len(voice_samples)} voice samples from {audio_source}")

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
        logger.info(f"Step 4/7: Synthesizing {target_lang} audio...")
        logger.info(f"  Using voice ID: {voice_id}")
        logger.info(f"  Voice settings: stability={stability}, similarity={similarity_boost}, style={style}")
        segments_dir = temp_path / "segments"
        audio_files = synthesize_segments(
            segments=translated_segments,
            voice_id=voice_id,
            output_dir=str(segments_dir),
            stability=stability,
            similarity_boost=similarity_boost,
            style=style,
            use_speaker_boost=speaker_boost
        )
        logger.info(f"  Generated {len(audio_files)} audio segments")

        # Step 5: Merge audio segments with time-stretching
        logger.info("Step 5/7: Merging audio segments with time alignment...")
        merged_audio_path = temp_path / f"{input_path.stem}_{target_lang}_audio.wav"

        if word_level_timing:
            logger.info("  Using word-level timing with forced alignment...")
            try:
                # Get forced alignment for original audio
                original_text = ' '.join([seg['text'] for seg in segments])
                original_alignment = get_forced_alignment(str(audio_path), original_text)
                logger.info(f"  Got forced alignment for {len(original_alignment.get('words', []))} words")

                # Align translated words with original timing
                translated_text = ' '.join([seg['text'] for seg in translated_segments])
                aligned_words = align_translated_words(original_alignment, translated_text, original_text)
                logger.info(f"  Aligned {len(aligned_words)} translated words")

                # Use word-level merging
                merge_word_level_segments(audio_files, translated_segments, aligned_words, str(merged_audio_path))
                logger.info("  Applied word-level time-stretching for perfect alignment")

            except Exception as e:
                logger.warning(f"  Word-level alignment failed: {e}")
                logger.info("  Falling back to segment-level alignment...")
                merge_time_aligned_segments(audio_files, translated_segments, str(merged_audio_path))
                logger.info("  Time-stretched each segment to match original timing")
        else:
            merge_time_aligned_segments(audio_files, translated_segments, str(merged_audio_path))
            logger.info("  Time-stretched each segment to match original timing")

        # Check duration mismatch
        new_duration = get_audio_duration(str(merged_audio_path))
        mismatch = calculate_duration_mismatch(original_duration, new_duration)
        logger.info(f"  Merged audio: {merged_audio_path} ({new_duration:.2f}s)")
        logger.info(f"  Duration difference: {mismatch['difference']:.2f}s ({mismatch['percentage']:.1f}%)")

        if mismatch['needs_adjustment']:
            logger.warning(f"  ⚠ Large duration mismatch detected. Consider using time-stretching.")

        final_audio_path = merged_audio_path

        # Step 6: Replace video audio with translated audio
        logger.info("Step 6/7: Merging translated audio with video...")
        merge_audio_video(str(input_path), str(final_audio_path), output)
        logger.info(f"  ✓ Translation complete: {output}")

        # Step 7: Save SRT file if requested
        if save_srt:
            output_path = Path(output)
            srt_output_path = output_path.with_suffix('.srt')
            save_segments_as_srt(translated_segments, str(srt_output_path))
            logger.info(f"  ✓ Subtitles saved: {srt_output_path}")

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