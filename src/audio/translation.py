from deep_translator import GoogleTranslator, DeeplTranslator
from typing import List, Dict, Optional
import os


def translate_text(text: str, source_lang: str = "en", target_lang: str = "de",
                   service: str = "google") -> str:
    """
    Translate text from source language to target language.

    Args:
        text: Text to translate
        source_lang: Source language code
        target_lang: Target language code
        service: Translation service ('google' or 'deepl')

    Returns:
        Translated text
    """
    if service == "deepl":
        api_key = os.getenv("DEEPL_API_KEY")
        if not api_key:
            raise ValueError("DEEPL_API_KEY not found in environment")
        translator = DeeplTranslator(api_key=api_key, source=source_lang, target=target_lang)
    else:
        translator = GoogleTranslator(source=source_lang, target=target_lang)

    translated = translator.translate(text)
    return translated


def translate_segments(segments: List[Dict], source_lang: str = "en",
                      target_lang: str = "de", service: str = "google") -> List[Dict]:
    """
    Translate segments while preserving timing information.

    Args:
        segments: List of segments with text and timestamps
        source_lang: Source language code
        target_lang: Target language code
        service: Translation service

    Returns:
        List of segments with translated text
    """
    translated_segments = []

    for segment in segments:
        translated_text = translate_text(
            segment['text'],
            source_lang=source_lang,
            target_lang=target_lang,
            service=service
        )

        translated_segment = segment.copy()
        translated_segment['original_text'] = segment['text']
        translated_segment['text'] = translated_text

        translated_segments.append(translated_segment)

    return translated_segments


def get_full_translation(segments: List[Dict]) -> str:
    """
    Combine all translated segments into single text.

    Args:
        segments: List of translated segments

    Returns:
        Combined translated text
    """
    return " ".join([seg['text'] for seg in segments])