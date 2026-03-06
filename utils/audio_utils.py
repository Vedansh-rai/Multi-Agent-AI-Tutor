"""
Audio transcription utilities using OpenAI Whisper (local model).
"""

import os
from utils.logger import get_logger
from utils.config import WHISPER_MODEL_SIZE

logger = get_logger("utils.audio")

# Cache the model globally to avoid reloading
_whisper_model = None


def _load_model():
    """Lazy-load the Whisper model."""
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper

            logger.info("Loading Whisper model: %s", WHISPER_MODEL_SIZE)
            _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
            logger.info("Whisper model loaded successfully")
        except ImportError:
            logger.error("Whisper not installed. Run: pip install openai-whisper")
            raise
    return _whisper_model


def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribe an audio file using Whisper.

    Args:
        audio_path: Path to audio file (WAV, MP3, M4A, etc.)

    Returns:
        dict with keys:
            - transcript (str): Full transcription text.
            - confidence (float): Approximate confidence (1 - avg no_speech_prob).
            - language (str): Detected language code.
            - segments (list): Per-segment details.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        model = _load_model()
        result = model.transcribe(audio_path)

        transcript = result.get("text", "").strip()
        language = result.get("language", "en")
        segments = result.get("segments", [])

        # Estimate confidence from no_speech_prob
        if segments:
            avg_no_speech = sum(
                s.get("no_speech_prob", 0.0) for s in segments
            ) / len(segments)
            confidence = round(1.0 - avg_no_speech, 3)
        else:
            confidence = 0.0

        logger.info(
            "Transcribed %d chars (confidence=%.2f, lang=%s) from %s",
            len(transcript),
            confidence,
            language,
            audio_path,
        )

        return {
            "transcript": transcript,
            "confidence": confidence,
            "language": language,
            "segments": [
                {
                    "text": s.get("text", ""),
                    "start": s.get("start", 0),
                    "end": s.get("end", 0),
                }
                for s in segments
            ],
        }

    except ImportError:
        logger.error("Whisper not installed")
        return {
            "transcript": "[Whisper not installed]",
            "confidence": 0.0,
            "language": "unknown",
            "segments": [],
        }
    except Exception as e:
        logger.error("Transcription failed: %s", str(e))
        return {
            "transcript": f"[Transcription Error: {e}]",
            "confidence": 0.0,
            "language": "unknown",
            "segments": [],
        }
