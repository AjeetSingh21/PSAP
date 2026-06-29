"""ASR backends available to the Call Agent."""

from .base import BaseASR, ASRResult
from .deepgram_asr import DeepgramASR
from .whisper_asr import WhisperLargeASR, WhisperBaseASR
from .google_asr import GoogleASR


def get_asr(name: str, **kwargs) -> BaseASR:
    """Factory that returns a ready-to-use ASR backend."""
    registry = {
        "deepgram_nova2": DeepgramASR,
        "whisper_large":  WhisperLargeASR,
        "whisper_base":   WhisperBaseASR,
        "google_asr":     GoogleASR,
    }
    if name not in registry:
        raise ValueError(f"Unknown ASR backend '{name}'. "
                         f"Available: {list(registry)}")
    return registry[name](**kwargs)


__all__ = ["BaseASR", "ASRResult", "get_asr",
           "DeepgramASR", "WhisperLargeASR", "WhisperBaseASR", "GoogleASR"]
