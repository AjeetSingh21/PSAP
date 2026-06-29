"""TTS backends available to the Call Agent."""

from .base import BaseTTS, TTSResult
from .deepgram_tts import DeepgramAuraTTS
from .google_tts import GoogleTTS
from .gtts_tts import GTTSTTS
from .pyttsx3_tts import Pyttsx3TTS


def get_tts(name: str, **kwargs) -> BaseTTS:
    registry = {
        "deepgram_aura": DeepgramAuraTTS,
        "google_tts":    GoogleTTS,
        "gtts":          GTTSTTS,
        "pyttsx3":       Pyttsx3TTS,
    }
    if name not in registry:
        raise ValueError(f"Unknown TTS backend '{name}'. "
                         f"Available: {list(registry)}")
    return registry[name](**kwargs)


__all__ = ["BaseTTS", "TTSResult", "get_tts",
           "DeepgramAuraTTS", "GoogleTTS", "GTTSTTS", "Pyttsx3TTS"]
