"""
Google Cloud Speech-to-Text backend (``speech_v1p1beta1``).

Falls back to the simulator when the package / credentials are missing.
"""

from __future__ import annotations

from .base import BaseASR, ASRResult


class GoogleASR(BaseASR):
    """Google Cloud Speech-to-Text."""

    name = "google_asr"

    def _transcribe_real(self, audio_path: str) -> ASRResult:
        try:
            from google.cloud import speech                  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "google-cloud-speech is required for mode='real'. "
                "Install with: pip install google-cloud-speech") from exc

        client = speech.SpeechClient()
        with open(audio_path, "rb") as fh:
            content = fh.read()
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-IN",
            enable_automatic_punctuation=True,
        )
        response = client.recognize(config=config, audio=audio)
        if not response.results:
            return ASRResult(text="", latency=0.0, confidence=0.0)
        best = response.results[0].alternatives[0]
        return ASRResult(text=best.transcript, latency=0.0,
                         confidence=float(best.confidence))
