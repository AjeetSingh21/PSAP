"""Google Cloud Text-to-Speech (Neural2) backend."""

from __future__ import annotations

from typing import Optional

from .base import BaseTTS, TTSResult


class GoogleTTS(BaseTTS):
    name = "google_tts"

    def _speak_real(self, text: str, out_path: Optional[str]) -> TTSResult:
        try:
            from google.cloud import texttospeech            # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "google-cloud-texttospeech required for mode='real'. "
                "pip install google-cloud-texttospeech") from exc
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-IN",
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16)
        resp = client.synthesize_speech(input=synthesis_input,
                                        voice=voice, audio_config=audio_config)
        out = out_path or "/tmp/google_tts.wav"
        with open(out, "wb") as fh:
            fh.write(resp.audio_content)
        return TTSResult(audio_path=out, latency=0.0)
