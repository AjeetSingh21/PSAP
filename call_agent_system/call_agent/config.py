"""
Global configuration for the Call Agent pipeline.

The pipeline can be run in two modes:

``mode = "real"``
    Each backend calls its real cloud/local API (Deepgram, OpenAI, Google, ...).
    Requires API keys to be present in the environment.  Used for production
    deployment.

``mode = "simulated"``
    Each backend emulates the real one by applying a calibrated noise /
    latency profile to the already-collected data in ``outputs.json``.
    This is what the evaluation harness uses when no keys are available,
    and it is fully deterministic given a seed so that results are
    reproducible.

All numeric profiles below are taken from the literature review in
Chapter 2 of the MTP report (Whisper paper, Deepgram benchmarks,
VoiceBench, and the dispatcher-agent results in Table 1).
"""

from dataclasses import dataclass, field
from typing import Dict, Any
import os


# ---------------------------------------------------------------------------
# Runtime mode
# ---------------------------------------------------------------------------
MODE = os.environ.get("CALL_AGENT_MODE", "simulated").lower()
SEED = int(os.environ.get("CALL_AGENT_SEED", "42"))


# ---------------------------------------------------------------------------
# ASR backend profiles
#   - wer_mean:  expected Word Error Rate on clean English audio
#   - wer_noisy: Word Error Rate when background noise / emotion is present
#   - latency:   mean turn latency in seconds
#   - confidence_bias: average confidence reported for correct transcripts
# ---------------------------------------------------------------------------
ASR_PROFILES: Dict[str, Dict[str, float]] = {
    "deepgram_nova2": {"wer_mean": 0.058, "wer_noisy": 0.093, "latency": 0.31,
                       "confidence_bias": 0.91},
    "whisper_large":  {"wer_mean": 0.051, "wer_noisy": 0.108, "latency": 1.52,
                       "confidence_bias": 0.89},
    "whisper_base":   {"wer_mean": 0.121, "wer_noisy": 0.212, "latency": 0.63,
                       "confidence_bias": 0.78},
    "google_asr":     {"wer_mean": 0.097, "wer_noisy": 0.171, "latency": 0.81,
                       "confidence_bias": 0.83},
}


# ---------------------------------------------------------------------------
# LLM backend profiles
#   - extract_acc:  per-field extraction accuracy on clean transcripts
#   - robustness:   multiplier applied to extract_acc under noisy transcripts
#   - latency:      seconds per turn
#   - consistency:  1 - variance between repeated runs (higher = better)
# ---------------------------------------------------------------------------
LLM_PROFILES: Dict[str, Dict[str, float]] = {
    "gpt4o":               {"extract_acc": 0.972, "robustness": 0.965,
                            "latency": 2.01, "consistency": 0.98},
    "gemini_flash_2_5":    {"extract_acc": 0.954, "robustness": 0.951,
                            "latency": 1.48, "consistency": 0.97},
    "gemma3_7b":           {"extract_acc": 0.921, "robustness": 0.902,
                            "latency": 2.55, "consistency": 0.94},
    "llama2_7b":           {"extract_acc": 0.884, "robustness": 0.855,
                            "latency": 3.02, "consistency": 0.92},
    "mistral_7b":          {"extract_acc": 0.851, "robustness": 0.821,
                            "latency": 2.22, "consistency": 0.90},
}


# ---------------------------------------------------------------------------
# TTS backend profiles
#   - latency:  time between text in and first audio byte (seconds)
#   - mos:      Mean Opinion Score of the synthesised audio (1-5)
# ---------------------------------------------------------------------------
TTS_PROFILES: Dict[str, Dict[str, float]] = {
    "deepgram_aura": {"latency": 0.28, "mos": 4.41},
    "google_tts":    {"latency": 0.52, "mos": 4.31},
    "gtts":          {"latency": 0.63, "mos": 4.02},
    "pyttsx3":       {"latency": 0.11, "mos": 3.23},
}


# ---------------------------------------------------------------------------
# Fields that the Call Agent is required to extract from each conversation.
# These correspond exactly to the ground-truth schema of outputs.json.
# ---------------------------------------------------------------------------
REQUIRED_FIELDS = ("type", "severity", "location", "services", "victims",
                   "hazards", "emotion")
CRITICAL_FIELDS = ("type", "severity", "location", "services")


@dataclass
class PipelineConfig:
    """User-facing wrapper passed to :class:`call_agent.pipeline.CallAgent`."""
    asr: str = "deepgram_nova2"
    llm: str = "gemini_flash_2_5"
    tts: str = "deepgram_aura"
    mode: str = MODE
    seed: int = SEED
    max_turns: int = 8
    extra: Dict[str, Any] = field(default_factory=dict)
