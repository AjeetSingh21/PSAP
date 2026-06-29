"""Llama-2 7B backend served by Ollama."""

from __future__ import annotations

from .gemma import Gemma7BLLM


class Llama27BLLM(Gemma7BLLM):
    """Reuses the Ollama HTTP plumbing, only the model name changes."""
    name = "llama2_7b"
    _model = "llama2:7b"
