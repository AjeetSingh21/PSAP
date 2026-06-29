"""Mistral 7B backend served by Ollama."""

from __future__ import annotations

from .gemma import Gemma7BLLM


class Mistral7BLLM(Gemma7BLLM):
    name = "mistral_7b"
    _model = "mistral:7b"
