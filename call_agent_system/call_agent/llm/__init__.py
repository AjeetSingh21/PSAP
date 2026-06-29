"""LLM backends available to the Call Agent."""

from .base import BaseLLM, LLMResult
from .gpt import GPT4oLLM
from .gemini import GeminiFlashLLM
from .gemma import Gemma7BLLM
from .llama import Llama27BLLM
from .mistral import Mistral7BLLM


def get_llm(name: str, **kwargs) -> BaseLLM:
    """Factory returning a ready-to-use LLM backend."""
    registry = {
        "gpt4o":             GPT4oLLM,
        "gemini_flash_2_5":  GeminiFlashLLM,
        "gemma3_7b":         Gemma7BLLM,
        "llama2_7b":         Llama27BLLM,
        "mistral_7b":        Mistral7BLLM,
    }
    if name not in registry:
        raise ValueError(f"Unknown LLM backend '{name}'. "
                         f"Available: {list(registry)}")
    return registry[name](**kwargs)


__all__ = ["BaseLLM", "LLMResult", "get_llm",
           "GPT4oLLM", "GeminiFlashLLM", "Gemma7BLLM",
           "Llama27BLLM", "Mistral7BLLM"]
