"""
call_agent
==========

The Call Agent is the front-end component of the PSAP (Public Safety
Answering Point) automation pipeline described in MTP Report (Semester 1/2/3).

It wraps three pluggable sub-modules:

    ASR  (Automatic Speech Recognition)   - speech -> text
    LLM  (Large Language Model)           - text   -> structured info / reply
    TTS  (Text-to-Speech)                 - text   -> audio

Each sub-module is designed so that the concrete backend
(e.g. Deepgram vs. Whisper, Gemini vs. Gemma, gTTS vs. Aura) can be swapped
without touching the orchestration layer in :mod:`call_agent.pipeline`.

This enables the evaluation framework in :mod:`evaluation` to run every
combination of (ASR x LLM x TTS) over the 700-call benchmark that was
produced by the dispatcher agent (outputs.json).
"""

from .pipeline import CallAgent                          # noqa: F401
from .extractor import InformationExtractor              # noqa: F401

__version__ = "1.0.0"
__author__ = "Baliji Manikanta (24SE60R04)"
