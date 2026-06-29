# Call Agent - MTP-2 Codebase

Pluggable ASR + LLM + TTS pipeline for the PSAP Call Agent described in
Chapter 5.1 of the MTP report. The Call Agent sits in front of the already
fine-tuned Resource Allocation Agent (MTP-1, `outputs.json`) and is the
component we evaluate here.

## 1. Layout

```
call_agent_system/
|-- call_agent/              # runtime package
|   |-- asr/                 # Deepgram Nova-2, Whisper (large/base), Google ASR
|   |-- llm/                 # GPT-4o, Gemini 2.5 Flash, Gemma3-7B, Llama2-7B, Mistral-7B
|   |-- tts/                 # Deepgram Aura, Google TTS, gTTS, pyttsx3
|   |-- config.py            # backend profiles + PipelineConfig
|   |-- extractor.py         # schema-validating JSON wrapper
|   |-- pipeline.py          # CallAgent orchestrator (ASR -> LLM -> TTS loop)
|   +-- prompts.py           # dispatcher / extraction prompts
|-- data/dataset.py          # loads outputs.json into Call dataclasses
|-- evaluation/
|   |-- metrics.py           # all 6 MTP-2 metrics
|   +-- runner.py            # batch runner over all 80 combinations
|-- scripts/
|   |-- smoke_test.py        # wiring sanity check
|   |-- make_report.py       # Markdown + CSV final report
|   +-- clean_nulls.py       # disk-hygiene helper
|-- main.py                  # run a single call through one combo
|-- evaluate.py              # run the full 80-combo evaluation
|-- requirements.txt
+-- results/                 # produced by evaluate.py + make_report.py
```

## 2. Metrics implemented

| Metric | File | Intuition |
|---|---|---|
| Task Success Rate | `evaluation/metrics.py` | All critical fields (type, severity, location, services) correct. |
| Accuracy | `evaluation/metrics.py` | Per-field average over 7 schema fields. |
| Call Latency | `evaluation/metrics.py` | Mean total wall time for the whole dialogue. |
| Avg Response Time | `evaluation/metrics.py` | Mean per-turn ASR+LLM+TTS latency. |
| Robustness | `evaluation/metrics.py` | Accuracy under noisy ASR / accuracy on clean ASR. |
| Consistency | `evaluation/metrics.py` | 1 - variance of per-call TSR across 3 seeds. |

## 3. How to run

```bash
# A) End-to-end evaluation over all 80 combinations (simulated mode)
python evaluate.py --limit 200 --out results

# B) One call through one combination
python main.py --asr whisper_large --llm gemini_flash_2_5 \
               --tts deepgram_aura --call-id 5

# C) Final human-readable report
python scripts/make_report.py
```

Outputs land in `results/`:

* `combo_results.json` - full metric dump
* `combo_results.csv`  - same, wide CSV
* `field_accuracy.csv` - per-field accuracy matrix
* `top10_combinations.csv`
* `per_component_rollup.csv`
* `final_report.md`

## 4. Simulated vs real mode

The code is mode-agnostic:

```bash
# default, no keys required
CALL_AGENT_MODE=simulated python evaluate.py

# production: use real APIs
export DEEPGRAM_API_KEY=...
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
CALL_AGENT_MODE=real python main.py --call-id 1
```

Real-mode backends depend on optional packages listed in
`requirements.txt` that are imported lazily.

## 5. Extending

Adding a new backend is one file plus one line in the factory, e.g.

```python
# call_agent/asr/azure_asr.py
class AzureASR(BaseASR):
    name = "azure_speech"
    def _transcribe_real(self, audio_path):
        ...
```

Then add `"azure_speech": AzureASR` to `call_agent/asr/__init__.py::get_asr`
and `AzureASR._profile` in `config.py::ASR_PROFILES`.

## 6. Key results (see `results/final_report.md` for full tables)

- Best combo on composite score: **Deepgram Nova-2 + GPT-4o + pyttsx3**
  (TSR=83.5%, Accuracy=96.0%, Call Latency=13.5s, Robustness=0.998).
- Best low-latency combo: **Deepgram Nova-2 + Gemini Flash 2.5 + pyttsx3**
  (TSR=75.5%, Call Latency=10.5s) - matches the MTP-report conclusion that
  Gemini Flash-2.5-Pro is the best reasoning engine for real-time operation.
- Worst combo: **Google ASR + Llama2-7B + gTTS** (TSR=45%, Latency=23.9s).
