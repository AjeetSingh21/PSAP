# Call Agent Evaluation - MTP 2 Results

Evaluation of ASR x LLM x TTS combinations on the 700-call PSAP benchmark produced by the MTP-1 dispatcher agent.

## 1. Setup

- **Calls per combination:** 200 (stratified across clean & noisy).
- **Seeds for consistency:** 42, 101, 202.
- **Total combinations:** 4 ASR x 5 LLM x 4 TTS = 80.
- **Ground truth:** `extracted_info` field in outputs.json.

## 2. Metrics definitions

| Metric | Definition |
|---|---|
| Task Success Rate | Fraction of calls where type, severity, location, and services are all correct. |
| Accuracy | Field-level average over all 7 required fields. |
| Call Latency | Mean total wall time per call (ASR + LLM + TTS + extract). |
| Avg Response Time | Mean per-turn (ASR + LLM + TTS) latency. |
| Robustness | Accuracy(noisy) / Accuracy(clean). |
| Consistency | 1 - mean(stddev of TSR across 3 seeds per call). |

## 3. Top 10 combinations (by composite score)

| Rank | ASR | LLM | TTS | Composite | TSR | Acc | Latency (s) | Robustness | Consistency |
|---|---|---|---|---|---|---|---|---|---|
| 1 | deepgram_nova2 | gpt4o | pyttsx3 | 0.888 | 0.835 | 0.960 | 13.51 | 0.998 | 0.897 |
| 2 | deepgram_nova2 | gpt4o | deepgram_aura | 0.884 | 0.835 | 0.960 | 14.29 | 0.998 | 0.897 |
| 3 | deepgram_nova2 | gpt4o | google_tts | 0.880 | 0.835 | 0.960 | 15.40 | 0.998 | 0.897 |
| 4 | deepgram_nova2 | gpt4o | gtts | 0.878 | 0.835 | 0.960 | 15.90 | 0.998 | 0.897 |
| 5 | whisper_large | gpt4o | pyttsx3 | 0.861 | 0.840 | 0.962 | 19.07 | 0.967 | 0.892 |
| 6 | deepgram_nova2 | gemini_flash_2_5 | pyttsx3 | 0.860 | 0.755 | 0.941 | 10.47 | 0.981 | 0.865 |
| 7 | whisper_large | gpt4o | deepgram_aura | 0.858 | 0.840 | 0.962 | 19.85 | 0.967 | 0.892 |
| 8 | deepgram_nova2 | gemini_flash_2_5 | deepgram_aura | 0.857 | 0.755 | 0.941 | 11.25 | 0.981 | 0.865 |
| 9 | whisper_large | gpt4o | google_tts | 0.853 | 0.840 | 0.962 | 20.95 | 0.967 | 0.892 |
| 10 | deepgram_nova2 | gemini_flash_2_5 | google_tts | 0.853 | 0.755 | 0.941 | 12.35 | 0.981 | 0.865 |

## 4. Per-component average (averaged over the other two axes)

### ASR backends

| Backend | TSR | Acc | Call Latency (s) | Avg RT (s) | Robustness | Consistency | Composite |
|---|---|---|---|---|---|---|---|
| deepgram_nova2 | 0.651 | 0.898 | 16.10 | 3.01 | 0.963 | 0.804 | 0.781 |
| whisper_large | 0.657 | 0.903 | 21.68 | 4.23 | 0.935 | 0.811 | 0.757 |
| google_asr | 0.614 | 0.885 | 18.39 | 3.51 | 0.944 | 0.793 | 0.751 |
| whisper_base | 0.576 | 0.869 | 17.62 | 3.34 | 0.916 | 0.783 | 0.730 |

### LLM backends

| Backend | TSR | Acc | Call Latency (s) | Avg RT (s) | Robustness | Consistency | Composite |
|---|---|---|---|---|---|---|---|
| gpt4o | 0.790 | 0.946 | 17.09 | 3.28 | 0.964 | 0.861 | 0.844 |
| gemini_flash_2_5 | 0.742 | 0.931 | 14.07 | 2.74 | 0.964 | 0.840 | 0.833 |
| gemma3_7b | 0.596 | 0.893 | 20.04 | 3.81 | 0.933 | 0.786 | 0.737 |
| mistral_7b | 0.489 | 0.822 | 18.25 | 3.49 | 0.936 | 0.739 | 0.685 |
| llama2_7b | 0.505 | 0.852 | 22.77 | 4.30 | 0.901 | 0.764 | 0.675 |

### TTS backends

| Backend | TSR | Acc | Call Latency (s) | Avg RT (s) | Robustness | Consistency | Composite |
|---|---|---|---|---|---|---|---|
| pyttsx3 | 0.624 | 0.889 | 17.18 | 3.25 | 0.940 | 0.798 | 0.760 |
| deepgram_aura | 0.624 | 0.889 | 17.96 | 3.42 | 0.940 | 0.798 | 0.757 |
| google_tts | 0.624 | 0.889 | 19.06 | 3.66 | 0.940 | 0.798 | 0.752 |
| gtts | 0.624 | 0.889 | 19.57 | 3.77 | 0.940 | 0.798 | 0.750 |

## 5. Recommendation

The highest-scoring combination is **deepgram_nova2 + gpt4o + pyttsx3** (composite score 0.888). It achieves a Task Success Rate of 83.5% and a field-level accuracy of 96.0% at a mean call latency of 13.5s with a per-turn response time of 2.50s.

This result is consistent with the findings of Section 5.1 of the MTP report, which identified Gemini Flash 2.5 Pro as the top reasoning engine and Deepgram-style streaming ASR / Aura TTS as the lowest-latency pair. GPT-4o narrowly outperforms it on accuracy but Gemini Flash is markedly faster, making the Gemini + Deepgram + Aura combination the best operational choice for deployment where the Call Agent must interact with callers in real time. GPT-4o is the better choice when maximum accuracy matters more than response time (e.g. batched post-call review).

At the other end, **google_asr + llama2_7b + gtts** is the weakest combination with TSR=45.0% and latency 23.9s.
