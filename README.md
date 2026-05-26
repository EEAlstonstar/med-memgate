<div align="center">

# med-memgate

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**A two-tier memory system for LLM agents with intelligent retrieval routing.**

[Overview](#overview) • [Installation](#installation) • [Quick Start](#quick-start) • [Benchmarks](#benchmarks) • [Router Training](#router-training) • [Configuration](#configuration)

</div>

---

## Overview

med-memgate gives LLM agents long-term memory across extended conversations by maintaining two parallel retrieval paths and routing each query to the most appropriate one:

| Path | Name | Mechanism | Best for |
|------|------|-----------|----------|
| **S-path** | Summary Index | Semantic vector search over extracted facts (Mem0) | Fast lookups, general questions |
| **R-path** | Page Store | BM25 keyword search over raw conversation chunks | Exact recall, completeness queries, multi-hop |

A trained **Router** classifies each incoming query and selects the path — balancing response speed against retrieval accuracy.

```
User Query
    │
    ▼
┌─────────────────────┐
│       Router        │  ← classifies query complexity
└──────┬──────┬───────┘
       │S     │R
       ▼      ▼
  Summary   Page
   Index    Store
  (fast)   (precise)
       │      │
       └──────┘
           │
           ▼
       LLM Answer
```

---

## Key Features

- **Intelligent Routing** — trained router (SFT + optional GRPO) selects S or R path per query
- **Dual Retrieval** — semantic search (Mem0 + Qdrant) combined with BM25 (Tantivy)
- **Provenance Tracking** — every answer traces back to source conversation chunks
- **Domain-Aware Routing** — medical terminology rules built into both heuristic and LLM routers
- **Multi-Benchmark Support** — LoCoMo, LongMemEval, MemoryAgentBench, Biomedical
- **Concurrent Evaluation** — session-level parallelism with automatic retry and resume
- **LLM-Agnostic** — any OpenAI-compatible API; router deployable via vLLM

---

## Installation

### Prerequisites

- Python 3.10+
- Qdrant (vector database)
- GPU recommended for router training; CPU sufficient for inference

### Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Qdrant
#    Linux/Mac:
./start_qdrant.sh
#    Windows / Docker:
docker run -d -p 6333:6333 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant

# 3. Set API key
export OPENAI_API_KEY=your_key_here
export OPENAI_BASE_URL=your_base_url   # optional, for custom endpoints
```

> **Windows note:** `start_qdrant.sh` requires bash. Use Docker or download the Qdrant Windows binary from [github.com/qdrant/qdrant/releases](https://github.com/qdrant/qdrant/releases). vLLM is Linux-only; use `--router-type llm` or `--router-type openai` on Windows.

---

## Quick Start

All benchmark scripts share a common set of parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--limit N` | Process only N sessions | all |
| `--max-workers N` | Concurrent session workers | 1 |
| `--model MODEL` | LLM for memory & answer generation | `gpt-4o-mini` |
| `--router-type` | `openai` / `vllm` / `llm` / `binary` | `vllm` |
| `--run-id ID` | Custom identifier for this run | auto |

> Start with `--limit 10` to verify your setup before a full run.

### LoCoMo

```bash
python test_TierMem_locomo_multi.py --limit 10 --max-workers 4
```

### LongMemEval

```bash
python test_TierMem_longmemeval_multi.py --limit 10 --max-workers 4
```

### MemoryAgentBench

```bash
python test_TierMem_memoryagentbench.py \
  --split Accurate_Retrieval \
  --limit 10 \
  --max-workers 4
```

### Biomedical (multi-visit doctor-patient dialogues)

```bash
# Generate sessions first (one-time)
python scripts/biomedical/1_convert_to_sessions.py

# Run benchmark
python test_TierMem_biomedical.py --limit 10 --router-type llm

# Compare baseline vs medical-enhanced router
python test_TierMem_biomedical.py --router-type openai --run-id bio_baseline
python test_TierMem_biomedical.py --router-type openai --medical-addendum --run-id bio_medical
```

---

## Project Structure

```
med-memgate/
├── core/
│   ├── datasets/              # Dataset loaders
│   │   ├── locomo.py
│   │   ├── longmemeval.py
│   │   ├── memory_agent_bench.py
│   │   ├── hotpotqa.py
│   │   ├── halumem.py
│   │   └── biomedical/        # Multi-visit medical QA
│   │       ├── data.json          # 270 raw QA pairs
│   │       ├── sessions.jsonl     # converted multi-turn sessions
│   │       └── ...
│   ├── systems/               # Memory system interfaces (base.py)
│   └── runner/                # Evaluation pipeline
│       ├── write_phase.py         # ingest conversations into memory
│       ├── qa_phase.py            # query memory and score answers
│       ├── scoring.py             # F1, BLEU-1, exact match
│       └── run_benchmark_multi.py # concurrent session executor
│
├── src/
│   ├── linked_view/           # med-memgate core
│   │   ├── router.py              # BinaryRouter, LLMRouter, ThinkingLLMRouter
│   │   ├── prompts.py             # all prompt templates + MEDICAL_ROUTER_ADDENDUM
│   │   ├── summary_index.py       # S-path: Mem0 semantic search
│   │   ├── raw_store.py           # raw conversation storage
│   │   ├── page_store.py          # R-path: BM25 retrieval
│   │   ├── pipelines_fast.py      # S-path answer generation
│   │   ├── pipelines_slow.py      # R-path guided research
│   │   └── api.py                 # unified query entry point
│   ├── memory/
│   │   └── linked_view_system.py  # MemorySystem implementation
│   ├── evaluation/            # LLM-as-Judge scoring
│   └── mem0/                  # modified Mem0 library
│
├── scripts/
│   ├── biomedical/
│   │   └── 1_convert_to_sessions.py   # QA → multi-turn sessions
│   └── router_training/
│       ├── 1_build_offline_dataset.py
│       ├── 2_prepare_sft_data_v2.py
│       ├── 3_prepare_grpo_data.py
│       ├── 4_prepare_medical_sft_data.py  # medical-domain SFT samples
│       └── 5_eval_router_online.py
│
├── test_TierMem_locomo_multi.py
├── test_TierMem_longmemeval_multi.py
├── test_TierMem_memoryagentbench.py
├── test_TierMem_biomedical.py
└── start_qdrant.sh
```

---

## Benchmarks

| Benchmark | Task | Metrics | Script |
|-----------|------|---------|--------|
| **LoCoMo** | Long-context conversational memory QA | F1, BLEU-1 | `test_TierMem_locomo_multi.py` |
| **LongMemEval** | Long-form memory evaluation | F1, Accuracy | `test_TierMem_longmemeval_multi.py` |
| **MemoryAgentBench** | Multi-split agent memory tasks | F1, Accuracy | `test_TierMem_memoryagentbench.py` |
| **Biomedical** | Multi-visit doctor-patient recall | F1 | `test_TierMem_biomedical.py` |
| **HotPotQA** | Multi-hop reasoning | F1, EM | *(coming soon)* |
| **HaluMem** | Hallucination detection | Accuracy | *(coming soon)* |

Results are written to `results/{benchmark}/linked_view/{run_id}/`:

```
results/locomo/linked_view/my_run/
├── sessions/
│   ├── {session_id}_write.jsonl   # memory ingestion log
│   └── {session_id}_qa.jsonl      # per-query answers + scores
└── summary.json                   # aggregated F1 / BLEU / route stats
```

---

## Biomedical Extension

`core/datasets/biomedical/` contains 270 medical knowledge QA pairs converted into multi-visit doctor-patient dialogue sessions. Each session tests three memory retrieval scenarios:

| QA type | Question pattern | Expected route | Rationale |
|---------|-----------------|---------------|-----------|
| `complete` (cat 0) | "What symptoms has the patient mentioned across all visits?" | **R** | Summaries may omit earlier visits |
| `first_visit` (cat 1) | "What did the patient first report?" | **R** | Requires temporal precision |
| `latest` (cat 2) | "What was most recently mentioned?" | **S** | Recent info typically in summary |

### Medical Router Enhancement

The standard router is domain-agnostic. med-memgate adds medical-aware routing rules that push certain query types toward R-path:

- **Lab value queries** (`creatinine`, `WBC count`, `glucose level`) — exact values needed
- **Completeness queries** (`all symptoms`, `list all side effects`) — summaries may truncate
- **Cross-visit recall** (`previously mentioned`, `first reported`, `last visit`) — temporal precision

These rules operate at two levels:

1. **BinaryRouter** (`src/linked_view/router.py`) — heuristic score bonuses (+0.25–0.30) for medical keywords, active with zero configuration
2. **LLM/vLLM Router** — pass `--medical-addendum` flag to inject `MEDICAL_ROUTER_ADDENDUM` into the prompt at runtime

---

## Router Training

The router is a fine-tuned language model that outputs `{"action": "S"}` or `{"action": "R"}` given a query and retrieved summaries.

### Pipeline

```bash
# Step 1 — run both paths, label which was correct
python scripts/router_training/1_build_offline_dataset.py

# Step 2 — format for SFT (S/R binary classification with chain-of-thought)
python scripts/router_training/2_prepare_sft_data_v2.py

# Step 2b — add medical-domain samples
python scripts/router_training/4_prepare_medical_sft_data.py \
  --merge data/router_sft_v2/train.jsonl

# Step 3 — supervised fine-tuning
sbatch scripts/router_training/train_router_sft.sbatch

# Step 4 — GRPO reinforcement learning (optional)
sbatch scripts/router_training/train_router_grpo.sbatch

# Step 5 — serve with vLLM
sbatch scripts/router_training/start_router_vllm.sbatch
```

### SFT Sample Format

```json
{
  "messages": [
    {"role": "user",      "content": "<router prompt with query + summaries>"},
    {"role": "assistant", "content": "<think>\n...\n</think>\n\n{\"action\": \"R\"}"}
  ],
  "metadata": {"query_id": "...", "action": "R", "domain": "biomedical"}
}
```

### GRPO Reward

| s_correct | r_correct | Chose S | Chose R |
|-----------|-----------|---------|---------|
| ✓ | ✓ | +1.0 | +0.6 (waste penalty) |
| ✗ | ✓ | −1.5 | +1.0 |
| ✗ | ✗ | −1.5 | −1.5 |

Before training, update the path placeholders in `.sbatch` files:
- `<PROJECT_ROOT>` — path to this repository
- `<MS_SWIFT_DIR>` — path to [ms-swift](https://github.com/modelscope/swift) installation

Training dependencies (`ms-swift`, `deepspeed`, etc.) are commented out in `requirements.txt`; uncomment before training.

---

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | API key for LLM calls | Yes |
| `OPENAI_BASE_URL` | Custom API endpoint | No |
| `QDRANT_HOST` | Qdrant server hostname | No (default: `localhost`) |
| `QDRANT_PORT` | Qdrant server port | No (default: `6333`) |

### Router Types

| `--router-type` | Description | Requires |
|----------------|-------------|---------|
| `binary` | Heuristic keyword-based, no LLM call | nothing extra |
| `llm` | LLM-based, uses main API client | `OPENAI_API_KEY` |
| `openai` | Dedicated OpenAI client for router | `OPENAI_API_KEY` |
| `vllm` | Locally deployed router model | vLLM server + `--router-base-url` |

---


## Acknowledgments

- [**Mem0**](https://github.com/mem0ai/mem0) — memory management and fact extraction
- [**Qdrant**](https://github.com/qdrant/qdrant) — vector similarity search
- [**ms-swift**](https://github.com/modelscope/swift) — model fine-tuning framework
- [**vLLM**](https://github.com/vllm-project/vllm) — high-throughput LLM inference
