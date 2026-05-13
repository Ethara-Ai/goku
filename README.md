# Terra

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset on HF](https://img.shields.io/badge/🤗_Dataset-ethara/terra-yellow.svg)](https://huggingface.co/datasets/ethara/terra)

**Reinforcement learning environment for training agentic LLMs on real-world multi-step tasks with verifiable rewards.**

Terra extends the [GAIA](https://arxiv.org/abs/2311.12983) evaluation methodology into a full RL training loop — producing clean binary reward signals suitable for GRPO-based RLVR training. Built on [OpenHands](https://github.com/OpenHands/OpenHands/) and [Agent SDK](https://github.com/OpenHands/software-agent-sdk).

---

## Key Features

- **10K-task corpus** — GAIA-derived tasks across 3 difficulty levels
- **Sandboxed tool surface** — web search, file I/O, code execution
- **Binary reward oracle** — exact-match scoring for GRPO/RLVR
- **Trace collection** — full action/observation histories for offline training
- **End-to-end pipeline** — inference → scoring → trace export in one harness

---

## How It Works

```
┌─────────────┐     ┌──────────────────┐     ┌────────────────┐     ┌────────────┐
│  GAIA Tasks │────▶│  Agent Execution │────▶│  Reward Oracle │────▶│   Traces   │
│  (10K)      │     │  (sandboxed)     │     │  (exact-match) │     │  (RLVR)    │
└─────────────┘     └──────────────────┘     └────────────────┘     └────────────┘
```

1. **Tasks** — Multi-step questions requiring web search, file processing, and reasoning
2. **Agent execution** — LLM agents solve tasks in sandboxed environments (browser, terminal, code interpreter)
3. **Reward signal** — Binary exact-match against gold answers → clean signal for GRPO
4. **Traces** — Full conversation histories stored for offline RL training

---

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) ≥ 0.8.13

### Installation

```bash
git clone git@github.com:Ethara-Ai/terra.git
cd terra
make build
```

### Configure LLM

```bash
# Create config
cat > .llm_config/your-model.json << 'EOF'
{
  "model": "your-model-name",
  "base_url": "https://your-endpoint",
  "api_key": "YOUR_API_KEY"
}
EOF

# Validate
uv run validate-cfg .llm_config/your-model.json
```

### Run Inference

```bash
TAVILY_API_KEY=xxx uv run gaia-infer .llm_config/your-model.json \
    --level 2023_level1 \
    --split validation \
    --num-workers 4
```

### Score Results

```bash
uv run python -m benchmarks.gaia.get_score --file outputs/gaia/output.jsonl
```

---

## Results

Validation subset (20 tasks), end-to-end with full execution traces:

| Model | Level 1 | Level 2 | Level 3 | Overall |
|-------|---------|---------|---------|---------|
| Kimi K2.5 | 100.0% | 85.7% | 0.0% | **50.0%** |
| Qwen3 VL | 25.0% | 14.3% | 0.0% | 10.0% |

---

## Dataset

**[ethara/terra](https://huggingface.co/datasets/ethara/terra)** on HuggingFace — 10,000 GAIA-derived tasks.

| Field | Description |
|-------|-------------|
| `task_id` | Unique identifier |
| `Question` | Multi-step task prompt |
| `Level` | Difficulty (1–3) |
| `Final answer` | Gold answer for exact-match scoring |
| `file_name` | Attached file (PNG, CSV, XLSX, PDF, MP4) |

Also includes full agent execution traces for Kimi K2.5 and Qwen3 VL.

---

## Project Structure

```
terra/
├── benchmarks/
│   └── gaia/
│       ├── run_infer.py        # Inference pipeline
│       ├── get_score.py        # Exact-match scoring
│       └── README.md           # Benchmark-specific docs
├── vendor/
│   └── software-agent-sdk/     # OpenHands Agent SDK (submodule)
├── .llm_config/                # LLM configuration files
├── outputs/                    # Inference results & traces
├── Makefile                    # Build automation
└── pyproject.toml              # Dependencies & CLI entrypoints
```

---

## Development

```bash
make build       # Install deps + pre-commit hooks
make format      # Format with ruff
make lint        # Lint with ruff
make pre-commit  # Run all pre-commit checks
make clean       # Remove cache files
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Citation

```bibtex
@misc{terra2025,
  title={Terra: RLVR Environment for General AI Assistants},
  author={Ethara AI},
  year={2025},
  url={https://github.com/Ethara-Ai/terra}
}
```

---

## License

MIT — see [LICENSE](LICENSE) for details.
