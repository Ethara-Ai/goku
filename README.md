<p align="center">
  <h1 align="center">Terra</h1>
  <p align="center">
    <strong>Reinforcement Learning Environment for Training Agentic LLMs</strong>
  </p>
  <p align="center">
    <a href="https://huggingface.co/datasets/ethara/terra">Dataset</a> •
    <a href="https://projects.ethara.ai/terra">Dashboard</a> •
    <a href="https://arxiv.org/abs/2311.12983">Paper</a> •
    <a href="https://ethara.ai">Ethara AI</a>
  </p>
</p>

---

## Terra

A reinforcement learning environment that trains agentic LLMs to solve real-world multi-step tasks using verifiable rewards. Built on the [GAIA](https://arxiv.org/abs/2311.12983) methodology, Terra extends the evaluation framework into a full RL training loop with:

- **10,000-task corpus** — GAIA-derived tasks spanning 3 difficulty levels
- **Sandboxed tool surface** — web search, file I/O, code execution
- **Binary reward oracle** — exact-match scoring designed for GRPO-based RLVR training
- **End-to-end evaluation harness** — inference, scoring, and trace collection in one pipeline

Built on the [OpenHands](https://github.com/OpenHands/OpenHands/) agent framework and [Agent SDK](https://github.com/OpenHands/software-agent-sdk).

---

## How It Works

Terra uses the GAIA benchmark methodology as its foundation:

1. **Tasks** — Multi-step questions requiring web search, file processing, and reasoning
2. **Agent execution** — LLM agents solve tasks using sandboxed tools (browser, terminal, code interpreter)
3. **Reward signal** — Binary exact-match scoring against gold answers (suitable for GRPO/RLVR)
4. **Traces** — Full action/observation histories stored for offline training

The key insight: GAIA's deterministic answers produce clean binary rewards — making it ideal for reinforcement learning from verifiable rewards (RLVR).

---

## Reference Results

Validation subset (20 tasks) evaluated end-to-end with full execution traces:

|  | Level 1 | Level 2 | Level 3 | Overall |
|---|---|---|---|---|
| **Kimi K2.5** | 100.0% | 85.7% | 0.0% | 50.0% |
| **Qwen3 VL** | 25.0% | 14.3% | 0.0% | 10.0% |

---

## Getting Started

### Installation

```bash
git clone git@github.com:Ethara-Ai/terra.git
cd terra
make build
```

### Configure LLM

Create a config at `.llm_config/your-model.json`:

```json
{
  "model": "your-model-name",
  "base_url": "https://your-endpoint",
  "api_key": "YOUR_API_KEY"
}
```

Validate the config:

```bash
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

## Dataset

The curated Terra dataset is available on HuggingFace:

**[ethara/terra](https://huggingface.co/datasets/ethara/terra)** — 10,000 GAIA-derived tasks

Contents:
- Task definitions in JSONL format (3 difficulty levels)
- File attachments (PNG, CSV, XLSX, PDF, MP4)
- Full agent execution traces for Kimi K2.5 and Qwen3 VL

### Task Format

```json
{
  "task_id": "d20ef263-7b51-4fae-9d32-3669faa68cff",
  "Question": "Examine the attached image of a mathematics problem...",
  "Level": 3,
  "Final answer": "905",
  "file_name": "d20ef263-7b51-4fae-9d32-3669faa68cff.png",
  "file_path": "validation/d20ef263-7b51-4fae-9d32-3669faa68cff.png"
}
```

---

## Project Structure

```
terra/
├── benchmarks/
│   └── gaia/              # GAIA evaluation harness
│       ├── run_infer.py   # Inference pipeline
│       ├── get_score.py   # Exact-match scoring
│       └── README.md      # GAIA-specific documentation
├── vendor/
│   └── agent-sdk/         # OpenHands Agent SDK (submodule)
├── .llm_config/           # LLM configuration files
├── outputs/               # Inference results & traces
├── Makefile               # Build automation
└── pyproject.toml         # Dependencies & CLI entrypoints
```

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

Contact [Ethara AI](https://www.ethara.ai) for licensing information.
