# Task template

A minimal task that conforms to the Goku dataset contract. Copy this folder,
rename it to `task_<your_hash>`, and edit the three pieces below.

## Layout

```
task_template/
├── instruction.md          # the literal user prompt
├── rubrics.jsonl           # one rubric item per line
├── data/
│   └── input_files/        # ALL media the agent receives at runtime
│       └── .gitkeep        # placeholder (delete after adding real assets)
└── README.md               # this file (you can delete it in your own tasks)
```

## How to customize

1. **`instruction.md`** — Rewrite the prompt as a natural human message.
   - Use **bare filenames** only (e.g. `item_id.json`, not `results/item_id.json`).
   - Do NOT bake harness paths (`/workspace/`, `results/`, `/home/`) into the prompt.
   - If you name an output filename explicitly, the rubric can use a
     deterministic `probe_file_exists`/`probe_file_contains` check. If you
     leave the filename open ("save it as a file"), the rubric must use
     LLM-judged `response_criteria` instead.

2. **`rubrics.jsonl`** — One JSON object per line. Replace
   `<REPLACE_WITH_YOUR_IMAGE_FILENAME>` placeholders in the `source` fields
   with the actual filenames you place in `data/input_files/`.

3. **`data/input_files/`** — Put your images, PDFs, or videos here. The
   harness uploads exactly this folder's contents to the agent's workspace
   at runtime. Delete `.gitkeep` once you've added real assets.

## Rubric anatomy

Each line of `rubrics.jsonl` is a JSON object with these fields:

| Field | Required | Meaning |
|---|---|---|
| `number` | yes | 1-based item index |
| `type` | yes | One of: `probe_file_exists`, `probe_file_contains`, `probe_dir_exists`, `shell_succeeds_real`, `response_contains`, `response_regex_present`, `response_criteria`, `response_not_criteria` |
| `category` | yes | One of: `CORRECTNESS`, `FORMAT`, `BEHAVIOR`, `MM_REASONING`, `HALLUCINATION`, `STYLE` |
| `points` | yes | `+5` or `+3` for positives; `-5` or `-3` for hallucination penalties |
| `importance` | yes | `mandatory` (gates pass/fail) or `nice_to_have` (bonus only) |
| `criterion` | yes | Natural-language assertion |
| `paths` | type-dep | List of bare filenames for `probe_file_exists` / `probe_dir_exists` |
| `path`, `pattern` | type-dep | For `probe_file_contains` |
| `raw_shell` | type-dep | Bash one-liner for `shell_succeeds_real` (pass = exit 0) |
| `needles` | type-dep | List of substrings for `response_contains` |
| `pattern` | type-dep | Regex for `response_regex_present` |
| `source` | optional | For factuality items, cite which asset/region/quote supports the criterion |

## Validate before delivering

```bash
uv run python -c "
from pathlib import Path
from benchmarks.goku.task_loader import discover_tasks
tasks = discover_tasks(Path('sample_tasks'))
for t in tasks:
    print(t.id, '->', len(t.rubric_items), 'rubrics,', len(t.input_files), 'files')
"
```

If the task is malformed, the loader will raise a clear error naming the
offending field.
