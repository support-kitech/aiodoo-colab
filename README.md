# AIODOO Colab

> Thin Google Colab orchestration for the AIODOO training stack.

---

## Purpose

`aiodoo-colab` integrates **Google Colab**, **Google Drive**, and the frozen
training framework **`aiodoo-training`**.

It mounts Drive, verifies workspace layout, clones or updates
`aiodoo-training`, manages model download locations, loads experiment
configuration, invokes the training CLI / entrypoints, and persists
outputs back to Drive.

**This repository does not train models.** It only orchestrates.

---

## Execution model

This is an **application repository**, not a reusable Python package —
same philosophy as `aiodoo-training`.

- **No** `pip install`
- **No** setuptools / wheels / editable installs
- **No** package namespace (`aiodoo_colab.…`)

Clone the repository and run it directly:

```text
git clone <aiodoo-colab>
        │
        ▼
Open Google Colab (or a local shell)
        │
        ▼
python3 main.py
```

Typical Colab usage: clone into the runtime (or Drive), add ``python/`` to
``PYTHONPATH`` (or ``sys.path``) from the notebook, then execute ``main.py``
from the repository root. Application modules live under ``python/`` and are
imported as flat modules (``from config import …``).

---

## Architecture

```text
Google Colab / main.py
              │
              ▼
        aiodoo-colab          ← THIS repository (orchestration)
              │
              ├── Google Drive workspace paths
              ├── aiodoo-training (cloned / updated)
              └── HuggingFace model cache directories
              │
              ▼
        aiodoo-training       ← owns all training logic (frozen)
```

### What this repository owns

- Drive mount and workspace verification
- Workspace path constants and helpers
- Fetch / update of `aiodoo-training`
- Model download path management (cache directories only)
- Experiment config location / load plumbing
- Launching `aiodoo-training` entrypoints
- Returning artifacts to Drive

### What this repository must never own

- Dataset loading / tokenization
- Model / LoRA / PEFT application logic
- Trainer loops, checkpoints, resume
- Evaluation / export / Artifact Contract
- Training CLI implementation

Those remain permanently in `aiodoo-training`.

---

## Relationship to other AIODOO repositories

| Repository | Role | Relationship to `aiodoo-colab` |
| ---------- | ---- | ------------------------------ |
| `aiodoo-core` | AIODOO framework (frozen) | Upstream product framework; Colab does not embed it |
| `aiodoo-datasets` | Dataset generation (frozen) | Provides versioned datasets consumed via Drive / paths |
| `aiodoo-training` | Training framework (frozen) | **Sole** training engine; Colab only invokes it |
| `aiodoo-models` | Published adapters / models (future) | Receives exports produced *by* `aiodoo-training` |
| `aiodoo-lab` | Experiments & research | May define experiment layouts Colab launches against |

---

## Status

**Phase 0 — Foundation** — application layout, tooling, entrypoint.  
**Phase 1 — Google Drive integration** (complete)  
**Phase 2 — Repository management** (complete)  
**Phase 3 — Model management** (complete)  
**Phase 4 — Experiment management** (complete)  
**Phase 5 — Training integration** (complete)  
**Phase 6 — Colab notebook** (complete)

Phase 1 mounts and verifies Google Drive, locates the AIODOO workspace,
validates required directories, and creates missing folders.

Phase 2 clones or updates `aiodoo-training` from GitHub via subprocess git
commands.

Phase 3 ensures Hugging Face base models under the Colab local SSD cache
``/content/aiodoo-model-cache/<org>__<name>/`` (collision-safe; model id from
experiment). Training artifacts remain on Google Drive.

Phase 4 discovers / loads semantic training ids (`coding`, …) and YAML configs (no training
interpretation).

Phase 5 builds a `TrainingContext` and invokes the public
`aiodoo-training/train.py` entrypoint via subprocess (no ML code in Colab).

Phase 6 provides `notebooks/01_train.ipynb` as a thin orchestration notebook.

---

## Storage layout

### Google Drive (persistent) — `AIODOO/`

Canonical layout authority: `aiodoo-training` `ArtifactOutputLayout`.
Colab sets `AIODOO_WORKSPACE_ROOT`; training derives all artifact paths.

```text
AIODOO/
├── datasets/
├── training/
│   ├── aiodoo-training/        # cloned source (not artifact storage)
│   └── cache/
│       └── coding/
│           └── checkpoints/    # runtime checkpoints
├── models/
│   ├── base/                   # layout placeholder (HF base models are NOT stored here)
│   ├── adapters/
│   │   └── aiodoo-coding/    # published adapter + artifact.json
│   ├── merged/
│   │   └── aiodoo-coding/
│   └── exports/
│       └── aiodoo-coding/
├── experiments/                 # read-only during training
│   └── coding/
│       ├── config/
│       ├── metrics/
│       └── logs/
└── logs/                        # legacy; new runs use experiments/{training_id}/logs/
```

### Colab local SSD (temporary) — Hugging Face base models only

```text
/content/aiodoo-model-cache/
└── Qwen__Qwen3-8B/             # HF id org/name → org__name
    └── artifact.json           # written by training finalize (validation handoff)
```

**Why local SSD for base models?**  
8B (and larger) snapshots are multi‑GB. Caching them on Drive wastes quota and
is slow. Colab’s local disk is fast enough for repeated `ensure()` within a
runtime session.

**Why adapters / checkpoints stay on Drive?**  
They are the durable training outputs that must survive runtime reconnects and
be available across Colab sessions.

**Colab → training contract**

Colab sets:

| Variable | Purpose |
|----------|---------|
| `AIODOO_WORKSPACE_ROOT` | **Required** — AIODOO Drive workspace root |
| `AIODOO_COLAB_MODEL_PATH` | Local SSD base model directory |
| `AIODOO_COLAB_DATASET_PATH` | Dataset version root |

Training rewrites all output paths from `AIODOO_WORKSPACE_ROOT`. Legacy
`AIODOO_COLAB_*_OUTPUT` path hints are not consumed.

**Runtime behavior**

Base Hugging Face models are intentionally **not persisted to Google Drive**.
If the Colab runtime is restarted or reset, the local SSD cache is lost.
On the next training run, `ModelStore.ensure()` automatically verifies the cache
and re-downloads the model from Hugging Face when necessary. Training artifacts
(adapters, checkpoints, logs, exports, and datasets) remain safely stored on
Google Drive.
---

## Training orchestration (`trainer.py`)

```text
prepare workspace
        ↓
ensure aiodoo-training
        ↓
load training id (e.g. coding)
  (Drive experiments/ OR aiodoo-training configs/training/<id>)
        ↓
ensure model (/content/aiodoo-model-cache/<org>__<name>)
        ↓
build TrainingContext
  (prefers configs/training/<training_id>/experiment.yaml)
        ↓
subprocess: python3 train.py --config …
  + AIODOO_WORKSPACE_ROOT (required)
  + AIODOO_COLAB_MODEL_PATH → local SSD model dir
  + AIODOO_COLAB_DATASET_PATH → dataset version root
        ↓
TrainingResult (execution metadata only)
```

Live Colab UI (optional):

```python
from training_ui import TrainingMonitor

monitor = TrainingMonitor(training_id="coding")
monitor.display()
result = run_training(context, on_log_line=monitor.on_line)  # streams logs + widgets
monitor.finish(result)
```

`run_training` streams `train.py` stdout/stderr into the notebook by default
(`stream_output=True`, `PYTHONUNBUFFERED=1`).

Canonical production training: **coding** in aiodoo-training  
(`configs/training/coding/`).

---

## Repository layout

```text
aiodoo-colab/
├── README.md
├── pyproject.toml          # tooling only (ruff / mypy / pytest)
├── .gitignore
├── main.py                 # entrypoint — run directly after clone
├── python/
│   ├── config.py
│   ├── constants.py
│   ├── exceptions.py
│   ├── colab_logging.py    # not named logging.py (avoids shadowing stdlib)
│   ├── version.py
│   ├── workspace.py
│   ├── drive.py
│   ├── repository.py
│   ├── models.py
│   ├── experiments.py
│   ├── trainer.py          # Phase 5 — invoke aiodoo-training only
│   └── launcher.py         # aliases to trainer
├── notebooks/
│   └── 01_train.ipynb      # Phase 6 — thin Colab orchestration
└── tests/
```

---

## Development

Python **3.12**. Install **dev tools only** as needed (pytest, ruff, mypy) —
never install this repository itself.

```bash
python3 main.py
python3 -m pytest
```

`pyproject.toml` configures tooling. It does **not** define a build / package.

---

## Design principles

- Orchestration only — zero training domain logic
- Application layout — execute from clone; no pip package
- Fully typed; PEP 8; Ruff + mypy clean
- `pathlib.Path` for filesystem paths
- Dataclasses where structured config appears (later phases)
- No side effects on import
- Deterministic, production-ready, minimal
