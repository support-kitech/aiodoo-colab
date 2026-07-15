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

Phase 4 discovers / loads `EXP-NNNN` experiments and YAML configs (no training
interpretation).

Phase 5 builds a `TrainingContext` and invokes the public
`aiodoo-training/train.py` entrypoint via subprocess (no ML code in Colab).

Phase 6 provides `notebooks/01_train.ipynb` as a thin orchestration notebook.

---

## Storage layout

### Google Drive (persistent) — `AIODOO/`

```text
AIODOO/
├── datasets/
├── models/
│   ├── base/                   # layout placeholder (HF base models are NOT stored here)
│   ├── adapters/
│   │   └── EXP-0001/           # adapters + checkpoints/
│   ├── merged/
│   │   └── EXP-0001/
│   └── exports/
│       └── EXP-0001/
├── experiments/                 # read-only during training
│   └── EXP-0001/
│       └── config/
├── logs/
│   └── EXP-0001/
└── training/
    └── aiodoo-training/
```

### Colab local SSD (temporary) — Hugging Face base models only

```text
/content/aiodoo-model-cache/
└── Qwen__Qwen3-8B/             # HF id org/name → org__name
```

**Why local SSD for base models?**  
8B (and larger) snapshots are multi‑GB. Caching them on Drive wastes quota and
is slow. Colab’s local disk is fast enough for repeated `ensure()` within a
runtime session.

**Why adapters / checkpoints stay on Drive?**  
They are the durable training outputs that must survive runtime reconnects and
be available across Colab sessions. `aiodoo-training` does not change: Colab
still sets `AIODOO_COLAB_MODEL_PATH` to the resolved local model directory and
points artifact overlays at Drive paths.

**No changes required in `aiodoo-training`.** Path overlays continue to work
exactly as before.

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
load EXP-NNNN
  (Drive experiments/ OR aiodoo-training production configs)
        ↓
ensure model (/content/aiodoo-model-cache/<org>__<name>)
        ↓
build TrainingContext
  (prefers configs/experiments/production/<EXP>/experiment.yaml)
        ↓
subprocess: python3 train.py --config …
  + AIODOO_COLAB_* path overlays
  (AIODOO_COLAB_MODEL_PATH → local SSD model dir; artifacts → Drive)
        ↓
TrainingResult (execution metadata only)
```
Canonical production experiment: **EXP-0001** in aiodoo-training  
(`configs/experiments/production/EXP-0001/`).

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
