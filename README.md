# AIODOO Colab

> Thin Google Colab orchestration for the AIODOO ecosystem: training,
> validation, and model packaging.
>
> **Current:** default training git ref is branch **`main`** (Git tags were removed; `main` is SoT).

---

## Purpose

`aiodoo-colab` integrates **Google Colab**, **Google Drive**, and three
frozen canonical frameworks:

- **`aiodoo-training`** — launches training (subprocess, pinned tag)
- **`aiodoo-validation`** — runs certification evaluations (in-process import)
- **`aiodoo-model`** — packages/publishes adapters (in-process import)

It mounts Drive, verifies workspace layout, clones or updates
`aiodoo-training`, manages model download locations, loads experiment
configuration, invokes the training CLI / entrypoints, runs evaluations,
publishes adapters into the model registry, and persists everything back to
Drive.

**This repository does not train models, validate/certify models, or
package/merge adapters.** It only orchestrates the repositories that do. See
`docs/archive/ECOSYSTEM_ADOPTION.md` for the full Phase 8 integration writeup.

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
Google Colab / main.py / notebooks/01_train.ipynb
              │
              ▼
        aiodoo-colab          ← THIS repository (orchestration)
              │
              ├── Google Drive workspace paths (canonical layout)
              ├── aiodoo-training (cloned / updated; subprocess)
              ├── aiodoo-validation (in-process import)
              ├── aiodoo-model (in-process import)
              └── HuggingFace model cache directories
              │
    ┌─────────┼──────────────┬──────────────────┐
    ▼         ▼               ▼                  ▼
aiodoo-      aiodoo-       aiodoo-model      Google Drive
training     validation    (registry +       (canonical
(training    (evaluation/  storage)          directory
logic)       certification)                  structure)
```

### What this repository owns

- Drive mount and workspace verification
- Workspace path constants and helpers (single canonical layout authority)
- Fetch / update of `aiodoo-training`
- Model download path management (cache directories only)
- Experiment selection, configuration location / load plumbing
- Launching `aiodoo-training` entrypoints (with resume support)
- Invoking `aiodoo-validation` evaluations for a completed training run
- Invoking `aiodoo-model` to publish/resolve/materialize adapters
- Read-only checkpoint / artifact discovery and browsing
- Progress display, logging, and notebook UX
- Returning artifacts to Drive

### What this repository must never own

- Dataset loading / tokenization
- Model / LoRA / PEFT application logic
- Trainer loops, optimizer/scheduler setup, resource planning
- Behavioral/structural validation, scoring, certification, reporting logic
- Adapter composition, merge planning, package metadata, export/compatibility logic
- Checkpoint *content* validation (RNG state, optimizer state, fingerprints)

Those remain permanently in `aiodoo-training`, `aiodoo-validation`, and
`aiodoo-model` respectively — see `docs/archive/ECOSYSTEM_ADOPTION.md`.

---

## Relationship to other AIODOO repositories

| Repository | Role | Relationship to `aiodoo-colab` |
| ---------- | ---- | ------------------------------ |
| `aiodoo-core` | Runtime foundation (frozen) | Upstream product framework; Colab does not embed it |
| `aiodoo-datasets` | Dataset generation (frozen) | Provides versioned datasets on Drive / paths; Colab reads dataset paths, never generates them |
| `aiodoo-validation` | Certification profiles (frozen) | **Sole** evaluation engine; Colab imports `aiodoo_validation.api` in-process (`python/validation.py`) |
| `aiodoo-training` | Training framework (frozen @ `v2.0.0`) | **Sole** training engine; Colab clones and invokes it via subprocess (`python/trainer.py`) |
| `aiodoo-model` | Canonical packaging/registry (frozen) | **Sole** packaging/registry engine; Colab imports `aiodoo_model` in-process to publish/resolve/materialize (`python/packaging.py`) |
| `aiodoo-vscode` | Thin-client IDE scaffold | Unrelated to Colab orchestration |

**Phase 8:** `aiodoo-colab` is the canonical orchestration environment for
training, validation, model packaging, and experiments — see
`docs/archive/ECOSYSTEM_ADOPTION.md` for the full integration writeup, duplication
removed/retained, and reliability improvements.

---

## Status

**Phase 0 — Foundation** — application layout, tooling, entrypoint.  
**Phase 1 — Google Drive integration** (complete)  
**Phase 2 — Repository management** (complete)  
**Phase 3 — Model management** (complete)  
**Phase 4 — Experiment management** (complete)  
**Phase 5 — Training integration** (complete)  
**Phase 6 — Colab notebook** (complete)  
**Phase 8 — Ecosystem adoption: validation + model packaging + resume/artifacts** (complete)

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

Phase 8 adds: resume-aware training launch (`trainer.resolve_resume_checkpoint`
/ `prepare_resume_config`), read-only checkpoint/artifact discovery
(`artifacts.py`), evaluation orchestration via `aiodoo-validation`
(`validation.py`), and adapter publishing/resolution via `aiodoo-model`
(`packaging.py`) — see `docs/archive/ECOSYSTEM_ADOPTION.md`.

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
│           └── checkpoints/    # runtime checkpoints (workspace.checkpoints_root)
│               └── checkpoint-<step>/
├── models/
│   ├── base/                   # layout placeholder (HF base models are NOT stored here)
│   ├── adapters/
│   │   └── aiodoo-coding/    # published adapter + artifact.json (aiodoo-training output)
│   ├── merged/
│   │   └── aiodoo-coding/
│   ├── exports/
│   │   └── aiodoo-coding/
│   ├── registry/               # aiodoo-model FileBackedRegistry (Release/Artifact records)
│   └── registry_storage/       # aiodoo-model StorageManager blob storage
├── experiments/                 # read-only during training
│   └── coding/
│       ├── config/
│       ├── metrics/
│       └── logs/
└── logs/                        # legacy; new runs use experiments/{training_id}/logs/
```

`registry/` and `registry_storage/` are new in Phase 8
(`workspace.Workspace.model_registry` / `.model_registry_storage`) — they are
the Drive-backed roots `packaging.ModelRegistry` opens
`aiodoo_model.registry.FileBackedRegistry` /
`aiodoo_model.storage.StorageManager` against; this repository never writes
into them directly.

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

**Resume** (`auto_resume=True` or an explicit `resume_from=<checkpoint dir>`):
`resolve_resume_checkpoint()` locates the latest (or requested) checkpoint
under `workspace.checkpoints_root(training_id)`, and `prepare_resume_config()`
writes a scratch copy of the training config with
`checkpointing.resume_from` injected — the original config file on Drive is
never mutated. All actual resume validation (RNG state, optimizer state,
model fingerprint) is `aiodoo-training`'s own `ResumeCoordinator`; Colab only
decides *whether to try* (see `artifacts.is_resumable`).

## Validation orchestration (`validation.py`)

```text
completed TrainingResult (result.success must be True — fails closed otherwise)
        ↓
resolve_validation_refs(context, result)
  → (base_model_ref, adapter_ref, merged_model_ref)
        ↓
aiodoo_validation.api.build_<training_id>_request(...)
        ↓
aiodoo_validation.api.ValidationService.create_default().validate(request)
        ↓
ValidationOutcome (successful / certified / run_result)
```

Requires `aiodoo_validation` importable (`pip install -e ../aiodoo-validation`
or its repo root on `sys.path`); raises `ValidationIntegrationError` — never a
silent no-op — if it is not.

## Model packaging orchestration (`packaging.py`)

```text
completed TrainingResult (result.success must be True — fails closed otherwise)
        ↓
wait_for_path(adapter dir), wait_for_path(artifact.json)   # Drive FUSE sync
        ↓
aiodoo_model.publishing.PublishingService.publish(...)
  registry/storage rooted at workspace.model_registry / .model_registry_storage
        ↓
PackagingResult (artifact_id / release_id / storage_uri / already_published)
```

Idempotent: re-running `publish_adapter(...)` for an already-published
`artifact_id` (e.g. after a Colab disconnect) reports
`already_published=True` instead of raising. Requires `aiodoo_model`
importable (`pip install -e ../aiodoo-model`); raises
`PackagingIntegrationError` otherwise.

## Artifact browsing (`artifacts.py`)

Read-only discovery of everything a training run has produced on Drive
(adapter/merged/export publication state, discovered checkpoints, latest
checkpoint, resumability, log/metric file counts) — `browse_training_artifacts`
+ `summarize_artifacts` for notebook printing. Never validates checkpoint
*contents* beyond a non-empty-directory heuristic.

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
│   ├── naming.py           # single source of truth for semantic training ids
│   ├── exceptions.py
│   ├── colab_logging.py    # not named logging.py (avoids shadowing stdlib)
│   ├── version.py
│   ├── workspace.py
│   ├── drive.py
│   ├── repository.py
│   ├── models.py
│   ├── experiments.py
│   ├── trainer.py          # Phase 5 — invoke aiodoo-training only (+ Phase 8 resume)
│   ├── launcher.py         # aliases to trainer
│   ├── artifacts.py        # Phase 8 — read-only checkpoint/artifact discovery
│   ├── validation.py       # Phase 8 — invoke aiodoo-validation only
│   ├── packaging.py        # Phase 8 — invoke aiodoo-model only
│   └── training_ui.py      # Colab progress UI (ipywidgets), presentation only
├── notebooks/
│   └── 01_train.ipynb      # Phase 6/8 — thin Colab orchestration (train → validate → package)
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

## Status

Living repository posture: [`docs/STATUS.md`](docs/STATUS.md).
Historical reports: [`docs/archive/`](docs/archive/).
