# aiodoo-colab — RELEASE_REPORT (v2.0.0)

**Release identity:** annotated tag `v2.0.0` (training-orchestration / tooling freeze)  
**Date:** 2026-07-19

---

## Production Ready

| Question | Answer |
| --- | --- |
| Training orchestration (Drive + pin `aiodoo-training@v2.0.0` + launch `train.py`)? | **YES** |
| Full multi-repo Colab pipeline (datasets → validation → model)? | **NO** (Future Work) |
| Training / validation / composition logic inside Colab? | **NO** (Out Of Scope) |
| Production score (orchestration in-boundary) | **8 / 10** |
| Multi-repo Colab product pipeline score | **2 / 10** |

---

## Quality gates (local)

| Gate | Result |
| --- | --- |
| `ruff check python tests main.py` | Pass |
| `ruff format --check python tests main.py` | Pass |
| `pytest` | **62 passed** |
| Default training ref | **`v2.0.0`** |

CI (`.github/workflows/ci.yml`): ruff + pytest. Mypy intentionally soft
(ipywidgets); not expanded in Batch B.

---

## What ships (orchestration)

- Google Drive mount / workspace verification
- Clone/update `aiodoo-training` at annotated tag `v2.0.0`
- Model cache path management (local SSD); experiment config plumbing
- Subprocess launch of `aiodoo-training/train.py`
- Thin Colab notebook `notebooks/01_train.ipynb`
- Minimal GitHub Actions CI

## Explicitly not in v2.0.0

- Colab clones/pipelines for `aiodoo-datasets`, `aiodoo-validation`, or
  `aiodoo-model` composition/publish
- Any training domain logic (PEFT, trainer loops, export) — stays in
  `aiodoo-training`
- Strict mypy/coverage CI gates

---

## Architecture impact

None. Training-orchestration boundaries unchanged from Batch A.

---

## Remaining blockers

None for training-orchestration tooling freeze.

---

## Remaining future work

- Optional multi-repo Colab orchestration (datasets / validation / model)
- Stricter typing gate if ipywidgets stubs improve

---

## Architectural debt

- Soft mypy in CI (ipywidgets noise)
- README historically used `aiodoo-models` / `aiodoo-lab` names (Batch B aligned)

---

## Repository health

**Strong** for orchestration freeze. Docs honest that Colab does not run the
full ecosystem pipeline.

---

## Release recommendation

**Ship annotated tag `v2.0.0`** as training-orchestration freeze. Do **not**
market as “Colab trains/serves all eight capabilities end-to-end.”
