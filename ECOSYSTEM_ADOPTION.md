# Ecosystem Adoption (Phase 8)

Scope: upgrading `aiodoo-colab` from a **training launcher** (v2.0.0 —
`AUDIT_RESOLUTION.md` Batch A/B: *"training / validation / composition logic
in Colab: Out Of Scope"*, *"Full datasets→validation→model Colab pipeline:
Future Work"*) into the canonical **orchestration environment** for
training, validation, model packaging, and experiments — closing exactly the
gap that document deferred, using only the frozen, already-complete
repositories `aiodoo-training`, `aiodoo-validation`, and `aiodoo-model`.

`aiodoo-colab` is not a training framework, a validation framework, a model
packaging framework, or a runtime. Every one of those responsibilities is
invoked, never reimplemented — see §1.

## 1. Architecture: what `aiodoo-colab` orchestrates, and what it never owns

| Owns (orchestration) | Invokes exclusively via |
| :--- | :--- |
| Experiment selection / configuration loading (`experiments.py`) | — |
| Drive workspace layout, mount, and sync (`workspace.py`, `drive.py`) | — |
| Repository management for `aiodoo-training` (`repository.py`) | — |
| Base-model cache management (`models.py`) | — |
| Training launch, resume, progress streaming (`trainer.py`, `training_ui.py`) | `aiodoo-training` (`train.py`, subprocess) |
| Evaluation / certification launch (`validation.py`) | `aiodoo-validation` (`aiodoo_validation.api`, in-process) |
| Adapter publish / resolve / materialize (`packaging.py`) | `aiodoo-model` (`aiodoo_model.publishing`/`.resolution`/`.loading`, in-process) |
| Read-only checkpoint / artifact discovery (`artifacts.py`) | — (reads paths `aiodoo-training` already wrote) |

| Never owns | Belongs exclusively to |
| :--- | :--- |
| Training loops, optimizer/scheduler setup, LoRA logic, resource planning | `aiodoo-training` |
| Behavioral/structural validation, scoring, certification, reporting | `aiodoo-validation` |
| Adapter composition, merge planning, package metadata, export/compatibility checks | `aiodoo-model` |
| Checkpoint *content* validation (RNG state, optimizer state, fingerprint match) | `aiodoo-training`'s `ResumeCoordinator` |

Two invocation styles, matching how each dependency is actually shipped
(verified by reading each, never assumed):

- **`aiodoo-training`** — application-layout repository (no
  `[build-system]`, "do not `pip install`" — see its own README); invoked as
  a **subprocess** (`python3 train.py --config …`), exactly as before this
  phase. Nothing changed here except resume support (§2).
- **`aiodoo-validation`** — also application-layout, dependency-free,
  explicitly documents a "Colab notebook" integration pattern in
  `aiodoo-validation/docs/integration.md`; imported **in-process** via
  `aiodoo_validation.api` once its repository root is on `sys.path`.
- **`aiodoo-model`** — a real installable package (has a `[build-system]`
  table), consumed the same way `aiodoo-core` already consumes it
  (`aiodoo-core/CONTRACT_ADOPTION.md` §2, `model_bridge.py`); imported
  **in-process** via `pip install -e ../aiodoo-model`.

Both in-process integrations use lazy, guarded imports
(`validation._import_validation_api`, `packaging._import_aiodoo_model`) that
raise a specific `*IntegrationError` with an actionable install hint instead
of an opaque `ImportError` traceback when the sibling repository is not
present in the runtime — fail-closed, never a silent `try/except: pass`.

## 2. Training integration (`trainer.py`) — resume added, invocation unchanged

Training was already exclusively subprocess-invoked through
`aiodoo-training/train.py` before this phase; this phase's addition is
**resume reliability**, previously entirely unimplemented:

```text
resolve_resume_checkpoint(context, requested=None)
  requested=<path>  → validate it exists and is non-empty, or raise CheckpointError
  requested=None    → artifacts.latest_checkpoint(context.checkpoints_output)
        ↓
prepare_resume_config(context, checkpoint)
  → scratch YAML copy of the training config with
    checkpointing.resume_from = str(checkpoint) injected
  → original config file on Drive is never mutated
        ↓
run_training(context, resume_from=..., auto_resume=True)
  → subprocess: python3 train.py --config <scratch resume config>
```

`aiodoo-training` reads `checkpointing.resume_from` from its config (not a
separate CLI flag — confirmed by reading its config schema); Colab's only
job is finding the right checkpoint path and placing it in the field
`aiodoo-training` already knows how to consume. `ResumeCoordinator`'s own
`load_and_validate` (RNG state, optimizer state, model fingerprint) remains
the only place resume *safety* is decided — `artifacts.is_resumable` is
explicitly documented as a lightweight, non-empty-directory heuristic only,
never a re-implementation of that check.

`workspace.Workspace.checkpoints_root(training_id)` replaces the previous,
incorrect `Workspace.checkpoints` property (which aliased `self.adapters` —
wrong path, matching neither `aiodoo-training`'s `ArtifactOutputLayout` nor
where checkpoints are actually written). This was a latent bug: any code that
had called it would have looked for checkpoints inside the published-adapter
directory.

## 3. Validation integration (`validation.py`) — new in this phase

```text
run_validation(context, result, execution_tier="standard", odoo_versions=(17,18,19))
        │
        ├─ fail closed if result.success is False
        │    (nothing to certify — never calls into aiodoo-validation)
        │
        ├─ resolve_validation_refs(context, result)
        │    → (base_model_ref, adapter_ref, merged_model_ref)
        │    derived from training's *own* context/result paths only —
        │    never independently re-derived
        │
        ├─ aiodoo_validation.api.build_<training_id>_request(...)
        │    (dynamic dispatch over _SUPPORTED_PROFILE_BUILDERS)
        │
        └─ aiodoo_validation.api.ValidationService.create_default().validate(request)
             → ValidationOutcome(successful, certified, run_result)
```

`_SUPPORTED_PROFILE_BUILDERS` documents (not hides) two intentional gaps that
are **upstream**, not something this phase invents a workaround for:

- `"repair"` is a validation profile (`list_profiles()` includes it) but
  `aiodoo_validation.api` ships no `build_repair_request` — verified by
  reading `aiodoo-validation/aiodoo_validation/api/builders.py`.
- `"context"` is not a validation profile at all in `aiodoo-validation`,
  mirroring the same documented gap already recorded in
  `aiodoo-core/CONTRACT_ADOPTION.md` §7.

Calling `run_validation` for either raises `ValidationIntegrationError`
naming the unsupported training id rather than guessing at a request shape.

`ValidationService.validate()` never raises for a validation *outcome* — a
failed or uncertified run is a legitimate result, surfaced via
`ValidationOutcome.successful`/`.certified`, not an exception. Only
environment/argument problems (missing package, unsupported profile, failed
training run) raise `ValidationIntegrationError`. `summarize_validation`
reads `run_result.run_context.errors`/`.warnings` (not top-level attributes
— `ValidationRunResult` nests them under `run_context`, confirmed by reading
`aiodoo_validation/domain/result.py`).

## 4. Model integration (`packaging.py`) — new in this phase

```text
publish_adapter(registry, experiment, training_result, version=None, channel="dev")
        │
        ├─ fail closed if training_result.success is False
        │
        ├─ drive.require_path_synced(adapter_dir)       # FUSE sync (§6)
        ├─ drive.require_path_synced(adapter_dir/artifact.json)
        │
        ├─ idempotency check: registry_store.artifact_exists(artifact_id)
        │    → already_published=True, no-op (re-running after a disconnect
        │      is safe)
        │
        └─ aiodoo_model.publishing.PublishingService(registry, storage).publish(...)
             → PackagingResult(artifact_id, release_id, storage_uri, ...)
```

`ModelRegistry.from_workspace(workspace)` roots the
`aiodoo_model.registry.FileBackedRegistry` /
`aiodoo_model.storage.StorageManager` pair at
`workspace.model_registry` / `workspace.model_registry_storage` — new Drive
paths this phase adds (§6) — so publishing, resolving, and materializing all
happen against the same Drive-persisted registry across Colab sessions.

**`release_id` design note**: `aiodoo-model`'s `Release` model is immutable
per `release_id` (confirmed empirically — reusing a family-scoped
`release_id` across versions raised `PublishDuplicateError` even for a new
version). `publish_adapter` therefore uses `artifact_id` (which already
encodes the version, e.g. `aiodoo-coding-0.1.0`) as `release_id`, and
`family_id` (`aiodoo-coding`) purely for version auto-increment queries via
`latest_release_id`.

`resolve_release` / `materialize_release` complete the loop —
`aiodoo_model.resolution.ResolutionService` / `aiodoo_model.loading.LoadingService`
resolve a published release back to its identities/locations and materialize
it into a destination directory, for consumption by, e.g., a merge/export
step outside this repository. This module never composes, merges, or
inspects package contents itself.

## 5. Experiment / notebook improvements

`notebooks/01_train.ipynb` (Phase 6) previously stopped after
`run_training(context)` — no monitor, no resume, no validation, no
packaging, matching the "training launcher" framing this phase supersedes.
It now demonstrates the full orchestration loop end to end:

1. Build a `TrainingMonitor` (progress bar, live loss/epoch, scrolling log —
   presentation-only widget code, `training_ui.py`, unchanged this phase)
   and launch `run_training(context, auto_resume=True, on_log_line=monitor.on_line)`.
2. `browse_training_artifacts` + `summarize_artifacts` — read-only Drive
   discovery of every published/checkpointed artifact for the run.
3. `run_validation` + `summarize_validation` — certification via
   `aiodoo-validation`.
4. `ModelRegistry.from_workspace` + `publish_adapter` + `summarize_packaging`
   — publish the adapter via `aiodoo-model`.

Every cell is a call into one of the modules above; no cell contains a
training loop, a scoring rule, or a package-composition step.

## 6. Google Drive integration — canonical layout, one authority

Before this phase, `constants.py` redefined `TRAINING_ID_PATTERN` as a
verbatim copy of `EXPERIMENT_ID_PATTERN` from `naming.py` (the actual
semantic-id authority) — duplicated, not delegated. It now imports the
pattern from `naming.py` directly; `naming.py` is the single source of truth
for every semantic training id (`coding`, `planner`, `context`,
`conversation`, `repair`, `execution`, `approval`, `evaluation`) and its
legacy `EXP-####` alias.

`workspace.py` gained the paths Phase 8's new orchestration needed, all
still derived from the same `AIODOO_WORKSPACE_ROOT` the training/validation/
model code already expects:

| New path | Purpose |
| :--- | :--- |
| `Workspace.merged` | `models/merged/` — merged-model outputs |
| `Workspace.model_registry` | `models/registry/` — `aiodoo-model` `FileBackedRegistry` root |
| `Workspace.model_registry_storage` | `models/registry_storage/` — `aiodoo-model` `StorageManager` root |
| `Workspace.training_cache` | `training/cache/` — parent of per-training checkpoint caches |
| `Workspace.checkpoints_root(training_id)` | `training/cache/<id>/checkpoints/` — **replaces** the previous, incorrect `Workspace.checkpoints` property (§2) |

`drive.py` gained `wait_for_path` / `require_path_synced` to address Google
Drive's FUSE mount being eventually consistent immediately after a
subprocess exits — a real, previously unhandled reliability gap (§7), not a
new "path logic" concern (it does not compute or duplicate any path; it only
polls one that another module already resolved).

No path is computed twice: every module that needs a Drive path
(`artifacts.py`, `packaging.py`, `validation.py`, `training_ui.py`) calls
into `Workspace`/`naming` rather than formatting a path string itself.

## 7. Reliability fixes

| Area | Before | After |
| :--- | :--- | :--- |
| Checkpoint path | `Workspace.checkpoints` aliased `self.adapters` — wrong location, matching neither `aiodoo-training`'s layout nor any real checkpoint writer | `Workspace.checkpoints_root(training_id)` derives the exact path `aiodoo-training`'s `ArtifactOutputLayout` writes to |
| Resume | Not implemented at all — every training launch was necessarily from scratch, even after a Colab disconnect mid-run | `resolve_resume_checkpoint` + `prepare_resume_config` + `run_training(..., auto_resume=True)` locate and inject the latest checkpoint automatically; an explicit, non-existent, or empty requested checkpoint raises `CheckpointError` instead of silently training from scratch or crashing deep inside `aiodoo-training` |
| Drive FUSE sync | A single `path.exists()` check immediately after a subprocess exit could false-negative on a path Drive had not yet synced, especially right after `aiodoo-training` finishes writing `artifact.json` | `drive.wait_for_path` / `require_path_synced` poll with a bounded timeout before treating a fresh Drive-written path as genuinely missing |
| Packaging fail-closed | N/A (packaging did not exist) | `publish_adapter` refuses a failed training run, refuses when `artifact.json` never appeared, and is idempotent on retry (`already_published=True`) rather than raising `PublishDuplicateError` on a routine notebook re-run |
| Validation fail-closed | N/A (validation did not exist) | `run_validation` refuses a failed training run before ever calling `aiodoo_validation`; unsupported training ids raise `ValidationIntegrationError` naming the gap instead of guessing a request shape |
| Missing sibling package | N/A | `_import_validation_api` / `_import_aiodoo_model` raise `ValidationIntegrationError` / `PackagingIntegrationError` with an actionable `pip install -e ../aiodoo-...` hint, instead of an opaque `ModuleNotFoundError` surfacing deep in a notebook cell |
| Silent config mutation risk | N/A (resume did not exist) | `prepare_resume_config` writes a scratch copy; the canonical training config on Drive is verified unchanged by test (`test_prepare_resume_config_injects_resume_from_without_mutating_source`) |
| Error propagation on Drive sync timeout | `wait_for_path` alone returns `False` (a caller could ignore it) | `require_path_synced` (used by `packaging.py`) raises `DriveSyncError` — a typed, fail-closed exception under `AiodooColabError`, not a boolean a caller can silently discard |

No new bare `except Exception: pass` (or equivalent silent-failure pattern)
was introduced anywhere in this phase's changes.

## 8. Duplication removed

| Removed | Was | Now |
| :--- | :--- | :--- |
| `TRAINING_ID_PATTERN` redefinition in `constants.py` | A second, hand-copied regex literal duplicating `naming.EXPERIMENT_ID_PATTERN` | Imported directly from `naming.py` — one definition |
| `Workspace.checkpoints` (incorrect alias) | `self.adapters` — wrong, and never used anywhere correctly | `Workspace.checkpoints_root(training_id)` — the one correct derivation, matching `aiodoo-training`'s layout |
| Ad hoc checkpoint-path string formatting (would have been needed by `trainer.py`/`artifacts.py` without a shared helper) | N/A — this phase added the need before it could duplicate | Both go through `Workspace.checkpoints_root` exclusively |

No training loop, optimizer/scheduler code, LoRA logic, behavioral/structural
validation, scoring, certification, adapter-composition, merge-planning, or
export/compatibility logic was found anywhere in this repository — confirmed
by a repository-wide search (`rg` for `optimizer|scheduler|lora_rank|
merge_lora|adapter_compos|certif|scoring|behavioral valid|structural valid`)
that returned only docstring references describing what belongs *elsewhere*
(e.g. `artifacts.py`/`trainer.py` explicitly documenting that RNG/optimizer
state validation is `aiodoo-training`'s `ResumeCoordinator`, never this
repository's).

## 9. Duplication intentionally retained, and why

- **`models.py` (Hugging Face base-model download/cache)** — retained; this
  is base-model *acquisition* onto Colab's local SSD, a Colab-environment
  concern with no owner in `aiodoo-training`/`aiodoo-validation`/
  `aiodoo-model` (none of which download base models — they consume a path
  Colab already resolved). Not adapter/package logic; out of scope for
  "packaging exclusively through aiodoo-model."
- **`repository.py` (git clone/update/verify of `aiodoo-training`)** —
  retained; this is repository *lifecycle* management (get the frozen
  training source onto the runtime), not training logic itself.
- **`training_ui.py` (`TrainingMonitor`, log-line parsing for progress
  display)** — retained; this is presentation-only UX (progress bar, loss/
  epoch display, scrolling log panel) explicitly assigned to Colab by the
  phase brief ("progress display, logging" are Colab's to own). It parses
  *its own* stdout stream for cosmetic display fields (`loss`, `epoch`,
  tqdm `%`) — it does not score, certify, or make any training decision from
  what it parses.
- **`artifacts.is_resumable` / `is_nonempty` (lightweight checkpoint
  heuristics)** — retained by design, not an accidental duplication of
  `aiodoo-training`'s `ResumeCoordinator.load_and_validate`. Explicitly
  documented in both files as a *decision to attempt resume*, never a
  re-implementation of resume *safety* validation (RNG/optimizer state,
  fingerprint matching) — that remains exclusively `aiodoo-training`'s.

## 10. Backward compatibility

- `run_training(context)` (no `resume_from`/`auto_resume`) behaves exactly as
  before — `auto_resume` defaults to `False`, so existing call sites (the
  pre-Phase-8 notebook cell, `main.py`) are unaffected; resume is strictly
  opt-in.
- `launcher.py`'s compatibility-alias contract is preserved and extended:
  every new `trainer.py` public symbol used by the notebook
  (`prepare_resume_config`, `resolve_resume_checkpoint`) is re-exported
  there too, so existing `from launcher import ...` call sites keep working
  and gain resume support automatically.
- `Workspace.checkpoints` is removed, not deprecated-in-place — it was never
  correct (§2/§8) and had no known caller (grep confirms zero references
  outside its own definition before this phase); no code path exercised the
  previous incorrect behavior, so nothing depends on it.
- All previously existing `Workspace` properties (`adapters`, `exports`,
  `experiments`, `datasets`, `model_cache`, …) are unchanged.
- `notebooks/01_train.ipynb`'s first eight cells (path setup → training
  launch) are extended in place, not restructured; a user who only runs
  through the original training cell sees identical behavior (`auto_resume`
  is the only behavioral addition, and is opt-in-safe — see above).

## 11. Testing

| Module | Covers |
| :--- | :--- |
| `tests/test_workspace.py` (extended) | New `Workspace` properties (`merged`, `model_registry`, `model_registry_storage`, `training_cache`), `checkpoints_root` (including legacy `EXP-####` id normalization) |
| `tests/test_artifacts.py` (new) | `discover_checkpoints`/`latest_checkpoint` sorting and filtering, `is_resumable` non-empty heuristic, `browse_training_artifacts` publication-state detection, legacy id normalization, `summarize_artifacts` |
| `tests/test_trainer.py` (extended) | `resolve_resume_checkpoint` (auto-discovery, explicit path, empty/missing rejection → `CheckpointError`), `prepare_resume_config` (scratch-file injection, source config never mutated), `run_training(..., auto_resume=True)` end-to-end config selection vs. the canonical (non-resume) path |
| `tests/test_validation.py` (new; self-skips via `pytest.importorskip` when `aiodoo_validation` is unavailable) | `resolve_validation_refs`, fail-closed rejection of a failed training run, unsupported-profile rejection, a real end-to-end `run_validation` call against `aiodoo_validation.api.ValidationService`, `summarize_validation` |
| `tests/test_packaging.py` (new; self-skips via `pytest.importorskip` when `aiodoo_model` is unavailable) | Fail-closed rejection of a failed training run and of a missing `artifact.json`, end-to-end `publish_adapter` + idempotent re-publish, version auto-increment, `resolve_release`/`materialize_release` round trip, `latest_release_id` |
| `tests/test_drive.py` (extended) | `wait_for_path` (immediate success, timeout, detects a path created mid-poll), `require_path_synced` (success, fail-closed `DriveSyncError` on timeout) |
| `tests/test_launcher.py` (new) | Compatibility-alias parity between `launcher.py` and `trainer.py`'s public surface |

Sibling-repository-dependent tests (`test_validation.py`, `test_packaging.py`)
use `pytest.importorskip` rather than mocking `aiodoo_validation`/
`aiodoo_model` internals — they exercise the *real* packages end-to-end when
available (as in this workspace), and self-skip cleanly (not fail) in an
environment where the sibling checkouts are absent, matching this
repository's "never reimplement, only orchestrate" principle even in its own
test suite.

## 12. Results

- `ruff check .` — clean (0 errors after this phase's changes).
- `mypy` (strict) — clean, 18 source files
  (`aiodoo_model.*`/`aiodoo_validation.*` set to `follow_imports = "skip"` in
  `pyproject.toml` — they are separately-owned, already-frozen repositories
  with their own mypy/CI gates and their own `aiodoo_contract` dependency
  this repository does not install; this repository only ever touches their
  public surface via `Any`-typed lazy imports, so re-typechecking their
  internals here would only surface unrelated noise, not integration bugs).
- `pytest` — **92 passed**, 0 failed, 0 skipped in this workspace (all
  sibling repos present); `test_validation.py`/`test_packaging.py` would
  self-skip (not fail) if `aiodoo-validation`/`aiodoo-model` were absent.
- Coverage (statements, `python/`): **80%** overall. Fully covered:
  `artifacts.py`, `config.py`, `constants.py`, `exceptions.py`, `launcher.py`,
  `version.py` (100% each). New integration modules: `validation.py` 96%,
  `packaging.py` 91%, `workspace.py` 96%. Lower-coverage files are
  pre-existing and orthogonal to this phase's scope: `training_ui.py` (36% —
  interactive ipywidgets rendering, not exercisable headlessly),
  `repository.py` (65% — git subprocess error branches), `colab_logging.py`
  (56% — logging handler configuration branches).

## 13. Deferred / out of scope for this phase

- **End-to-end certification** — explicitly out of scope per this phase's
  instructions ("Do NOT begin end-to-end certification"). This phase wires
  the *call path* to `aiodoo-validation`/`aiodoo-model`; it does not run or
  assert against a real training→validate→publish pipeline outcome.
- **`aiodoo-validation` builder gap for `"repair"`** — `list_profiles()`
  includes it but `aiodoo_validation.api` has no `build_repair_request`
  (§3). A pre-existing upstream gap; `run_validation("repair", ...)` raises
  `ValidationIntegrationError` naming it rather than inventing a request
  shape. Fixing it requires modifying `aiodoo-validation`, out of scope
  (frozen).
- **Datasets orchestration** — `aiodoo-datasets` is read by path
  (`Workspace.datasets`) but this repository does not clone, update, or
  invoke it; unchanged from before this phase and not called out as
  in-scope by the Phase 8 brief (training/validation/model packaging were).
- **Merge/export orchestration beyond `materialize_release`** —
  `packaging.materialize_release` resolves and loads a published release to
  a destination directory; anything downstream of that (e.g. driving
  `aiodoo-model`'s own merge-plan APIs, if any, or export-format selection)
  was not exercised because the Phase 8 brief scopes "package adapters
  exclusively through aiodoo-model," not a merge/export workflow beyond
  publish/resolve/materialize.
- Any change to `aiodoo-contract`, `aiodoo-datasets`, `aiodoo-training`,
  `aiodoo-validation`, `aiodoo-model`, `aiodoo-core`, `aiodoo-vscode` — out of
  scope per this phase's instructions; every integration point above was
  built by reading each dependency's real public API, never by modifying it.

## 14. Compliance verdict

**Compliant.** `aiodoo-colab` no longer owns training algorithms, validation
algorithms, model packaging/merge/certification logic, resource planning, or
checkpoint *management* implementations (only checkpoint *discovery* and a
resume-attempt decision, both explicitly non-authoritative — see §9). It owns
exactly experiment orchestration, configuration, progress reporting,
logging, Google Drive integration, artifact browsing, resume workflow, and
notebook UX, invoking `aiodoo-training` (subprocess), `aiodoo-validation`
(in-process), and `aiodoo-model` (in-process) for everything else. `ruff`,
`mypy`, and `pytest` all pass.
