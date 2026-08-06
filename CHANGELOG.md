# Changelog

## Unreleased

- Documentation sync: `main` is source of truth; living posture in `docs/STATUS.md`; historical reports under `docs/archive/`; cross-references updated after archive moves and Git tag metadata reset.


## [2.0.0] — 2026-07-20 (Phase 8 — ecosystem adoption)

### Added
- `python/artifacts.py` — read-only checkpoint / artifact discovery
  (`discover_checkpoints`, `latest_checkpoint`, `is_resumable`,
  `browse_training_artifacts`, `summarize_artifacts`).
- `python/validation.py` — evaluation orchestration exclusively via
  `aiodoo_validation.api` (`run_validation`, `resolve_validation_refs`,
  `summarize_validation`); fails closed on a failed training run or an
  unsupported profile.
- `python/packaging.py` — adapter packaging orchestration exclusively via
  `aiodoo_model` (`ModelRegistry`, `publish_adapter`, `latest_release_id`,
  `resolve_release`, `materialize_release`, `summarize_packaging`);
  idempotent publish, Drive-sync-aware.
- `Workspace.checkpoints_root(training_id)`, `.merged`, `.model_registry`,
  `.model_registry_storage`, `.training_cache` — canonical Drive paths
  matching `aiodoo-training`'s `ArtifactOutputLayout`.
- `trainer.resolve_resume_checkpoint` / `.prepare_resume_config`;
  `run_training(..., resume_from=..., auto_resume=...)` — resume support.
- `drive.wait_for_path` / `.require_path_synced` — bounded polling for
  Google Drive FUSE mount sync latency; fail-closed `DriveSyncError` variant.
- `DriveSyncError`, `CheckpointError`, `ValidationIntegrationError`,
  `PackagingIntegrationError` exception types.
- `notebooks/01_train.ipynb` extended: `TrainingMonitor` + resume-aware
  launch, artifact browsing, validation, and packaging cells.
- `docs/archive/ECOSYSTEM_ADOPTION.md`.
- Test suites: `test_artifacts.py`, `test_validation.py`, `test_packaging.py`,
  `test_launcher.py`; `test_workspace.py`/`test_trainer.py`/`test_drive.py`
  extended for the additions above.
- `mypy` CI step (previously optional-only); `PYTHONPATH`/`pip install -e`
  steps for sibling repos.

### Changed
- `constants.TRAINING_ID_PATTERN` now imports from `naming.py` instead of
  redefining the regex (deduplication).
- `pyproject.toml`: `mypy_path` includes `../aiodoo-model`/
  `../aiodoo-validation`; `follow_imports = "skip"` for both (they are
  separately-owned, already-frozen repositories with their own gates);
  `pytest` `pythonpath` includes `../aiodoo-validation`.

### Fixed
- `Workspace.checkpoints` incorrectly aliased `self.adapters` (wrong path,
  never matched any real checkpoint writer) — removed, replaced by
  `checkpoints_root(training_id)`.

## [2.0.0] — 2026-07-19

### Changed
- `__version__ = 2.0.0`
- Default training git ref `v2.0.0` (freeze pin)
- Minimal GitHub Actions CI

### Fixed
- Ruff E501 / UP035

### Completion residuals (Batch B)
- `docs/archive/RELEASE_REPORT.md`; audit Batch A/B structure
- README ecosystem table aligned (`aiodoo-model`; multi-repo pipeline deferred)
