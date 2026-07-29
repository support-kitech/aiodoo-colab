"""Unit tests for training orchestration (mocked invocation — no ML)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from config import load_config
from exceptions import CheckpointError, ExperimentValidationError, LauncherError
from experiments import ExperimentStore
from trainer import (
    TrainingResult,
    build_training_context,
    prepare_resume_config,
    resolve_resume_checkpoint,
    run_training,
    summarize_result,
)
from workspace import Workspace, ensure_workspace_layout


@pytest.fixture
def prepared(tmp_path: Path) -> tuple[Workspace, ExperimentStore]:
    mount = tmp_path / "MyDrive"
    mount.mkdir()
    config = load_config(
        drive_mount_root=mount,
        auto_mount_drive=False,
        model_cache_root=tmp_path / "aiodoo-model-cache",
    )
    ws = Workspace.from_config(config)
    ensure_workspace_layout(ws)
    return ws, ExperimentStore(workspace=ws)


def _write_experiment(ws: Workspace, experiment_id: str = "coding") -> Path:
    exp = ws.experiments / experiment_id
    config_dir = exp / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "dataset.yaml").write_text(
        "dataset_version: v1.0.0\ndataset_root: datasets/v1.0.0\n",
        encoding="utf-8",
    )
    (config_dir / "model.yaml").write_text(
        "base_model: Qwen/Qwen3-8B\n",
        encoding="utf-8",
    )
    (config_dir / "training.yaml").write_text("epochs: 1\n", encoding="utf-8")
    (config_dir / "evaluation.yaml").write_text("metrics: []\n", encoding="utf-8")
    (config_dir / "export.yaml").write_text("merge_adapter: false\n", encoding="utf-8")
    (ws.datasets / "v1.0.0").mkdir(parents=True, exist_ok=True)
    model_dir = ws.model_cache / "Qwen__Qwen3-8B"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"x")
    return exp


def test_build_training_context_resolves_paths(prepared: tuple[Workspace, ExperimentStore]) -> None:
    ws, store = prepared
    _write_experiment(ws)
    experiment = store.load("coding")
    context = build_training_context(ws, experiment)

    assert context.model_path == ws.model_cache / "Qwen__Qwen3-8B"
    assert context.dataset_path == ws.datasets / "v1.0.0"
    assert context.adapter_output == ws.models / "adapters" / "aiodoo-coding"
    assert context.checkpoints_output == ws.training / "cache" / "coding" / "checkpoints"
    assert context.metrics_output == ws.experiments / "coding" / "metrics"
    assert context.logs_output == ws.experiments / "coding" / "logs"
    assert context.training_repository == ws.training_repository
    assert context.training_config_path.name == "training.yaml"


def test_build_training_context_dual_base_cache_id(
    prepared: tuple[Workspace, ExperimentStore],
) -> None:
    """cache_id like context-deepseek must not be validated as a training id."""
    ws, store = prepared
    _write_experiment(ws, "context")
    experiment = store.load("context")
    context = build_training_context(
        ws,
        experiment,
        adapter_id="aiodoo-context-deepseek",
        cache_id="context-deepseek",
    )
    assert context.adapter_output == ws.models / "adapters" / "aiodoo-context-deepseek"
    assert (
        context.checkpoints_output
        == ws.training / "cache" / "context-deepseek" / "checkpoints"
    )
    assert context.logs_output == ws.experiments / "context" / "logs"


def test_run_training_invokes_public_entrypoint(
    prepared: tuple[Workspace, ExperimentStore],
) -> None:
    ws, store = prepared
    _write_experiment(ws)
    # Fake public entrypoint
    (ws.training_repository).mkdir(parents=True)
    (ws.training_repository / "train.py").write_text("# fake\n", encoding="utf-8")

    experiment = store.load("coding")
    context = build_training_context(ws, experiment)

    class _FakeStdout:
        def __iter__(self):
            return iter(["INFO boot\n", "{'loss': 0.5, 'epoch': 0.1}\n"])

    class _FakeProc:
        stdout = _FakeStdout()

        def wait(self, timeout: float | None = None) -> int:
            _ = timeout
            return 0

        def kill(self) -> None:
            return None

    lines: list[str] = []
    with patch("trainer.subprocess.Popen", return_value=_FakeProc()) as popen_mock:
        result = run_training(context, on_log_line=lines.append)

    assert result.success is True
    assert result.exit_code == 0
    assert result.adapter_path == context.adapter_output
    assert lines == ["INFO boot", "{'loss': 0.5, 'epoch': 0.1}"]
    args = popen_mock.call_args.args[0]
    assert args[1] == "-u"
    assert args[2].endswith("train.py")
    assert "--config" in args
    env = popen_mock.call_args.kwargs["env"]
    assert env["AIODOO_WORKSPACE_ROOT"] == str(ws.root)
    assert env["AIODOO_COLAB_MODEL_PATH"] == str(context.model_path)
    assert env["PYTHONUNBUFFERED"] == "1"
    assert "AIODOO_COLAB_CHECKPOINTS_OUTPUT" not in env


def test_run_training_reports_failure_exit_code(
    prepared: tuple[Workspace, ExperimentStore],
) -> None:
    ws, store = prepared
    _write_experiment(ws)
    (ws.training_repository).mkdir(parents=True)
    (ws.training_repository / "train.py").write_text("# fake\n", encoding="utf-8")
    experiment = store.load("coding")
    context = build_training_context(ws, experiment)

    class _FakeStdout:
        def __iter__(self):
            return iter([])

    class _FakeProc:
        stdout = _FakeStdout()

        def wait(self, timeout: float | None = None) -> int:
            _ = timeout
            return 7

        def kill(self) -> None:
            return None

    with patch("trainer.subprocess.Popen", return_value=_FakeProc()):
        result = run_training(context)

    assert result.success is False
    assert result.exit_code == 7


def test_run_training_non_streaming_uses_subprocess_run(
    prepared: tuple[Workspace, ExperimentStore],
) -> None:
    ws, store = prepared
    _write_experiment(ws)
    (ws.training_repository).mkdir(parents=True)
    (ws.training_repository / "train.py").write_text("# fake\n", encoding="utf-8")
    experiment = store.load("coding")
    context = build_training_context(ws, experiment)

    with patch(
        "trainer.subprocess.run",
        return_value=type("R", (), {"returncode": 0})(),
    ) as run_mock:
        result = run_training(context, stream_output=False)

    assert result.success is True
    run_mock.assert_called_once()
    assert "-u" in run_mock.call_args.args[0]


def test_run_training_missing_entrypoint(prepared: tuple[Workspace, ExperimentStore]) -> None:
    ws, store = prepared
    _write_experiment(ws)
    ws.training_repository.mkdir(parents=True)
    experiment = store.load("coding")
    context = build_training_context(ws, experiment)
    result = run_training(context)
    assert result.success is False
    assert "entrypoint missing" in result.message


def test_summarize_result() -> None:
    result = TrainingResult(
        success=True,
        exit_code=0,
        adapter_path=Path("/a"),
        checkpoint_path=Path("/c"),
        logs_path=Path("/l"),
        metrics_path=Path("/m"),
        duration_seconds=1.5,
        message="ok",
    )
    summary = summarize_result(result)
    assert summary["success"] is True
    assert summary["duration_seconds"] == 1.5


def test_missing_model_id_raises(prepared: tuple[Workspace, ExperimentStore]) -> None:
    ws, store = prepared
    exp = ws.experiments / "coding"
    config_dir = exp / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "dataset.yaml").write_text("dataset_version: v1\n", encoding="utf-8")
    (config_dir / "model.yaml").write_text("tokenizer: x\n", encoding="utf-8")
    (config_dir / "training.yaml").write_text("epochs: 1\n", encoding="utf-8")
    (config_dir / "evaluation.yaml").write_text("{}\n", encoding="utf-8")
    (config_dir / "export.yaml").write_text("{}\n", encoding="utf-8")
    (ws.datasets / "v1").mkdir(parents=True)

    experiment = store.load("coding")
    with pytest.raises(LauncherError, match="no model id"):
        build_training_context(ws, experiment)


def test_invalid_experiment_rejected_on_validate(
    prepared: tuple[Workspace, ExperimentStore],
) -> None:
    ws, store = prepared
    bad = ws.experiments / "not-an-exp"
    (bad / "config").mkdir(parents=True)
    with pytest.raises(ExperimentValidationError, match="Invalid training id"):
        store.validate("not-an-exp")


def test_resolve_resume_checkpoint_returns_none_when_absent(
    prepared: tuple[Workspace, ExperimentStore],
) -> None:
    ws, store = prepared
    _write_experiment(ws)
    experiment = store.load("coding")
    context = build_training_context(ws, experiment)
    assert resolve_resume_checkpoint(context) is None


def test_resolve_resume_checkpoint_auto_discovers_latest(
    prepared: tuple[Workspace, ExperimentStore],
) -> None:
    ws, store = prepared
    _write_experiment(ws)
    experiment = store.load("coding")
    context = build_training_context(ws, experiment)

    for step in (100, 300):
        ckpt = context.checkpoints_output / f"checkpoint-{step}"
        ckpt.mkdir(parents=True)
        (ckpt / "trainer_state.json").write_text("{}", encoding="utf-8")

    resolved = resolve_resume_checkpoint(context)
    assert resolved == context.checkpoints_output / "checkpoint-300"


def test_resolve_resume_checkpoint_rejects_empty_explicit_request(
    prepared: tuple[Workspace, ExperimentStore],
    tmp_path: Path,
) -> None:
    ws, store = prepared
    _write_experiment(ws)
    experiment = store.load("coding")
    context = build_training_context(ws, experiment)

    empty = tmp_path / "checkpoint-999"
    empty.mkdir()
    with pytest.raises(CheckpointError, match="empty"):
        resolve_resume_checkpoint(context, requested=empty)


def test_resolve_resume_checkpoint_rejects_missing_explicit_request(
    prepared: tuple[Workspace, ExperimentStore],
    tmp_path: Path,
) -> None:
    ws, store = prepared
    _write_experiment(ws)
    experiment = store.load("coding")
    context = build_training_context(ws, experiment)

    with pytest.raises(CheckpointError, match="does not exist"):
        resolve_resume_checkpoint(context, requested=tmp_path / "checkpoint-999")


def test_prepare_resume_config_injects_resume_from_without_mutating_source(
    prepared: tuple[Workspace, ExperimentStore],
) -> None:
    ws, store = prepared
    _write_experiment(ws)
    experiment = store.load("coding")
    context = build_training_context(ws, experiment)
    original_text = context.training_config_path.read_text(encoding="utf-8")

    checkpoint = context.checkpoints_output / "checkpoint-100"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")

    scratch_path = prepare_resume_config(context, checkpoint)

    assert scratch_path != context.training_config_path
    assert context.training_config_path.read_text(encoding="utf-8") == original_text

    merged = yaml.safe_load(scratch_path.read_text(encoding="utf-8"))
    assert merged["checkpointing"]["resume_from"] == str(checkpoint)
    assert merged["epochs"] == 1


def test_prepare_resume_config_rewrites_relative_includes_to_absolute(
    prepared: tuple[Workspace, ExperimentStore],
) -> None:
    """
    Real production configs (``configs/training/<id>/experiment.yaml``) compose
    via a relative ``include:`` list resolved against *that file's own*
    directory (``aiodoo_training.config.system.ConfigComposer``). The scratch
    resume config lives under ``training/cache/<id>/`` instead — a bare
    relative ``include:`` copied as-is would 404 at compose time on the real
    aiodoo-training side. This guards the fix: entries become absolute paths
    pointing back at the *original* config's directory, so composition still
    finds ``dataset.yaml`` et al. regardless of where the scratch file lives.
    """
    ws, store = prepared
    exp = _write_experiment(ws)
    config_dir = exp / "config"
    # Real experiment.yaml shape: an `include:` list alongside a few
    # directly-declared fields, exactly like configs/training/<id>/experiment.yaml.
    # `_resolve_training_config_path` prefers `experiment.root/experiment.yaml`
    # (the composed root, one level above `config/`) over the flat
    # `config/training.yaml` fallback — matches the real Drive/canonical layout.
    experiment_yaml = exp / "experiment.yaml"
    experiment_yaml.write_text(
        "schema_version: '1.0'\n"
        "name: coding\n"
        "include:\n"
        "  - config/dataset.yaml\n"
        "  - config/model.yaml\n"
        "  - config/training.yaml\n"
        "  - config/evaluation.yaml\n"
        "  - config/export.yaml\n",
        encoding="utf-8",
    )
    experiment = store.load("coding")
    context = build_training_context(ws, experiment)
    assert context.training_config_path == experiment_yaml

    checkpoint = context.checkpoints_output / "checkpoint-200"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")

    scratch_path = prepare_resume_config(context, checkpoint)
    merged = yaml.safe_load(scratch_path.read_text(encoding="utf-8"))

    assert merged["checkpointing"]["resume_from"] == str(checkpoint)
    for entry in merged["include"]:
        assert Path(entry).is_absolute()
        assert Path(entry).parent == config_dir
        assert Path(entry).is_file()


def test_run_training_auto_resume_uses_latest_checkpoint(
    prepared: tuple[Workspace, ExperimentStore],
) -> None:
    ws, store = prepared
    _write_experiment(ws)
    (ws.training_repository).mkdir(parents=True)
    (ws.training_repository / "train.py").write_text("# fake\n", encoding="utf-8")
    experiment = store.load("coding")
    context = build_training_context(ws, experiment)

    checkpoint = context.checkpoints_output / "checkpoint-100"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")

    class _FakeStdout:
        def __iter__(self):
            return iter([])

    class _FakeProc:
        stdout = _FakeStdout()

        def wait(self, timeout: float | None = None) -> int:
            _ = timeout
            return 0

        def kill(self) -> None:
            return None

    with patch("trainer.subprocess.Popen", return_value=_FakeProc()) as popen_mock:
        result = run_training(context, auto_resume=True)

    assert result.success is True
    args = popen_mock.call_args.args[0]
    config_arg = Path(args[args.index("--config") + 1])
    assert config_arg.name == "resume_config.yaml"
    merged = yaml.safe_load(config_arg.read_text(encoding="utf-8"))
    assert merged["checkpointing"]["resume_from"] == str(checkpoint)


def test_run_training_without_resume_uses_canonical_config(
    prepared: tuple[Workspace, ExperimentStore],
) -> None:
    ws, store = prepared
    _write_experiment(ws)
    (ws.training_repository).mkdir(parents=True)
    (ws.training_repository / "train.py").write_text("# fake\n", encoding="utf-8")
    experiment = store.load("coding")
    context = build_training_context(ws, experiment)

    checkpoint = context.checkpoints_output / "checkpoint-100"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")

    class _FakeStdout:
        def __iter__(self):
            return iter([])

    class _FakeProc:
        stdout = _FakeStdout()

        def wait(self, timeout: float | None = None) -> int:
            _ = timeout
            return 0

        def kill(self) -> None:
            return None

    with patch("trainer.subprocess.Popen", return_value=_FakeProc()) as popen_mock:
        result = run_training(context)

    assert result.success is True
    args = popen_mock.call_args.args[0]
    config_arg = Path(args[args.index("--config") + 1])
    assert config_arg == context.training_config_path
