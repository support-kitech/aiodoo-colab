"""Unit tests for Colab training progress UI helpers."""

from __future__ import annotations

from training_ui import parse_training_log_line


def test_parse_loss_epoch_from_hf_log_line() -> None:
    parsed = parse_training_log_line(
        "{'loss': 0.6279, 'grad_norm': 0.3, 'learning_rate': 8e-05, 'epoch': 0.03}"
    )
    assert parsed["loss"] == 0.6279
    assert parsed["epoch"] == 0.03
    assert parsed["learning_rate"] == 8e-05


def test_parse_tqdm_progress() -> None:
    parsed = parse_training_log_line(" 45%|████     | 150/335 [01:00<01:10, 2.5it/s]")
    assert parsed["percent"] == 45
    assert parsed["step"] == 150
    assert parsed["total_steps"] == 335


def test_parse_status_markers() -> None:
    assert parse_training_log_line("INFO Pipeline completed in 12s")["status"] == "completed"
    assert parse_training_log_line("ERROR Pipeline failed: boom")["status"] == "failed"


def test_parse_ignores_unrelated_lines() -> None:
    assert parse_training_log_line("Bootstrapping registries") == {}
