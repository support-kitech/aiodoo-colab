"""Colab training progress UI (ipywidgets) — presentation only, no ML logic."""

from __future__ import annotations

import ast
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# HF Trainer log lines often look like: {'loss': 0.42, 'epoch': 0.15, ...}
_METRIC_LINE_RE = re.compile(r"\{[^{}]*'loss'\s*:")
_TQDM_RE = re.compile(r"(\d+)%\|.*?\|\s*(\d+)/(\d+)")
_STEP_EPOCH_RE = re.compile(r"'epoch'\s*:\s*([0-9.]+)|\"epoch\"\s*:\s*([0-9.]+)")


def parse_training_log_line(line: str) -> dict[str, float | int | str]:
    """
    Extract loss / epoch / tqdm progress from one trainer log line.

    Returns an empty dict when the line has no recognized progress fields.
    """
    text = line.strip()
    if not text:
        return {}

    found: dict[str, float | int | str] = {}

    tqdm_match = _TQDM_RE.search(text)
    if tqdm_match:
        found["percent"] = int(tqdm_match.group(1))
        found["step"] = int(tqdm_match.group(2))
        found["total_steps"] = int(tqdm_match.group(3))

    if _METRIC_LINE_RE.search(text):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            blob = text[start : end + 1]
            try:
                payload = ast.literal_eval(blob)
            except (SyntaxError, ValueError):
                payload = None
            if isinstance(payload, dict):
                loss = payload.get("loss")
                if isinstance(loss, (int, float)):
                    found["loss"] = float(loss)
                epoch = payload.get("epoch")
                if isinstance(epoch, (int, float)):
                    found["epoch"] = float(epoch)
                learning_rate = payload.get("learning_rate")
                if isinstance(learning_rate, (int, float)):
                    found["learning_rate"] = float(learning_rate)

    if "epoch" not in found:
        epoch_match = _STEP_EPOCH_RE.search(text)
        if epoch_match:
            raw = epoch_match.group(1) or epoch_match.group(2)
            found["epoch"] = float(raw)

    if "Pipeline completed" in text or "Training entrypoint completed" in text:
        found["status"] = "completed"
    if "Training failed" in text or "Pipeline failed" in text:
        found["status"] = "failed"

    return found


@dataclass
class TrainingMonitor:
    """
    Fancy Colab UI: progress bar, live loss/epoch, scrolling log panel.

    Usage::

        monitor = TrainingMonitor(training_id=\"coding\")
        monitor.display()
        result = run_training(context, on_log_line=monitor.on_line)
        monitor.finish(result)
    """

    training_id: str = ""
    experiment_id: str = ""  # deprecated alias; prefer training_id
    model_name: str = ""
    dataset_version: str = ""
    stage: str = ""
    run_id: str = ""
    max_log_lines: int = 40
    _started: float = field(default=0.0, init=False, repr=False)
    _log_lines: list[str] = field(default_factory=list, init=False, repr=False)
    _widgets: Any = field(default=None, init=False, repr=False)
    _last_loss: float | None = field(default=None, init=False, repr=False)
    _last_epoch: float | None = field(default=None, init=False, repr=False)
    _last_step: int | None = field(default=None, init=False, repr=False)
    _total_steps: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.training_id and not self.experiment_id:
            self.experiment_id = self.training_id
        elif self.experiment_id and not self.training_id:
            self.training_id = self.experiment_id
        if not self.training_id:
            raise ValueError("TrainingMonitor requires training_id (or legacy experiment_id)")
        # Never show legacy EXP-* in the notebook UI; normalize when known.
        try:
            from naming import is_acceptable_training_ref, normalize_training_id

            if is_acceptable_training_ref(self.training_id):
                self.training_id = normalize_training_id(self.training_id)
                self.experiment_id = self.training_id
        except ImportError:  # pragma: no cover - Colab always has python/ on path
            pass
        if not self.stage:
            self.stage = self.training_id.replace("_", " ").title()

    def display(self) -> None:
        """Render widgets into the current notebook output cell."""
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError as exc:  # pragma: no cover - Colab always has these
            raise RuntimeError(
                "ipywidgets / IPython required for TrainingMonitor. "
                "Fall back to run_training(...) without the monitor."
            ) from exc

        run_line = f"Run : {self.run_id}" if self.run_id else "Run : (pending)"
        model_line = f"Model : {self.model_name}" if self.model_name else "Model : —"
        dataset_line = (
            f"Dataset : {self.dataset_version}" if self.dataset_version else "Dataset : —"
        )
        title = widgets.HTML(
            value=(
                "<pre style='margin:0 0 8px 0;font-family:ui-monospace,monospace;"
                "font-size:13px;line-height:1.45'>"
                f"Training : {self.training_id}\n"
                f"{run_line}\n"
                f"{model_line}\n"
                f"{dataset_line}\n"
                f"Stage : {self.stage}"
                "</pre>"
            )
        )
        status = widgets.HTML(value="<b>Status:</b> starting…")
        metrics = widgets.HTML(
            value="<b>Progress:</b> — &nbsp; <b>Loss:</b> — &nbsp; <b>ETA:</b> —"
        )
        bar = widgets.FloatProgress(
            value=0.0,
            min=0.0,
            max=1.0,
            description="Progress",
            bar_style="info",
            orientation="horizontal",
            layout=widgets.Layout(width="100%", height="24px"),
        )
        elapsed = widgets.HTML(value="<b>Elapsed:</b> 0s")
        log = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                height="280px",
                overflow="auto",
                border="1px solid #444",
                padding="6px",
            )
        )
        panel = widgets.VBox(
            [title, status, metrics, bar, elapsed, widgets.HTML("<b>Live log</b>"), log],
            layout=widgets.Layout(width="100%"),
        )
        self._widgets = {
            "status": status,
            "metrics": metrics,
            "bar": bar,
            "elapsed": elapsed,
            "log": log,
            "panel": panel,
        }
        self._started = time.perf_counter()
        display(panel)  # type: ignore[no-untyped-call]  # IPython.display ships no stubs
        self._set_status("running", "Training subprocess started — streaming logs…")

    def on_line(self, line: str) -> None:
        """Callback for ``run_training(..., on_log_line=monitor.on_line)``."""
        text = line.rstrip("\n")
        self._append_log(text)
        parsed = parse_training_log_line(text)
        if not parsed and self._widgets is not None:
            self._refresh_elapsed()
            return

        if "loss" in parsed:
            self._last_loss = float(parsed["loss"])
        if "epoch" in parsed:
            self._last_epoch = float(parsed["epoch"])
        if "step" in parsed:
            self._last_step = int(parsed["step"])
        if "total_steps" in parsed:
            self._total_steps = int(parsed["total_steps"])
        if "percent" in parsed and self._widgets is not None:
            self._widgets["bar"].value = float(parsed["percent"]) / 100.0
        elif (
            self._last_epoch is not None
            and self._widgets is not None
            and 0.0 <= self._last_epoch <= 1.0
        ):
            # Single-epoch runs: epoch fraction ≈ overall progress.
            self._widgets["bar"].value = min(1.0, max(0.0, self._last_epoch))

        status = parsed.get("status")
        if status == "completed":
            self._set_status("ok", "Pipeline reported completion")
            if self._widgets is not None:
                self._widgets["bar"].value = 1.0
                self._widgets["bar"].bar_style = "success"
        elif status == "failed":
            self._set_status("error", "Pipeline reported failure")
            if self._widgets is not None:
                self._widgets["bar"].bar_style = "danger"

        self._refresh_metrics()
        self._refresh_elapsed()

    def finish(self, result: Any) -> None:
        """Update UI from a ``TrainingResult`` after ``run_training`` returns."""
        success = bool(getattr(result, "success", False))
        exit_code = getattr(result, "exit_code", None)
        duration = float(getattr(result, "duration_seconds", 0.0) or 0.0)
        message = str(getattr(result, "message", "") or "")
        if success:
            self._set_status(
                "ok",
                f"Finished OK (exit={exit_code}) in {duration:.1f}s — {message}",
            )
            if self._widgets is not None:
                self._widgets["bar"].value = 1.0
                self._widgets["bar"].bar_style = "success"
        else:
            self._set_status(
                "error",
                f"Finished with failure (exit={exit_code}) in {duration:.1f}s — {message}",
            )
            if self._widgets is not None:
                self._widgets["bar"].bar_style = "danger"
        self._refresh_elapsed()

    def as_callback(self) -> Callable[[str], None]:
        return self.on_line

    def _append_log(self, text: str) -> None:
        self._log_lines.append(text)
        if len(self._log_lines) > self.max_log_lines:
            self._log_lines = self._log_lines[-self.max_log_lines :]
        if self._widgets is None:
            return
        log = self._widgets["log"]
        log.clear_output(wait=True)
        with log:
            # Keep newest lines visible; plain print is Colab-friendly.
            print("\n".join(self._log_lines))

    def _refresh_metrics(self) -> None:
        if self._widgets is None:
            return
        loss = f"{self._last_loss:.4f}" if self._last_loss is not None else "—"
        if self._last_step is not None and self._total_steps is not None:
            progress = f"{self._last_step} / {self._total_steps}"
            remaining = None
            elapsed = time.perf_counter() - self._started if self._started else 0.0
            if self._last_step > 0 and elapsed > 0:
                rate = elapsed / self._last_step
                remaining = rate * (self._total_steps - self._last_step)
        elif self._last_step is not None:
            progress = str(self._last_step)
            remaining = None
        else:
            progress = "—"
            remaining = None

        if remaining is None:
            eta = "—"
        else:
            mins, secs = divmod(int(remaining), 60)
            hours, mins = divmod(mins, 60)
            eta = f"{hours:02d}:{mins:02d}:{secs:02d}"

        self._widgets[
            "metrics"
        ].value = f"<b>Progress:</b> {progress} &nbsp; <b>Loss:</b> {loss} &nbsp; <b>ETA:</b> {eta}"

    def _refresh_elapsed(self) -> None:
        if self._widgets is None or self._started <= 0:
            return
        elapsed = time.perf_counter() - self._started
        mins, secs = divmod(int(elapsed), 60)
        hours, mins = divmod(mins, 60)
        if hours:
            label = f"{hours}h {mins}m {secs}s"
        elif mins:
            label = f"{mins}m {secs}s"
        else:
            label = f"{secs}s"
        self._widgets["elapsed"].value = f"<b>Elapsed:</b> {label}"

    def _set_status(self, kind: str, message: str) -> None:
        if self._widgets is None:
            return
        colors = {"running": "#3584e4", "ok": "#2ec27e", "error": "#e01b24"}
        color = colors.get(kind, "#888")
        self._widgets[
            "status"
        ].value = f"<b>Status:</b> <span style='color:{color}'>{message}</span>"


__all__ = ["TrainingMonitor", "parse_training_log_line"]
