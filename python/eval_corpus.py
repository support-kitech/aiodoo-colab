"""Materialize Drive ``aiodoo-datasets`` eval corpora into validation package layout.

Drive layout (production)::

    AIODOO/datasets/v1.0.0/corpus/
      coding_eval_corpus.jsonl
      coding_eval_corpus_manifest.json
      …

``aiodoo-validation`` requires a directory package::

    <pkg>/
      manifest.json   # CorpusManifest (role=evaluation + fingerprint)
      records.jsonl   # capability parser compact/native records

This module converts contract-shaped ``(request, expected_response)`` gold
pairs into compact validation records and writes packages under::

    AIODOO/datasets/v1.0.0/corpus/validation_packages/<capability>/
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "MaterializedCorpus",
    "corpus_jsonl_name",
    "materialize_drive_eval_corpus",
    "resolve_corpus_root",
]


def resolve_corpus_root(dataset_path: Path) -> Path:
    """``…/datasets/v1.0.0/corpus`` next to the SFT JSONL files."""
    return Path(dataset_path) / "corpus"


def corpus_jsonl_name(training_id: str) -> str:
    return f"{training_id}_eval_corpus.jsonl"


@dataclass(frozen=True, slots=True)
class MaterializedCorpus:
    """Paths + identity for one capability's validation corpus package."""

    training_id: str
    corpus_id: str
    package_dir: Path
    source_jsonl: Path
    record_count: int
    skipped: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _guess_type(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".py"):
        return "python"
    if lower.endswith(".xml"):
        return "xml"
    if lower.endswith((".yml", ".yaml")):
        return "yaml"
    if lower.endswith(".csv"):
        return "data"
    return "data"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError(f"{path}:{line_no}: record must be a JSON object")
        rows.append(obj)
    return rows


def _edits_to_artifacts(edits: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for index, edit in enumerate(edits):
        path = str(edit.get("path") or "").strip()
        content = edit.get("content")
        if not path or not isinstance(content, str) or not content.strip():
            continue
        artifacts.append(
            {
                "id": f"edit_{index}",
                "path": path,
                "type": _guess_type(path),
                "content": content,
            }
        )
    return artifacts


def _convert_record(training_id: str, record: Mapping[str, Any], index: int) -> dict[str, Any] | None:
    """Convert one datasets contract eval record → validation compact record."""
    request = record.get("request")
    response = record.get("expected_response")
    if not isinstance(request, Mapping) or not isinstance(response, Mapping):
        return None

    record_id = str(request.get("request_id") or f"{training_id}-eval-{index}")

    if training_id == "coding":
        problem = str(request.get("instruction") or "").strip()
        edits = list(response.get("edits") or [])
        if not problem or not edits:
            return None
        # Prefer real file-content gold when present; otherwise keep contract
        # expected_response JSON so the package still loads for production runs.
        artifacts = _edits_to_artifacts(edits)
        if not artifacts:
            payload = json.dumps(
                {
                    "edits": edits,
                    "rationale": response.get("rationale"),
                },
                ensure_ascii=True,
                indent=2,
            ) + "\n"
            artifacts = [
                {
                    "id": "coding_expected",
                    "path": "coding/expected.json",
                    "type": "data",
                    "content": payload,
                }
            ]
        return {
            "record_id": record_id,
            "problem": problem,
            "artifacts": artifacts,
            "operations": [],
            "explanation": str(response.get("rationale") or ""),
        }

    if training_id == "repair":
        problem = str(request.get("failure_description") or "").strip()
        fix = response.get("fix")
        if not isinstance(fix, Mapping):
            return None
        artifacts = _edits_to_artifacts(list(fix.get("edits") or []))
        if not problem or not artifacts:
            return None
        return {
            "record_id": record_id,
            "problem": problem,
            "artifacts": artifacts,
            "operations": [],
            "explanation": str(fix.get("description") or ""),
        }

    if training_id == "planner":
        problem = str(request.get("goal") or "").strip()
        steps = response.get("steps")
        if not problem or not isinstance(steps, list) or not steps:
            return None
        payload = json.dumps(steps, ensure_ascii=True, indent=2) + "\n"
        return {
            "record_id": record_id,
            "problem": problem,
            "artifacts": [
                {
                    "id": "plan_steps",
                    "path": "plan/steps.json",
                    "type": "data",
                    "content": payload,
                }
            ],
            "operations": [],
            "explanation": "Expected planner steps.",
        }

    if training_id == "execution":
        command = str(request.get("command") or "").strip()
        if not command:
            return None
        expected = {
            "status": response.get("status"),
            "exit_code": response.get("exit_code"),
            "stdout": response.get("stdout"),
            "stderr": response.get("stderr"),
        }
        payload = json.dumps(expected, ensure_ascii=True, indent=2) + "\n"
        return {
            "record_id": record_id,
            "problem": command,
            "artifacts": [
                {
                    "id": "execution_expected",
                    "path": "execution/expected.json",
                    "type": "data",
                    "content": payload,
                }
            ],
            "operations": [],
            "explanation": "Expected execution outcome.",
        }

    if training_id == "approval":
        subject = str(request.get("subject") or "").strip()
        status = str(response.get("status") or "").strip()
        if not subject or not status:
            return None
        payload = json.dumps(
            {"status": status, "reason": response.get("reason")},
            ensure_ascii=True,
            indent=2,
        ) + "\n"
        return {
            "record_id": record_id,
            "problem": subject,
            "artifacts": [
                {
                    "id": "approval_expected",
                    "path": "approval/expected.json",
                    "type": "data",
                    "content": payload,
                }
            ],
            "operations": [],
            "explanation": str(response.get("reason") or ""),
        }

    if training_id == "conversation":
        turns = request.get("turns")
        reply = response.get("reply")
        if not isinstance(turns, list) or reply is None:
            return None
        if isinstance(reply, Mapping):
            reply_text = str(reply.get("content") or "").strip()
        else:
            reply_text = str(reply).strip()
        if not reply_text:
            return None
        problem = json.dumps(turns, ensure_ascii=True)
        return {
            "record_id": record_id,
            "problem": problem,
            "artifacts": [
                {
                    "id": "conversation_reply",
                    "path": "conversation/reply.txt",
                    "type": "data",
                    "content": reply_text if reply_text.endswith("\n") else reply_text + "\n",
                }
            ],
            "operations": [],
            "explanation": "Expected conversation reply.",
        }

    if training_id == "evaluation":
        # Evaluation SFT judgment corpora are separate; contract eval pairs vary.
        # Keep a minimal text gold when verdict-like fields exist.
        problem = json.dumps(dict(request), ensure_ascii=True, sort_keys=True)
        payload = json.dumps(dict(response), ensure_ascii=True, indent=2) + "\n"
        if len(problem) < 8:
            return None
        return {
            "record_id": record_id,
            "problem": problem,
            "artifacts": [
                {
                    "id": "evaluation_expected",
                    "path": "evaluation/expected.json",
                    "type": "data",
                    "content": payload,
                }
            ],
            "operations": [],
            "explanation": "Expected evaluation judgment.",
        }

    return None


def materialize_drive_eval_corpus(
    *,
    corpus_root: Path,
    training_id: str,
    dataset_version: str = "v1.0.0",
    force: bool = False,
) -> MaterializedCorpus:
    """
    Build ``corpus_root/validation_packages/<id>/{manifest.json,records.jsonl}``.

    Raises ``FileNotFoundError`` when the Drive JSONL is missing.
    Raises ``RuntimeError`` when no convertible records exist (e.g. coding
    eval corpus with empty edit contents — not usable for production behavior).
    """
    corpus_root = Path(corpus_root)
    source = corpus_root / corpus_jsonl_name(training_id)
    if not source.is_file():
        raise FileNotFoundError(
            f"Missing Drive eval corpus for {training_id!r}: {source}\n"
            f"Expected production layout: …/datasets/{dataset_version}/corpus/"
            f"{corpus_jsonl_name(training_id)}"
        )

    package_dir = corpus_root / "validation_packages" / training_id
    records_path = package_dir / "records.jsonl"
    manifest_path = package_dir / "manifest.json"
    corpus_id = f"production.{dataset_version}.{training_id}.eval"

    if (
        not force
        and records_path.is_file()
        and manifest_path.is_file()
        and package_dir.is_dir()
    ):
        # Reuse existing package when source fingerprint still matches.
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            source_fp = _sha256_file(source)
            if existing.get("metadata", {}).get("source_jsonl_sha256") == source_fp:
                n_records = sum(
                    1 for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()
                )
                return MaterializedCorpus(
                    training_id=training_id,
                    corpus_id=str(existing.get("corpus_id") or corpus_id),
                    package_dir=package_dir,
                    source_jsonl=source,
                    record_count=n_records,
                    skipped=0,
                )
        except (OSError, json.JSONDecodeError, TypeError):
            pass

    raw_rows = _load_jsonl(source)
    converted: list[dict[str, Any]] = []
    skipped = 0
    for index, row in enumerate(raw_rows):
        item = _convert_record(training_id, row, index)
        if item is None:
            skipped += 1
            continue
        converted.append(item)

    if not converted:
        raise RuntimeError(
            f"No usable validation records for {training_id!r} from {source}.\n"
            f"Read {len(raw_rows)} contract rows, skipped {skipped}.\n"
            "For coding: preferred gold is nonempty edits[].content. "
            "If your Drive corpus still has empty contents, regenerate "
            "coding_eval_corpus.jsonl from aiodoo-datasets (project_coding "
            "now backfills from metadata.module_path)."
        )

    package_dir.mkdir(parents=True, exist_ok=True)
    with records_path.open("w", encoding="utf-8") as handle:
        for item in converted:
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")

    fingerprint = _sha256_file(records_path)
    source_fp = _sha256_file(source)
    manifest = {
        "corpus_id": corpus_id,
        "capability_id": training_id,
        "role": "evaluation",
        "dataset_version": dataset_version,
        "fingerprint": fingerprint,
        "source_package": "aiodoo-datasets-drive",
        "denied_training_fingerprints": [],
        "metadata": {
            "kind": "production_drive_eval_corpus",
            "source_jsonl": source.name,
            "source_jsonl_sha256": source_fp,
            "converted_records": len(converted),
            "skipped_records": skipped,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return MaterializedCorpus(
        training_id=training_id,
        corpus_id=corpus_id,
        package_dir=package_dir,
        source_jsonl=source,
        record_count=len(converted),
        skipped=skipped,
    )
