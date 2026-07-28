"""Adapter packaging orchestration via aiodoo-model (no packaging logic here).

Publishing, fingerprinting, metadata normalization, resolution, and
materialization are performed exclusively by ``aiodoo_model.publishing``,
``aiodoo_model.resolution``, and ``aiodoo_model.loading`` — this module only
decides *which* aiodoo-training output directory to publish, *what*
identifiers to use, and *where* the registry/storage roots live on Drive
(see ``workspace.Workspace.model_registry`` /
``workspace.Workspace.model_registry_storage``). It never composes adapters,
plans merges, builds package metadata, generates artifacts, or evaluates
export/compatibility logic itself.

``aiodoo_model`` is a real installable package (unlike aiodoo-training /
aiodoo-validation's application-layout style); it is imported directly
assuming it is installed in the Colab runtime (e.g.
``pip install -e ../aiodoo-model``), exactly as aiodoo-core consumes it (see
``aiodoo-core/CONTRACT_ADOPTION.md`` Section 2 / ``model_bridge.py``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from drive import require_path_synced
from exceptions import DriveSyncError, PackagingIntegrationError
from experiments import Experiment
from naming import adapter_product_id, normalize_training_id
from trainer import TrainingResult
from workspace import Workspace

if TYPE_CHECKING:
    from aiodoo_model.domain import ResolvedModel
    from aiodoo_model.loading import MaterializationResult

logger = logging.getLogger("aiodoo_colab")

DEFAULT_CHANNEL: str = "dev"
ARTIFACT_METADATA_FILENAME: str = "artifact.json"


def _import_aiodoo_model() -> Any:
    try:
        import aiodoo_model  # noqa: PLC0415, F401
        from aiodoo_model import domain, exceptions, publishing, registry, resolution, storage
    except ImportError as exc:
        raise PackagingIntegrationError(
            "aiodoo_model is not importable. Install it alongside aiodoo-colab, "
            "e.g. `pip install -e ../aiodoo-model`, before publishing adapters."
        ) from exc
    return domain, exceptions, publishing, registry, resolution, storage


@dataclass(frozen=True, slots=True)
class ModelRegistry:
    """Registry + storage roots for aiodoo-model, rooted under the Drive workspace."""

    registry_root: Path
    storage_root: Path

    @classmethod
    def from_workspace(cls, workspace: Workspace) -> ModelRegistry:
        return cls(
            registry_root=workspace.model_registry,
            storage_root=workspace.model_registry_storage,
        )

    def ensure(self) -> None:
        """Create the registry/storage roots if missing (never touches contents)."""
        self.registry_root.mkdir(parents=True, exist_ok=True)
        self.storage_root.mkdir(parents=True, exist_ok=True)

    def open(self) -> tuple[Any, Any]:
        """Return ``(FileBackedRegistry, StorageManager)`` bound to this registry's roots."""
        _domain, _exceptions, _publishing, registry_mod, _resolution, storage_mod = (
            _import_aiodoo_model()
        )
        self.ensure()
        registry_store = registry_mod.FileBackedRegistry(self.registry_root)
        storage_manager = storage_mod.StorageManager.create_default(self.storage_root)
        return registry_store, storage_manager


@dataclass(frozen=True, slots=True)
class PackagingResult:
    """Execution metadata for one adapter publish (no packaging internals)."""

    training_id: str
    artifact_id: str
    release_id: str
    family_id: str
    version: str
    channel: str
    storage_uri: str | None
    already_published: bool


def _next_patch_version(registry_store: Any, registry_mod: Any, domain: Any, family_id: str) -> Any:
    """Auto-increment the patch version for ``family_id`` (0.1.0 if none exist)."""
    query = registry_mod.ReleaseQuery(family_id=family_id)
    existing = registry_store.list_releases(query)
    if not existing:
        return domain.Version.parse("0.1.0")
    latest = max(
        (release.version for release in existing), key=lambda v: (v.major, v.minor, v.patch)
    )
    return domain.Version(major=latest.major, minor=latest.minor, patch=latest.patch + 1)


def publish_adapter(
    registry: ModelRegistry,
    experiment: Experiment,
    training_result: TrainingResult,
    *,
    version: str | None = None,
    channel: str = DEFAULT_CHANNEL,
    drive_sync_timeout: float = 30.0,
) -> PackagingResult:
    """
    Publish a completed training run's adapter into the aiodoo-model registry.

    Fails closed:
    - refuses to package a failed training run (nothing to certify/ship);
    - waits (bounded) for the adapter directory to become visible on Drive
      before treating it as missing — Drive's FUSE mount is eventually
      consistent immediately after a subprocess exits;
    - refuses when aiodoo-training never finished writing ``artifact.json``
      (the required Publishing ingest shape — see
      ``aiodoo-model/docs/publishing.md``).

    Idempotent: re-publishing an already-registered ``artifact_id`` (e.g. a
    notebook cell re-run after a Colab disconnect) is reported via
    ``already_published=True`` instead of raising, matching this
    repository's resume-friendly design elsewhere (``repository.py``,
    ``models.py``).
    """
    if not training_result.success:
        raise PackagingIntegrationError(
            f"Refusing to package a failed training run for "
            f"{experiment.experiment_id!r} (exit_code={training_result.exit_code}): "
            f"{training_result.message}"
        )

    source = training_result.adapter_path
    try:
        require_path_synced(
            source,
            timeout=drive_sync_timeout,
            description="Adapter output directory",
        )
    except DriveSyncError as exc:
        raise PackagingIntegrationError(str(exc)) from exc

    artifact_metadata = source / ARTIFACT_METADATA_FILENAME
    try:
        require_path_synced(
            artifact_metadata,
            timeout=drive_sync_timeout,
            description=(
                f"Adapter output missing {ARTIFACT_METADATA_FILENAME} "
                "(aiodoo-training did not finish publishing the Artifact Contract package)"
            ),
        )
    except DriveSyncError as exc:
        raise PackagingIntegrationError(str(exc)) from exc

    domain, exceptions_mod, publishing, registry_mod, _resolution, _storage = _import_aiodoo_model()

    training_id = normalize_training_id(experiment.experiment_id)
    family_id = adapter_product_id(training_id)
    registry_store, storage_manager = registry.open()

    resolved_version = (
        domain.Version.parse(version)
        if version
        else _next_patch_version(registry_store, registry_mod, domain, family_id)
    )
    # One Release per published version (aiodoo-model releases are immutable
    # per release_id — a family_id groups every version's release together;
    # see ``latest_release_id`` / ``docs/registry.md``'s Release model).
    artifact_id = f"{family_id}-{resolved_version}"
    release_id = artifact_id
    channel_value = domain.ReleaseChannel(channel)

    if registry_store.artifact_exists(artifact_id):
        logger.info("Adapter already published: %s (idempotent no-op)", artifact_id)
        release = (
            registry_store.get_release(release_id)
            if registry_store.release_exists(release_id)
            else None
        )
        return PackagingResult(
            training_id=training_id,
            artifact_id=artifact_id,
            release_id=release_id,
            family_id=family_id,
            version=str(resolved_version),
            channel=str(release.channel) if release else channel,
            storage_uri=None,
            already_published=True,
        )

    request = publishing.PublishingRequest(
        source_path=str(source),
        artifact_id=artifact_id,
        family_id=family_id,
        release_id=release_id,
        version=resolved_version,
        channel=channel_value,
    )
    try:
        result = publishing.PublishingService(registry_store, storage_manager).publish(request)
    except exceptions_mod.PublishDuplicateError:
        logger.info("Adapter publish raced to duplicate: %s (idempotent no-op)", artifact_id)
        return PackagingResult(
            training_id=training_id,
            artifact_id=artifact_id,
            release_id=release_id,
            family_id=family_id,
            version=str(resolved_version),
            channel=channel,
            storage_uri=None,
            already_published=True,
        )
    except exceptions_mod.AiodooModelError as exc:
        raise PackagingIntegrationError(f"aiodoo-model publish failed: {exc}") from exc

    logger.info(
        "Adapter published training_id=%s artifact_id=%s storage_uri=%s",
        training_id,
        result.artifact.artifact_id,
        result.storage_uri,
    )
    return PackagingResult(
        training_id=training_id,
        artifact_id=result.artifact.artifact_id,
        release_id=release_id,
        family_id=family_id,
        version=str(resolved_version),
        channel=channel,
        storage_uri=result.storage_uri,
        already_published=False,
    )


def latest_release_id(registry: ModelRegistry, family_id: str) -> str | None:
    """Return the highest-version release id published for ``family_id``, if any."""
    _domain, _exceptions, _publishing, registry_mod, _resolution, _storage = _import_aiodoo_model()
    registry_store, _storage_manager = registry.open()
    existing = registry_store.list_releases(registry_mod.ReleaseQuery(family_id=family_id))
    if not existing:
        return None

    def _version_key(release: Any) -> tuple[int, int, int]:
        version = release.version
        return (version.major, version.minor, version.patch)

    latest = max(existing, key=_version_key)
    return cast(str, latest.release_id)


def resolve_release(registry: ModelRegistry, release_id: str) -> ResolvedModel:
    """Resolve one published release (specific version) to its identities/locations."""
    domain, _exceptions, _publishing, _registry_mod, resolution, _storage = _import_aiodoo_model()
    registry_store, _storage_manager = registry.open()
    reference = domain.ModelReference.for_release(release_id)
    result = resolution.ResolutionService(registry_store).resolve(
        resolution.ResolveRequest(reference=reference)
    )
    return cast("ResolvedModel", result.resolved)


def materialize_release(
    registry: ModelRegistry,
    release_id: str,
    *,
    destination_root: Path,
) -> MaterializationResult:
    """Resolve and materialize a published release into ``destination_root``."""
    _domain, exceptions_mod, _publishing, _registry_mod, _resolution, _storage = (
        _import_aiodoo_model()
    )
    _registry_store, storage_manager = registry.open()
    resolved = resolve_release(registry, release_id)
    try:
        from aiodoo_model.loading import LoadingService  # noqa: PLC0415

        return LoadingService(storage_manager).load(resolved, destination_root=destination_root)
    except exceptions_mod.AiodooModelError as exc:
        raise PackagingIntegrationError(f"aiodoo-model load failed: {exc}") from exc


def summarize_packaging(result: PackagingResult) -> dict[str, object]:
    """Plain dict summary suitable for notebook printing."""
    return {
        "training_id": result.training_id,
        "artifact_id": result.artifact_id,
        "release_id": result.release_id,
        "version": result.version,
        "channel": result.channel,
        "storage_uri": result.storage_uri,
        "already_published": result.already_published,
    }


__all__ = [
    "DEFAULT_CHANNEL",
    "ModelRegistry",
    "PackagingResult",
    "latest_release_id",
    "materialize_release",
    "publish_adapter",
    "resolve_release",
    "summarize_packaging",
]
