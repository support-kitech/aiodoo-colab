"""launcher.py is a compatibility re-export shim over trainer.py; verify parity."""

from __future__ import annotations

import launcher
import trainer


def test_launcher_reexports_match_trainer_module() -> None:
    for name in launcher.__all__:
        assert getattr(launcher, name) is getattr(trainer, name), name


def test_launcher_all_matches_trainer_public_surface() -> None:
    assert set(launcher.__all__) <= set(trainer.__all__)
