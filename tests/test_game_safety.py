"""Regression guard: the RDR2 installation and settings must stay untouched.

The agent is only allowed to (a) read screen pixels, (b) send synthetic
input via SendInput, (c) enumerate/focus the game window and (d) write
outputs inside this project folder. This module scans the source for
registry/process/file-deletion APIs and game-folder path literals, and
verifies that config validation rejects output paths inside the RDR2
installation or the Rockstar settings folders.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.config import AppConfig, ConfigError, game_dir_marker, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (PROJECT_ROOT / "src", PROJECT_ROOT / "tools")

FORBIDDEN_PATTERNS: dict[str, str] = {
    "registry access (winreg)": r"\bwinreg\b",
    "process spawning (subprocess)": r"\bsubprocess\b",
    "shell execution (os.system)": r"\bos\.system\s*\(",
    "shell execution (os.popen)": r"\bos\.popen\s*\(",
    "tree deletion (shutil.rmtree)": r"\brmtree\s*\(",
    "file deletion (os.remove)": r"\bos\.remove\s*\(",
    "Rockstar folder used as a path": r"rockstar\s+games\s*[\\/]",
    "RDR2 folder used as a path": r"red dead redemption 2\s*[\\/]",
    "absolute drive path": r"[A-Za-z]:\\",
}


def python_sources() -> list[Path]:
    files: set[Path] = set()
    for root in SOURCE_ROOTS:
        files.update(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    return sorted(files)


@pytest.mark.parametrize(
    "label, pattern", sorted(FORBIDDEN_PATTERNS.items()), ids=lambda v: str(v)
)
def test_source_contains_no_forbidden_apis(label: str, pattern: str) -> None:
    offenders: list[str] = []
    for path in python_sources():
        text = path.read_text(encoding="utf-8")
        if re.search(pattern, text, re.IGNORECASE):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert not offenders, f"{label} found in: {offenders}"


class TestGameDirMarker:
    def test_detects_rockstar_settings_and_install(self) -> None:
        assert game_dir_marker(r"C:\Users\me\Documents\Rockstar Games\RDR2") == \
            "rockstar games"
        assert game_dir_marker(
            r"D:\Steam\steamapps\common\Red Dead Redemption 2"
        ) == "red dead redemption 2"
        assert game_dir_marker("Rockstar Games/Red Dead/logs") == "rockstar games"

    def test_allows_normal_paths(self) -> None:
        assert game_dir_marker(PROJECT_ROOT / "logs") is None
        assert game_dir_marker(r"D:\captures\agent") is None
        assert game_dir_marker("/tmp/recordings") is None

    def test_is_case_insensitive(self) -> None:
        assert game_dir_marker("c:/rockstar games/x") is not None


class TestConfigRejectsGamePaths:
    @pytest.mark.parametrize("which", ["telemetry", "recording", "ckpt", "buffer"])
    def test_absolute_game_path_rejected(self, tmp_path: Path, which: str) -> None:
        game = tmp_path / "Rockstar Games" / "Red Dead Redemption 2"
        game.mkdir(parents=True)
        cfg = AppConfig()
        target = str(game / "sub")
        if which == "telemetry":
            cfg.telemetry.dir = target
        elif which == "recording":
            cfg.recording.dir = target
        elif which == "ckpt":
            cfg.rl.checkpoint = target
        else:
            cfg.rl.buffer = target
        with pytest.raises(ConfigError, match="must not point inside"):
            cfg.validate()

    def test_relative_game_path_rejected_by_load(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "telemetry:\n  dir: Rockstar Games/Red Dead/logs\n", encoding="utf-8"
        )
        with pytest.raises(ConfigError, match="must not point inside"):
            load_config(str(cfg_file))

    def test_default_output_paths_stay_in_project(self) -> None:
        cfg = load_config(PROJECT_ROOT / "config.yaml")
        for value in (cfg.telemetry.dir, cfg.recording.dir,
                      cfg.rl.checkpoint, cfg.rl.buffer):
            resolved = cfg.resolve(value)
            resolved.relative_to(PROJECT_ROOT)
