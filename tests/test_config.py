"""Configuration loading, merging, overrides and validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config import AppConfig, ConfigError, load_config, save_effective_config


def test_project_config_loads() -> None:
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    assert cfg.agent.loop_hz > 0
    assert cfg.capture.backend == "auto"
    assert cfg.safety.emergency_key.upper() == "F12"
    assert cfg.warnings == []


def test_defaults_used_when_optional_file_absent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg.source_path is None
    assert cfg.capture.target_fps > 0


def test_explicit_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml")


def test_unknown_keys_are_warned_not_fatal(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"capture": {"target_fps": 30, "bogus_key": 1}}))
    cfg = load_config(path)
    assert cfg.capture.target_fps == 30
    assert any("bogus_key" in w for w in cfg.warnings)


def test_overrides_applied(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"capture": {"target_fps": 30}}))
    cfg = load_config(path, overrides=["capture.target_fps=60", "debug.gui=false"])
    assert cfg.capture.target_fps == 60
    assert cfg.debug.gui is False


def test_invalid_override_value_fails_validation(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("{}")
    with pytest.raises(ConfigError):
        load_config(path, overrides=["capture.target_fps=-5"])


def test_invalid_backend_fails_validation(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"capture": {"backend": "quantum"}}))
    with pytest.raises(ConfigError):
        load_config(path)


def test_malformed_override_raises(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("{}")
    with pytest.raises(ConfigError):
        load_config(path, overrides=["no_equals_sign"])


def test_resolve_relative_paths(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("{}")
    cfg = load_config(path)
    resolved = cfg.resolve("logs")
    assert resolved == (tmp_path / "logs").resolve()
    assert cfg.resolve(Path(__file__).resolve()).is_absolute()


def test_save_effective_config(tmp_path: Path) -> None:
    cfg = load_config()
    out = save_effective_config(cfg, tmp_path / "eff.yaml")
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["agent"]["loop_hz"] == cfg.agent.loop_hz
    assert "source_path" not in data


def test_to_dict_has_no_internal_fields() -> None:
    cfg = load_config()
    data = cfg.to_dict()
    assert "source_path" not in data
    assert "warnings" not in data
    assert isinstance(data["capture"]["synthetic"], dict)


def test_bad_root_type_raises(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("- 1\n- 2\n")
    with pytest.raises(ConfigError):
        load_config(path)


def test_app_config_defaults_are_valid() -> None:
    AppConfig().validate()


def test_vision_hud_config_present(app_config) -> None:
    hud = app_config.vision.hud
    assert hud.enabled is True
    expected = {"health", "stamina", "dead_eye", "minimap", "prompt", "wanted"}
    assert expected <= set(hud.regions)
    for frac in hud.regions.values():
        assert len(frac) == 4


def test_invalid_hud_region_rejected(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    bad = {"vision": {"hud": {"regions": {"health": [0.1, 0.2, 0.3]}}}}
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ConfigError):
        load_config(path)


def test_hud_region_out_of_frame_rejected(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    bad = {"vision": {"hud": {"regions": {"health": [0.9, 0.9, 0.5, 0.5]}}}}
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ConfigError):
        load_config(path)


def test_invalid_ocr_engine_rejected(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"vision": {"ocr": {"engine": "magic"}}}))
    with pytest.raises(ConfigError):
        load_config(path)


def test_debug_show_hud_defaults() -> None:
    assert AppConfig().debug.show_hud is True
