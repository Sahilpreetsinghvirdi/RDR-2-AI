"""Phase 11 tests: guided control-verification harness."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import AppConfig, RemedyConfig, load_config
from src.control_test import (
    ControlStep,
    build_steps,
    execute_step,
    main,
    print_plan,
    run_steps,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = str(PROJECT_ROOT / "config.yaml")


class FakeController:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.calls: list[tuple] = []

    def hold(self, key: str, duration_s: float) -> bool:
        self.calls.append(("hold", key, round(duration_s, 3)))
        return self.ok

    def mouse_move(self, dx: int, dy: int) -> bool:
        self.calls.append(("move", int(dx), int(dy)))
        return self.ok

    def mouse_click(self, button: str, hold_ms: int) -> bool:
        self.calls.append(("click", button, int(hold_ms)))
        return self.ok


class TestBuildSteps:
    def test_covers_all_control_keys(self) -> None:
        cfg = AppConfig()
        steps = build_steps(cfg)
        hold_targets = {s.target for s in steps if s.kind == "hold"}
        assert set(cfg.control.keys.values()) <= hold_targets

    def test_every_step_has_expect_and_reference(self) -> None:
        steps = build_steps(AppConfig())
        assert len(steps) >= 11
        for step in steps:
            assert step.expect.strip()
            assert step.reference.strip()
            assert step.kind in {"hold", "move", "click"}
            assert step.name.strip()

    def test_order_starts_with_movement_and_ends_with_mouse(self) -> None:
        steps = build_steps(AppConfig())
        assert steps[0].name == "walk forward"
        assert steps[-2].name.startswith("aim")
        assert steps[-1].name.startswith("fire")

    def test_fire_step_is_caution_aim_is_not(self) -> None:
        steps = build_steps(AppConfig())
        aim = next(s for s in steps if s.name.startswith("aim"))
        fire = next(s for s in steps if s.name.startswith("fire"))
        assert aim.caution is False
        assert fire.caution is True

    def test_reference_entries_match_docs_list(self) -> None:
        cfg = AppConfig()
        steps = build_steps(cfg)
        forward = next(s for s in steps if s.name == "walk forward")
        assert "W" in forward.reference
        sprint = next(s for s in steps if s.name == "sprint")
        assert "Shift" in sprint.reference

    def test_unknown_key_falls_back_to_docs_pointer(self) -> None:
        cfg = AppConfig()
        cfg.control.keys["forward"] = "k"
        steps = build_steps(cfg)
        forward = next(s for s in steps if s.name == "walk forward")
        assert "docs/CONTROLS.md" in forward.reference

    def test_survival_remedy_keys_included_once(self) -> None:
        cfg = AppConfig()
        cfg.survival.remedies["eat"] = RemedyConfig(keys=["i", "i", "e"])
        steps = build_steps(cfg)
        quick = [s for s in steps if s.name.startswith("quick-use")]
        names = [s.name for s in quick]
        assert names == ["quick-use 'i'"]  # "i" once, "e" = interact key skipped

    def test_no_duplicate_targets_within_a_kind(self) -> None:
        steps = build_steps(AppConfig())
        keys = [s.target for s in steps if s.kind == "hold"]
        assert len(keys) == len(set(keys))


class TestExecuteStep:
    def test_hold_routes_to_controller_hold(self) -> None:
        ctrl = FakeController()
        step = ControlStep("walk", "hold", "w", 0.6, "e", "r")
        assert execute_step(step, ctrl, 100) is True
        assert ctrl.calls == [("hold", "w", 0.6)]

    def test_move_direction_signs(self) -> None:
        ctrl = FakeController()
        left = ControlStep("l", "move", "left", 0.0, "e", "r")
        right = ControlStep("r", "move", "right", 0.0, "e", "r")
        execute_step(left, ctrl, 120)
        execute_step(right, ctrl, 120)
        assert ctrl.calls == [("move", -120, 0), ("move", 120, 0)]

    def test_click_uses_button_and_ms(self) -> None:
        ctrl = FakeController()
        step = ControlStep("fire", "click", "left", 0.2, "e", "r")
        execute_step(step, ctrl, 100)
        assert ctrl.calls == [("click", "left", 200)]

    def test_unknown_kind_raises(self) -> None:
        step = ControlStep("bad", "zap", "x", 0.0, "e", "r")
        with pytest.raises(ValueError, match="unknown step kind"):
            execute_step(step, FakeController(), 100)


class TestRunSteps:
    def _steps(self) -> list[ControlStep]:
        return [
            ControlStep("walk forward", "hold", "w", 0.6,
                        "walks", "Forward movement = W"),
            ControlStep("fire", "click", "left", 0.2, "fires",
                        "Fire = LMB", caution=True),
            ControlStep("jump", "hold", "space", 0.15, "jumps",
                        "Jump = Space-bar"),
        ]

    def test_enter_sends_and_yes_passes(self) -> None:
        answers = iter(["", "y", "", "", ""])
        sent: list[str] = []

        def fake_input(prompt: str) -> str:
            return next(answers)

        passed, failed, skipped, failures = run_steps(
            self._steps(),
            execute=lambda s: sent.append(s.name) or True,
            input_fn=fake_input,
            out=lambda _line: None,
            sleep=lambda _s: None,
        )
        assert (passed, failed, skipped) == (2, 0, 1)  # caution skipped on enter
        assert sent == ["walk forward", "jump"]
        assert failures == []

    def test_no_marks_failure_and_q_quits(self) -> None:
        answers = iter(["", "n", "q"])

        def fake_input(prompt: str) -> str:
            return next(answers)

        passed, failed, skipped, failures = run_steps(
            self._steps(),
            execute=lambda _s: True,
            input_fn=fake_input,
            out=lambda _line: None,
            sleep=lambda _s: None,
        )
        assert (passed, failed, skipped) == (0, 1, 0)
        assert failures == ["walk forward"]

    def test_refused_input_counts_as_failure(self) -> None:
        answers = iter([""])

        def fake_input(prompt: str) -> str:
            return next(answers)

        passed, failed, skipped, failures = run_steps(
            self._steps()[:1],
            execute=lambda _s: False,
            input_fn=fake_input,
            out=lambda _line: None,
            sleep=lambda _s: None,
        )
        assert failed == 1
        assert failures == ["walk forward"]

    def test_explicit_y_runs_caution_step(self) -> None:
        answers = iter(["y", "y"])
        sent: list[str] = []

        def fake_input(prompt: str) -> str:
            return next(answers)

        passed, failed, _skipped, _failures = run_steps(
            self._steps()[1:2],
            execute=lambda s: sent.append(s.name) or True,
            input_fn=fake_input,
            out=lambda _line: None,
            sleep=lambda _s: None,
        )
        assert sent == ["fire"]
        assert (passed, failed) == (1, 0)


class TestCli:
    def test_dry_run_prints_plan_and_sends_nothing(self, capsys) -> None:
        rc = main(["--config", CONFIG_PATH, "--dry-run"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "control test plan" in out
        assert "walk forward" in out
        assert "reference:" in out
        assert "nothing was sent" in out

    def test_print_plan_shape(self) -> None:
        lines: list[str] = []
        print_plan(build_steps(AppConfig()), lines.append)
        assert any("CAUTION" in line for line in lines)
        assert any("agent sends:" in line for line in lines)

    def test_missing_config_exits_2(self, tmp_path: Path) -> None:
        assert main(["--config", str(tmp_path / "no.yaml"), "--dry-run"]) == 2

    def test_project_config_builds_valid_plan(self) -> None:
        cfg = load_config(CONFIG_PATH)
        steps = build_steps(cfg)
        assert steps
        assert all(s.kind in {"hold", "move", "click"} for s in steps)
