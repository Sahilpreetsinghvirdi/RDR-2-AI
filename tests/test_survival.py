"""Phase 8: survival provisioning macros (core triggers, cooldowns, gating)."""

from __future__ import annotations

import pytest

from src.config import AppConfig, ConfigError, RemedyConfig, SurvivalConfig
from src.control.locomotion import LocomotionController
from src.planning.survival import SurvivalPlanner, survival_summary
from src.state.agent_status import StatusTracker
from src.state.game_state import EconomyState, GameState, SurvivalState
from src.ui.debug_overlay import build_lines


class FakeInput:
    def __init__(self) -> None:
        self.presses: list[str] = []
        self.attempts: list[str] = []
        self.fail: set[str] = set()

    def press(self, key: str, hold_ms: int | None = None) -> bool:
        self.attempts.append(key)
        if key in self.fail:
            return False
        self.presses.append(key)
        return True

    def key_down(self, key: str) -> bool:
        return True

    def key_up(self, key: str) -> bool:
        return True

    def mouse_move(self, dx: int, dy: int = 0) -> bool:
        return True


def make_survival(
    *, keys: list[str] | None = None, gap_s: float = 0.5,
    cooldown_s: float = 10.0, recover_margin: float = 0.15,
    require_clear: bool = True, needs: dict[str, float] | None = None,
) -> tuple[SurvivalPlanner, FakeInput, StatusTracker, LocomotionController]:
    cfg = SurvivalConfig(
        enabled=True,
        require_clear=require_clear,
        recover_margin=recover_margin,
        remedies={
            "eat": RemedyConfig(
                keys=keys if keys is not None else ["i"],
                gap_s=gap_s,
                cooldown_s=cooldown_s,
                needs=needs if needs is not None else {"health": 0.4},
            )
        },
    )
    fake = FakeInput()
    loco = LocomotionController(AppConfig().control, fake)  # type: ignore[arg-type]
    planner = SurvivalPlanner(cfg, fake, loco)  # type: ignore[arg-type]
    return planner, fake, StatusTracker(phase=8), loco


def low_state(health: float | None = 0.2) -> GameState:
    gs = GameState()
    gs.player.health = health
    return gs


class TestGameStateDefaults:
    def test_survival_and_economy_defaults(self) -> None:
        gs = GameState()
        assert isinstance(gs.survival, SurvivalState)
        assert gs.survival.active_remedy is None
        assert gs.survival.low_cores == []
        assert isinstance(gs.economy, EconomyState)
        assert gs.economy.cash is None


class TestSurvivalSummary:
    def test_shape(self) -> None:
        summary = survival_summary(GameState())
        assert set(summary) == {"active", "low_cores"}


class TestSurvivalPlanner:
    def test_no_low_core_does_nothing(self) -> None:
        planner, fake, status, _ = make_survival()
        gs = low_state(health=0.9)
        assert planner.step(100.0, status, None, gs) is False
        assert fake.presses == []
        assert gs.survival.low_cores == []

    def test_low_health_starts_remedy_and_taps(self) -> None:
        planner, fake, status, _ = make_survival()
        gs = low_state(health=0.2)
        assert planner.step(100.0, status, None, gs) is True
        assert gs.survival.active_remedy == "eat"
        planner.step(100.1, status, None, gs)
        assert fake.presses == ["i"]
        assert planner.completed == 1
        assert status.snapshot().goal == "SURVIVAL (phase 8)"

    def test_unknown_core_does_not_trigger(self) -> None:
        planner, fake, status, _ = make_survival()
        gs = GameState()
        gs.player.health = None
        assert planner.step(100.0, status, None, gs) is False
        assert fake.presses == []

    def test_any_of_semantics_triggers_on_single_core(self) -> None:
        planner, fake, status, _ = make_survival(
            needs={"health": 0.4, "stamina": 0.4, "dead_eye": 0.4}
        )
        gs = GameState()
        gs.player.health = 0.9
        gs.player.stamina = 0.9
        gs.player.dead_eye = 0.1
        assert planner.step(100.0, status, None, gs) is True
        assert "dead_eye" in gs.survival.low_cores
        assert "health" not in gs.survival.low_cores

    def test_sequence_respects_gap_and_completes(self) -> None:
        planner, fake, status, _ = make_survival(
            keys=["i", "e"], gap_s=0.5, cooldown_s=0.001
        )
        gs = low_state(health=0.2)
        planner.step(100.0, status, None, gs)
        planner.step(100.1, status, None, gs)     # press 1
        planner.step(100.2, status, None, gs)     # inside the gap
        assert fake.presses == ["i"]
        planner.step(100.7, status, None, gs)     # press 2 -> completes
        assert fake.presses == ["i", "e"]
        assert planner.busy is False
        assert planner.completed == 1
        assert gs.survival.active_remedy is None

    def test_refused_press_retries_same_step(self) -> None:
        planner, fake, status, _ = make_survival()
        fake.fail.add("i")
        gs = low_state(health=0.2)
        planner.step(100.0, status, None, gs)
        planner.step(100.1, status, None, gs)
        assert fake.presses == []
        assert planner.busy is True
        fake.fail.clear()
        planner.step(100.2, status, None, gs)
        assert fake.presses == ["i"]

    def test_threat_blocks_start(self) -> None:
        planner, fake, status, _ = make_survival()
        gs = low_state(health=0.2)
        gs.threat.level = "wanted"
        assert planner.step(100.0, status, None, gs) is False
        assert fake.presses == []

    def test_threat_aborts_running_sequence(self) -> None:
        planner, fake, status, _ = make_survival(keys=["i", "e"])
        gs = low_state(health=0.2)
        planner.step(100.0, status, None, gs)
        planner.step(100.1, status, None, gs)
        assert fake.presses == ["i"]
        gs.threat.level = "under_fire"
        assert planner.step(100.2, status, None, gs) is False
        assert planner.busy is False
        assert gs.survival.active_remedy is None
        planner.step(100.9, status, None, gs)
        assert fake.presses == ["i"]  # no continuation after the abort

    def test_cooldown_blocks_repeat_while_still_low(self) -> None:
        planner, fake, status, _ = make_survival(cooldown_s=10.0)
        gs = low_state(health=0.2)
        planner.step(100.0, status, None, gs)
        planner.step(100.1, status, None, gs)
        assert planner.completed == 1
        for t in (100.5, 102.0, 105.0):
            assert planner.step(t, status, None, gs) is False
        assert planner.completed == 1

    def test_hysteresis_requires_recovery_before_rearm(self) -> None:
        planner, fake, status, _ = make_survival(
            cooldown_s=1.0, recover_margin=0.15, needs={"health": 0.4}
        )
        gs = low_state(health=0.2)
        planner.step(100.0, status, None, gs)
        planner.step(100.1, status, None, gs)
        assert planner.completed == 1
        # cooldown expired but core never recovered: still not re-armed
        assert planner.step(102.0, status, None, gs) is False
        # core recovers past threshold + margin -> disarmed, then re-arms
        gs.player.health = 0.6
        assert planner.step(103.0, status, None, gs) is False
        gs.player.health = 0.2
        assert planner.step(104.0, status, None, gs) is True
        assert planner.completed == 1

    def test_low_cores_reflected_even_before_trigger(self) -> None:
        planner, fake, status, _ = make_survival(
            needs={"health": 0.4}, require_clear=True
        )
        gs = low_state(health=0.2)
        gs.threat.level = "wanted"
        planner.step(100.0, status, None, gs)
        assert gs.survival.low_cores == ["health"]
        assert planner.busy is False

    def test_abort_drops_sequence_without_completion(self) -> None:
        planner, fake, status, _ = make_survival()
        gs = low_state(health=0.2)
        planner.step(100.0, status, None, gs)
        planner.abort()
        assert planner.busy is False
        assert planner.completed == 0


class TestSurvivalConfigValidation:
    def test_default_config_valid(self) -> None:
        AppConfig().validate()

    @pytest.mark.parametrize(
        ("mutate", "match"),
        [
            (lambda c: setattr(c.survival, "enabled", True)
             or c.survival.remedies.clear(), "survival.enabled"),
            (lambda c: setattr(c.survival, "recover_margin", 1.5),
             "recover_margin"),
            (lambda c: setattr(c.survival.remedies["eat"], "keys", []),
             r"\.keys"),
            (lambda c: setattr(c.survival.remedies["eat"], "keys", ["???"]),
             "unknown key"),
            (lambda c: setattr(c.survival.remedies["eat"], "gap_s", 0.0),
             "gap_s"),
            (lambda c: setattr(c.survival.remedies["eat"], "cooldown_s", -1),
             "cooldown_s"),
            (lambda c: setattr(c.survival.remedies["eat"], "hold_ms", -5),
             "hold_ms"),
            (lambda c: setattr(c.survival.remedies["eat"], "needs", {}),
             "needs"),
            (lambda c: setattr(c.survival.remedies["eat"], "needs",
                               {"mana": 0.5}), "unknown core"),
            (lambda c: setattr(c.survival.remedies["eat"], "needs",
                               {"health": 2.0}), "within 0..1"),
        ],
    )
    def test_invalid_values_rejected(self, mutate, match: str) -> None:
        cfg = AppConfig()
        mutate(cfg)
        with pytest.raises(ConfigError, match=match):
            cfg.validate()

    def test_enabled_requires_remedies(self) -> None:
        cfg = AppConfig()
        cfg.survival.enabled = True
        cfg.survival.remedies = {}
        with pytest.raises(ConfigError, match="survival.enabled"):
            cfg.validate()

    def test_dict_remedies_are_normalized(self) -> None:
        cfg = AppConfig()
        cfg.survival.remedies = {
            "eat": {"keys": ["i"], "needs": {"stamina": 0.5}}
        }
        cfg.validate()
        assert isinstance(cfg.survival.remedies["eat"], RemedyConfig)
        assert cfg.survival.remedies["eat"].needs == {"stamina": 0.5}

    def test_unknown_remedy_setting_rejected(self) -> None:
        cfg = AppConfig()
        cfg.survival.remedies = {"eat": {"bogus": 1}}
        with pytest.raises(ConfigError, match="unknown settings"):
            cfg.validate()

    def test_null_remedy_removes_default(self) -> None:
        cfg = AppConfig()
        cfg.survival.remedies = {"eat": None}
        cfg.validate()
        assert cfg.survival.remedies == {}


class TestSurvivalOverlay:
    def test_surv_line_shows_when_active(self) -> None:
        status = StatusTracker(phase=8)
        status.update(survival_active="eat", survival_low="health")
        text = "\n".join(build_lines(status.snapshot()))
        assert "SURV" in text
        assert "eat" in text
