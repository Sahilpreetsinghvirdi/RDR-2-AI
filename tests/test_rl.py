"""Phase 9 tests: RL scaffold (featurizer, policy, buffer, trainer, planner)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.ai.rl.actions import ACTIONS, action_index, execute, n_actions
from src.ai.rl.buffer import TransitionBuffer
from src.ai.rl.features import FEATURE_NAMES, feature_dim, featurize
from src.ai.rl.planner import PolicyPlanner, rl_summary
from src.ai.rl.policy import Policy
from src.ai.rl.reward import reward_delta, threat_value
from src.ai.rl.train import main as train_main
from src.ai.rl.train import train_bc, train_reinforce
from src.config import AppConfig, ConfigError, load_config
from src.state.agent_status import AgentStatus, StatusTracker
from src.state.game_state import GameState
from src.ui.debug_overlay import build_lines

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def idx(name: str) -> int:
    return FEATURE_NAMES.index(name)


def make_rl_cfg(**overrides):
    cfg = AppConfig()
    cfg.rl.enabled = True
    for key, value in overrides.items():
        setattr(cfg.rl, key, value)
    return cfg


class FakeLocomotion:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.stopped = 0

    @property
    def active(self) -> bool:
        return False

    def tick(self) -> None:
        self.calls.append(("tick",))

    def move(self, direction: str, duration_s: float, *, sprint: bool = False) -> bool:
        self.calls.append(("move", direction, round(duration_s, 3), sprint))
        return True

    def turn(self, degrees: float) -> bool:
        self.calls.append(("turn", round(degrees, 1)))
        return True

    def tap(self, action: str) -> bool:
        self.calls.append(("tap", action))
        return True

    def stop(self) -> None:
        self.stopped += 1


def make_planner(tmp_path: Path, **overrides) -> tuple[PolicyPlanner, FakeLocomotion]:
    cfg = make_rl_cfg(**overrides)
    loco = FakeLocomotion()
    planner = PolicyPlanner(
        cfg.rl, loco,
        checkpoint_path=tmp_path / "policy.npz",
        buffer_path=tmp_path / "buffer.npz",
    )
    return planner, loco


def acted(loco: FakeLocomotion) -> list[tuple]:
    return [c for c in loco.calls if c[0] in ("move", "turn")]


class TestFeatures:
    def test_dim_matches_names(self) -> None:
        assert feature_dim() == len(FEATURE_NAMES)
        assert featurize(GameState()).shape == (len(FEATURE_NAMES),)

    def test_default_state_is_threat_none_only(self) -> None:
        vec = featurize(GameState())
        assert vec.dtype == np.float32
        assert vec[idx("threat_none")] == 1.0
        assert float(vec.sum()) == 1.0

    def test_known_values_with_presence(self) -> None:
        state = GameState()
        state.player.health = 0.7
        state.threat.enemies_detected = 5
        state.threat.wanted_level = 3
        state.threat.incoming_fire = True
        vec = featurize(state)
        assert vec[idx("health")] == pytest.approx(0.7)
        assert vec[idx("health_present")] == 1.0
        assert vec[idx("stamina_present")] == 0.0
        assert vec[idx("enemies")] == pytest.approx(0.5)
        assert vec[idx("enemies_present")] == 1.0
        assert vec[idx("wanted")] == pytest.approx(0.6)
        assert vec[idx("incoming_fire")] == 1.0

    def test_unknown_cores_stay_zero(self) -> None:
        vec = featurize(GameState())
        assert vec[idx("health")] == 0.0
        assert vec[idx("health_present")] == 0.0
        assert vec[idx("enemies")] == 0.0
        assert vec[idx("enemies_present")] == 0.0

    def test_core_values_clamped(self) -> None:
        state = GameState()
        state.player.health = 1.8
        state.player.stamina = -0.5
        vec = featurize(state)
        assert vec[idx("health")] == 1.0
        assert vec[idx("stamina")] == 0.0

    def test_threat_one_hot(self) -> None:
        state = GameState()
        state.threat.level = "under_fire"
        vec = featurize(state)
        assert vec[idx("threat_under_fire")] == 1.0
        assert vec[idx("threat_none")] == 0.0

    def test_unknown_threat_level_is_all_zero(self) -> None:
        state = GameState()
        state.threat.level = "bogus"
        vec = featurize(state)
        for key in ("threat_none", "threat_enemy", "threat_wanted",
                    "threat_under_fire"):
            assert vec[idx(key)] == 0.0

    def test_encounter_time_prompt_and_env(self) -> None:
        state = GameState()
        state.dialogue.active = True
        state.dialogue.encounter = "mount"
        state.mission.prompt_text = "PRESS E TO MOUNT"
        state.environment.time_of_day = "dusk"
        state.environment.road_detected = True
        state.confidence.navigation = 0.4
        vec = featurize(state)
        assert vec[idx("enc_mount")] == 1.0
        assert vec[idx("enc_greet")] == 0.0
        assert vec[idx("dialogue_active")] == 1.0
        assert vec[idx("prompt_present")] == 1.0
        assert vec[idx("time_dusk")] == 1.0
        assert vec[idx("env_road")] == 1.0
        assert vec[idx("env_present")] == 1.0

    def test_env_present_zero_without_navigation_confidence(self) -> None:
        state = GameState()
        state.environment.road_detected = True
        vec = featurize(state)
        assert vec[idx("env_road")] == 1.0
        assert vec[idx("env_present")] == 0.0

    def test_mounted_presence(self) -> None:
        vec = featurize(GameState())
        assert vec[idx("mounted")] == 0.0
        assert vec[idx("mounted_present")] == 0.0
        state = GameState()
        state.player.mounted = True
        vec = featurize(state)
        assert vec[idx("mounted")] == 1.0
        assert vec[idx("mounted_present")] == 1.0

    def test_deterministic(self) -> None:
        state = GameState()
        state.player.health = 0.42
        assert np.array_equal(featurize(state), featurize(state))


class TestActions:
    def test_space_is_unique(self) -> None:
        assert len(set(ACTIONS)) == len(ACTIONS)
        assert n_actions() == len(ACTIONS)

    def test_action_index(self) -> None:
        assert action_index("noop") == 0
        with pytest.raises(KeyError):
            action_index("nope")

    def test_execute_routes_to_locomotion(self) -> None:
        loco = FakeLocomotion()
        assert execute("noop", loco, step_s=0.5, turn_step_deg=15.0) is True
        assert acted(loco) == []
        assert execute("forward", loco, step_s=0.5, turn_step_deg=15.0)
        assert execute("sprint", loco, step_s=0.5, turn_step_deg=15.0)
        assert execute("turn_left", loco, step_s=0.5, turn_step_deg=15.0)
        assert execute("turn_right", loco, step_s=0.5, turn_step_deg=15.0)
        moves = [c for c in loco.calls if c[0] == "move"]
        turns = [c for c in loco.calls if c[0] == "turn"]
        assert ("move", "forward", 0.5, False) in moves
        assert ("move", "forward", 0.5, True) in moves
        assert turns == [("turn", -15.0), ("turn", 15.0)]

    def test_execute_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            execute("fly", FakeLocomotion(), step_s=0.5, turn_step_deg=15.0)


class TestPolicy:
    def test_forward_shapes_and_probs(self) -> None:
        policy = Policy(6, 4, n_actions(), rng=np.random.default_rng(1))
        obs = np.zeros(6)
        assert policy.logits(obs).shape == (n_actions(),)
        probs = policy.probs(obs)
        assert probs.shape == (n_actions(),)
        assert float(probs.sum()) == pytest.approx(1.0)

    def test_act_returns_valid_index(self) -> None:
        policy = Policy(6, 4, n_actions(), rng=np.random.default_rng(1))
        index, prob = policy.act(np.zeros(6), rng=np.random.default_rng(2))
        assert 0 <= index < n_actions()
        assert 0.0 < prob <= 1.0

    def test_bad_temperature_raises(self) -> None:
        policy = Policy(6, 4, n_actions(), rng=np.random.default_rng(1))
        with pytest.raises(ValueError):
            policy.act(np.zeros(6), temperature=0.0)

    def test_bad_dims_raise(self) -> None:
        with pytest.raises(ValueError):
            Policy(0, 4, n_actions())
        with pytest.raises(ValueError):
            Policy(4, 4, 0)

    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        policy = Policy(6, 4, n_actions(), rng=np.random.default_rng(3))
        path = policy.save(tmp_path / "p.npz")
        loaded = Policy.load(path)
        obs = np.linspace(-1, 1, 6)
        assert np.allclose(loaded.logits(obs), policy.logits(obs))
        assert (loaded.dim_in, loaded.dim_hidden, loaded.dim_out) == (6, 4, n_actions())

    def test_load_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            Policy.load(tmp_path / "nope.npz")

    def test_load_corrupt_checkpoint(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.npz"
        np.savez(bad, w1=np.zeros((2, 2)))
        with pytest.raises(ValueError):
            Policy.load(bad)

    def test_load_version_mismatch(self, tmp_path: Path) -> None:
        path = tmp_path / "v.npz"
        np.savez(
            path, version=99,
            w1=np.zeros((2, 4)), b1=np.zeros(2),
            w2=np.zeros((3, 2)), b2=np.zeros(3),
        )
        with pytest.raises(ValueError, match="version"):
            Policy.load(path)


class TestBuffer:
    def test_add_and_lengths(self) -> None:
        buf = TransitionBuffer()
        assert len(buf) == 0
        assert buf.feature_dim is None
        buf.add(np.zeros(4), 1, 0.5)
        buf.add(np.ones(4), 2, -0.25)
        assert len(buf) == 2
        assert buf.feature_dim == 4

    def test_shape_mismatch_raises(self) -> None:
        buf = TransitionBuffer()
        buf.add(np.zeros(4), 0, 0.0)
        with pytest.raises(ValueError):
            buf.add(np.zeros(5), 0, 0.0)

    def test_negative_action_raises(self) -> None:
        with pytest.raises(ValueError):
            TransitionBuffer().add(np.zeros(4), -1, 0.0)

    def test_first_row_forced_boundary(self) -> None:
        buf = TransitionBuffer()
        buf.add(np.zeros(4), 0, 0.0, boundary=False)
        assert bool(buf.boundary_flags()[0]) is True

    def test_returns_without_boundaries_accumulate(self) -> None:
        buf = TransitionBuffer()
        for reward in (1.0, 2.0, 3.0):
            buf.add(np.zeros(4), 0, reward)
        returns = buf.discounted_returns(1.0)
        assert returns.tolist() == [6.0, 5.0, 3.0]

    def test_returns_reset_at_boundaries(self) -> None:
        buf = TransitionBuffer()
        buf.add(np.zeros(4), 0, 5.0, boundary=True)
        buf.add(np.zeros(4), 1, 1.0, boundary=True)
        buf.add(np.zeros(4), 2, 2.0, boundary=False)
        returns = buf.discounted_returns(0.5)
        assert returns.tolist() == [5.0, 2.0, 2.0]

    def test_bad_gamma_raises(self) -> None:
        buf = TransitionBuffer()
        buf.add(np.zeros(4), 0, 1.0)
        with pytest.raises(ValueError):
            buf.discounted_returns(0.0)

    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        buf = TransitionBuffer()
        buf.add(np.arange(4, dtype=np.float32), 1, 0.5)
        buf.add(np.zeros(4, dtype=np.float32), 3, -1.0, boundary=True)
        path = buf.save(tmp_path / "b.npz")
        loaded = TransitionBuffer.load(path)
        assert len(loaded) == 2
        assert loaded.boundary_flags().tolist() == [True, True]
        obs, actions, rewards = loaded.as_arrays()
        assert np.allclose(obs[0], np.arange(4))
        assert actions.tolist() == [1, 3]
        assert rewards.tolist() == pytest.approx([0.5, -1.0])

    def test_load_missing_and_corrupt(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            TransitionBuffer.load(tmp_path / "nope.npz")
        bad = tmp_path / "bad.npz"
        np.savez(bad, obs=np.zeros((3, 2)), actions=np.zeros(2, dtype=np.int64))
        with pytest.raises(ValueError):
            TransitionBuffer.load(bad)

    def test_max_rows_drops_oldest_and_promotes_boundary(self) -> None:
        buf = TransitionBuffer(max_rows=3)
        for i in range(5):
            buf.add(np.full(4, float(i)), 0, float(i),
                    boundary=(i == 0))
        assert len(buf) == 3
        _, _, rewards = buf.as_arrays()
        assert rewards.tolist() == [2.0, 3.0, 4.0]
        flags = buf.boundary_flags().tolist()
        assert flags[0] is True

    def test_max_rows_stores_oldest_within_cap(self) -> None:
        buf = TransitionBuffer(max_rows=10)
        for i in range(4):
            buf.add(np.full(4, float(i)), 1, 0.0)
        assert len(buf) == 4
        obs, _, _ = buf.as_arrays()
        assert obs[0][0] == 0.0


class TestReward:
    def test_threat_values(self) -> None:
        assert threat_value("none") == 0
        assert threat_value("enemy") == 1
        assert threat_value("wanted") == 2
        assert threat_value("under_fire") == 3
        assert threat_value("bogus") == 0

    def test_core_deltas(self) -> None:
        cfg = make_rl_cfg().rl.reward
        before, after = GameState(), GameState()
        before.player.health = 0.0
        after.player.health = 0.9
        assert reward_delta(before, after, cfg) == pytest.approx(0.9)
        assert reward_delta(after, before, cfg) == pytest.approx(-0.9)

    def test_unknown_cores_contribute_nothing(self) -> None:
        cfg = make_rl_cfg().rl.reward
        assert reward_delta(GameState(), GameState(), cfg) == 0.0

    def test_threat_penalty(self) -> None:
        cfg = make_rl_cfg().rl.reward
        cur = GameState()
        cur.threat.level = "under_fire"
        assert reward_delta(GameState(), cur, cfg) == pytest.approx(-3.0)


class TestTrainer:
    def _bc_buffer(self) -> TransitionBuffer:
        rng = np.random.default_rng(11)
        buf = TransitionBuffer()
        for _ in range(200):
            x = float(rng.normal())
            obs = np.zeros(4)
            obs[0] = x
            buf.add(obs, 1 if x > 0 else 0, 0.0, boundary=True)
        return buf

    def test_bc_loss_decreases_and_learns(self) -> None:
        buf = self._bc_buffer()
        policy = Policy(4, 8, n_actions(), rng=np.random.default_rng(12))
        losses = train_bc(policy, buf, epochs=30, lr=0.3)
        assert losses[-1] < losses[0]
        obs, actions, _ = buf.as_arrays()
        predictions = np.array(
            [int(np.argmax(policy.logits(row))) for row in obs]
        )
        accuracy = float((predictions == actions).mean())
        assert accuracy > 0.9

    def test_reinforce_prefers_rewarded_action(self) -> None:
        rng = np.random.default_rng(13)
        buf = TransitionBuffer()
        for _ in range(120):
            action = int(rng.integers(n_actions()))
            buf.add(np.zeros(4), action, 1.0 if action == 2 else 0.0,
                    boundary=True)
        policy = Policy(4, 8, n_actions(), rng=np.random.default_rng(14))
        before = float(policy.probs(np.zeros(4))[2])
        losses = train_reinforce(
            policy, buf, epochs=60, lr=0.5, gamma=0.99, entropy=0.0
        )
        after = float(policy.probs(np.zeros(4))[2])
        assert after > before
        assert after > 0.8
        assert losses[-1] < losses[0]

    def test_empty_buffer_raises(self) -> None:
        policy = Policy(4, 8, n_actions(), rng=np.random.default_rng(1))
        with pytest.raises(ValueError):
            train_bc(policy, TransitionBuffer(), epochs=1, lr=0.1)
        with pytest.raises(ValueError):
            train_reinforce(policy, TransitionBuffer(), epochs=1, lr=0.1,
                            gamma=0.99, entropy=0.0)

    def test_dim_mismatch_raises(self) -> None:
        buf = TransitionBuffer()
        buf.add(np.zeros(5), 0, 1.0, boundary=True)
        policy = Policy(4, 8, n_actions(), rng=np.random.default_rng(1))
        with pytest.raises(ValueError):
            train_bc(policy, buf, epochs=1, lr=0.1)

    def test_cli_end_to_end(self, tmp_path: Path, capsys) -> None:
        buf = TransitionBuffer()
        rng = np.random.default_rng(15)
        for _ in range(60):
            action = int(rng.integers(n_actions()))
            buf.add(np.zeros(4), action, 1.0 if action == 3 else 0.0,
                    boundary=True)
        data = buf.save(tmp_path / "data.npz")
        out = tmp_path / "policy.npz"
        rc = train_main([
            "--data", str(data), "--out", str(out),
            "--epochs", "5", "--hidden", "8",
        ])
        assert rc == 0
        assert out.exists()
        assert "saved policy" in capsys.readouterr().out
        rc = train_main([
            "--data", str(data), "--out", str(out),
            "--method", "bc", "--epochs", "5", "--hidden", "8",
        ])
        assert rc == 0

    def test_cli_resume(self, tmp_path: Path) -> None:
        buf = TransitionBuffer()
        buf.add(np.zeros(4), 1, 1.0, boundary=True)
        data = buf.save(tmp_path / "data.npz")
        out = tmp_path / "policy.npz"
        assert train_main(["--data", str(data), "--out", str(out),
                           "--epochs", "2"]) == 0
        assert train_main(["--data", str(data), "--out", str(out),
                           "--epochs", "2", "--resume"]) == 0

    def test_cli_errors(self, tmp_path: Path) -> None:
        assert train_main([
            "--data", str(tmp_path / "missing.npz"),
            "--out", str(tmp_path / "out.npz"),
        ]) == 2
        assert train_main([
            "--data", "x", "--out", "y", "--lr", "0.0",
        ]) == 2
        empty = TransitionBuffer().save(tmp_path / "empty.npz")
        assert train_main(["--data", str(empty),
                           "--out", str(tmp_path / "o.npz")]) == 2


class TestPolicyPlanner:
    def test_policy_mode_inactive_without_checkpoint(self, tmp_path: Path) -> None:
        planner, loco = make_planner(tmp_path)
        status = StatusTracker(9)
        assert planner.active is False
        assert "no policy checkpoint" in planner.reason
        assert planner.step(100.0, status, None, GameState()) is False
        assert acted(loco) == []
        snap = status.snapshot()
        assert snap.rl_mode == ""
        assert "no policy checkpoint" in snap.rl_note

    def test_inactive_on_dim_mismatch(self, tmp_path: Path) -> None:
        wrong = Policy(feature_dim() + 1, 4, n_actions(),
                       rng=np.random.default_rng(1))
        wrong.save(tmp_path / "policy.npz")
        planner, loco = make_planner(tmp_path)
        assert planner.active is False
        assert "dims" in planner.reason
        planner.step(100.0, StatusTracker(9), None, GameState())
        assert acted(loco) == []

    def test_inactive_on_corrupt_checkpoint(self, tmp_path: Path) -> None:
        np.savez(tmp_path / "policy.npz", w1=np.zeros((2, 2)))
        planner, _ = make_planner(tmp_path)
        assert planner.active is False
        assert "bad policy checkpoint" in planner.reason

    def test_random_mode_acting_and_status(self, tmp_path: Path) -> None:
        planner, loco = make_planner(tmp_path, mode="random")
        status = StatusTracker(9)
        assert planner.active is True
        assert planner.step(100.0, status, None, GameState()) is True
        assert planner.steps == 1
        snap = status.snapshot()
        assert snap.rl_mode == "random"
        assert snap.rl_action in ACTIONS
        assert snap.goal == "RL POLICY"
        assert snap.action.startswith("rl:")

    def test_action_interval_gates(self, tmp_path: Path) -> None:
        planner, loco = make_planner(tmp_path, mode="random",
                                     action_interval_s=0.5)
        planner._choose = lambda obs: (1, 1.0)  # deterministic "forward"
        status = StatusTracker(9)
        planner.step(100.0, status, None, GameState())
        first = len(acted(loco))
        assert first == 1
        assert planner.step(100.1, status, None, GameState()) is True
        assert len(acted(loco)) == 1
        planner.step(100.6, status, None, GameState())
        assert len(acted(loco)) == 2

    def test_records_transitions_with_rewards(self, tmp_path: Path) -> None:
        planner, loco = make_planner(tmp_path, mode="random",
                                     action_interval_s=0.5)
        status = StatusTracker(9)
        gs = GameState()
        gs.player.health = 0.3
        planner.step(100.0, status, None, gs)
        assert planner.buffer_len == 0
        gs.player.health = 0.8
        planner.step(100.5, status, None, gs)
        assert planner.buffer_len == 1
        planner.step(101.0, status, None, gs)
        assert planner.buffer_len == 2
        _, _, rewards = planner._buffer.as_arrays()
        assert float(rewards[0]) == pytest.approx(0.5)

    def test_records_false_disables_buffering(self, tmp_path: Path) -> None:
        planner, _ = make_planner(tmp_path, mode="random", record=False)
        status = StatusTracker(9)
        for i in range(4):
            planner.step(100.0 + i, status, None, GameState())
        assert planner.buffer_len == 0

    def test_boundary_gap_starts_new_episode(self, tmp_path: Path) -> None:
        planner, _ = make_planner(tmp_path, mode="random",
                                  action_interval_s=0.5, boundary_gap_s=3.0)
        status = StatusTracker(9)
        planner.step(100.0, status, None, GameState())
        planner.step(100.5, status, None, GameState())
        planner.step(110.0, status, None, GameState())
        planner.step(110.5, status, None, GameState())
        assert planner.buffer_len == 3
        assert planner._buffer.boundary_flags().tolist() == [True, False, True]

    def test_none_state_is_noop(self, tmp_path: Path) -> None:
        planner, loco = make_planner(tmp_path, mode="random")
        assert planner.step(100.0, StatusTracker(9), None, None) is False
        assert acted(loco) == []

    def test_close_saves_buffer_once(self, tmp_path: Path) -> None:
        planner, _ = make_planner(tmp_path, mode="random")
        status = StatusTracker(9)
        for i in range(3):
            planner.step(100.0 + i, status, None, GameState())
        saved = planner.close()
        assert saved is not None and saved.exists()
        assert len(TransitionBuffer.load(saved)) == 2
        assert planner.close() is None

    def test_close_empty_returns_none(self, tmp_path: Path) -> None:
        planner, _ = make_planner(tmp_path, mode="random")
        assert planner.close() is None

    def test_stop_clears_pending_transition(self, tmp_path: Path) -> None:
        planner, _ = make_planner(tmp_path, mode="random")
        status = StatusTracker(9)
        planner.step(100.0, status, None, GameState())
        assert planner._last is not None
        assert planner._prev_action_at == 100.0
        planner.stop()
        assert planner._last is None
        assert planner._prev is None
        assert planner._prev_action_at is None
        assert planner._next_at == 0.0
        planner.stop()  # idempotent when already cleared
        assert planner.steps == 1

    def test_buffer_max_bounds_planner_buffer(self, tmp_path: Path) -> None:
        planner, _ = make_planner(tmp_path, mode="random",
                                  action_interval_s=0.0, buffer_max=5)
        status = StatusTracker(9)
        for i in range(9):
            planner.step(100.0 + i, status, None, GameState())
        assert planner.buffer_len == 5
        _, _, rewards = planner._buffer.as_arrays()
        assert rewards.shape == (5,)

    def test_summary_helpers(self, tmp_path: Path) -> None:
        assert rl_summary(None) is None
        planner, _ = make_planner(tmp_path, mode="random")
        summary = rl_summary(planner)
        assert summary is not None
        assert summary["mode"] == "random"
        assert summary["active"] is True
        assert summary["steps"] == 0


class TestRlConfig:
    def test_defaults_disabled(self) -> None:
        cfg = AppConfig()
        assert cfg.rl.enabled is False
        assert cfg.rl.mode == "policy"
        cfg.validate()

    @pytest.mark.parametrize(
        ("mutate", "match"),
        [
            (lambda c: setattr(c.rl, "mode", "chaos"), "rl.mode"),
            (lambda c: setattr(c.rl, "explore", 1.5), "rl.explore"),
            (lambda c: setattr(c.rl, "explore", -0.1), "rl.explore"),
            (lambda c: setattr(c.rl, "temperature", 0.0), "rl.temperature"),
            (lambda c: setattr(c.rl, "hidden", 0), "rl.hidden"),
            (lambda c: setattr(c.rl, "buffer_max", 0), "rl.buffer_max"),
            (lambda c: setattr(c.rl, "buffer_max", 50), "rl.buffer_max"),
            (lambda c: setattr(c.rl, "buffer_max", "many"), "rl.buffer_max"),
            (lambda c: setattr(c.rl, "action_interval_s", 0.0),
             "rl.action_interval_s"),
            (lambda c: setattr(c.rl, "boundary_gap_s", -1.0),
             "rl.boundary_gap_s"),
            (lambda c: setattr(c.rl, "turn_step_deg", 0.0), "rl.turn_step_deg"),
            (lambda c: setattr(c.rl, "step_s", 0.0), "rl.step_s"),
            (lambda c: setattr(c.rl, "checkpoint", ""), "rl.checkpoint"),
            (lambda c: setattr(c.rl, "buffer", "  "), "rl.buffer"),
            (lambda c: setattr(c.rl.reward, "health", "lots"),
             "rl.reward.health"),
        ],
    )
    def test_invalid_values_rejected(self, mutate, match: str) -> None:
        cfg = AppConfig()
        mutate(cfg)
        with pytest.raises(ConfigError, match=match):
            cfg.validate()

    def test_conflicts_with_other_primary_drivers(self) -> None:
        cfg = AppConfig()
        cfg.rl.enabled = True
        cfg.mission.enabled = True
        with pytest.raises(ConfigError, match="rl.enabled conflicts"):
            cfg.validate()
        cfg2 = AppConfig()
        cfg2.rl.enabled = True
        cfg2.control.enabled = True
        cfg2.control.mode = "route"
        with pytest.raises(ConfigError, match="rl.enabled conflicts"):
            cfg2.validate()

    def test_project_config_loads_rl_section(self, app_config) -> None:
        assert app_config.rl.enabled is False
        assert app_config.rl.mode == "policy"
        assert app_config.rl.hidden == 16
        assert app_config.rl.buffer_max == 20000
        assert app_config.rl.reward.health == 1.0

    def test_rl_disabled_by_cli_only(self) -> None:
        cfg = load_config(PROJECT_ROOT / "config.yaml")
        assert cfg.rl.enabled is False


class TestOverlay:
    def test_rl_line_when_active(self) -> None:
        status = AgentStatus(rl_mode="policy", rl_action="forward", rl_prob=0.72,
                             rl_steps=12, rl_buffer=11)
        lines = build_lines(status)
        rl_lines = [line for line in lines if line.startswith("RL")]
        assert len(rl_lines) == 1
        assert "forward" in rl_lines[0]
        assert "steps=12" in rl_lines[0]

    def test_rl_note_line(self) -> None:
        lines = build_lines(AgentStatus(rl_note="no policy checkpoint at x"))
        assert any(line.startswith("RL") and "no policy checkpoint" in line
                   for line in lines)

    def test_no_rl_line_by_default(self) -> None:
        assert not any(line.startswith("RL") for line in build_lines(AgentStatus()))
