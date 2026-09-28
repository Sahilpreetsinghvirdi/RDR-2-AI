"""Mission task model and config parser (Phase 5).

Tasks are declared in ``config.yaml`` under ``mission.tasks`` as one-key
mappings (``turn``, ``walk``, ``wait``, ``wait_for``, ``interact``, ``key``,
``log``). Parsing happens once at startup; the runner only ever executes
validated :class:`Task` objects.
"""

from __future__ import annotations

from dataclasses import dataclass

TASK_KINDS = ("turn", "walk", "wait", "wait_for", "interact", "key", "log")


@dataclass(frozen=True)
class Task:
    """One validated mission step."""

    kind: str
    deg: float = 0.0                # turn: relative camera degrees (+ right)
    meters: float = 0.0             # walk: distance to dead-reckon
    bearing: float | None = None    # walk: absolute bearing (None = keep heading)
    seconds: float = 0.0            # wait: duration
    field: str = ""                 # wait_for: dotted GameState path
    equals: object = None           # wait_for: required value
    timeout_s: float | None = None  # wait_for: give up after this long
    key: str = ""                   # key: control.keys role to tap once
    text: str = ""                  # log: message

    @property
    def label(self) -> str:
        if self.kind == "turn":
            return f"turn:{self.deg:+.0f}deg"
        if self.kind == "walk":
            bearing = "keep" if self.bearing is None else f"{self.bearing:.0f}deg"
            return f"walk:{self.meters:.1f}m @{bearing}"
        if self.kind == "wait":
            return f"wait:{self.seconds:.1f}s"
        if self.kind == "wait_for":
            return f"wait_for:{self.field}"
        if self.kind == "interact":
            return "interact"
        if self.kind == "key":
            return f"key:{self.key}"
        return f"log:{self.text}"


def parse_task(raw: object, index: int) -> Task:
    """Convert one config entry into a Task; raises ValueError with context."""
    where = f"mission.tasks[{index}]"
    if not isinstance(raw, dict) or len(raw) != 1:
        raise ValueError(f"{where} must be a single-key mapping, got {raw!r}")
    (kind, value), = raw.items()
    if kind not in TASK_KINDS:
        raise ValueError(f"{where} unknown task {kind!r} (expected one of {TASK_KINDS})")
    value = {} if value is None else value

    if kind == "turn":
        if not isinstance(value, dict) or "deg" not in value:
            raise ValueError(f"{where}.turn requires a 'deg' number")
        deg = value["deg"]
        if not _is_number(deg):
            raise ValueError(f"{where}.turn.deg must be a number, got {deg!r}")
        return Task(kind="turn", deg=float(deg))

    if kind == "walk":
        if not isinstance(value, dict) or "meters" not in value:
            raise ValueError(f"{where}.walk requires a 'meters' number")
        meters = value["meters"]
        if not _is_number(meters) or meters <= 0:
            raise ValueError(f"{where}.walk.meters must be > 0, got {meters!r}")
        bearing = value.get("bearing")
        if bearing is not None and not _is_number(bearing):
            raise ValueError(f"{where}.walk.bearing must be a number, got {bearing!r}")
        return Task(kind="walk", meters=float(meters),
                    bearing=None if bearing is None else float(bearing))

    if kind == "wait":
        if not isinstance(value, dict) or "seconds" not in value:
            raise ValueError(f"{where}.wait requires a 'seconds' number")
        seconds = value["seconds"]
        if not _is_number(seconds) or seconds <= 0:
            raise ValueError(f"{where}.wait.seconds must be > 0, got {seconds!r}")
        return Task(kind="wait", seconds=float(seconds))

    if kind == "wait_for":
        if not isinstance(value, dict):
            raise ValueError(f"{where}.wait_for requires 'field' and 'equals'")
        if "field" not in value or "equals" not in value:
            raise ValueError(f"{where}.wait_for requires 'field' and 'equals'")
        field = value["field"]
        if not isinstance(field, str) or not field.strip():
            raise ValueError(f"{where}.wait_for.field must be a non-empty string")
        timeout = value.get("timeout_s")
        if timeout is not None and (not _is_number(timeout) or timeout <= 0):
            raise ValueError(f"{where}.wait_for.timeout_s must be > 0, got {timeout!r}")
        return Task(kind="wait_for", field=field, equals=value["equals"],
                    timeout_s=None if timeout is None else float(timeout))

    if kind == "interact":
        if not isinstance(value, dict):
            raise ValueError(f"{where}.interact takes an empty mapping")
        return Task(kind="interact")

    if kind == "key":
        if not isinstance(value, dict) or "name" not in value:
            raise ValueError(f"{where}.key requires a 'name' role from control.keys")
        name = value["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{where}.key.name must be a non-empty string")
        return Task(kind="key", key=name)

    # log
    if not isinstance(value, dict) or "text" not in value:
        raise ValueError(f"{where}.log requires a 'text' string")
    text = value["text"]
    if not isinstance(text, str):
        raise ValueError(f"{where}.log.text must be a string, got {text!r}")
    return Task(kind="log", text=text)


def parse_tasks(raw_tasks: list[object]) -> list[Task]:
    """Parse every config entry; raises ValueError naming the first bad task."""
    return [parse_task(raw, i) for i, raw in enumerate(raw_tasks)]


def state_field(game_state: object, path: str) -> object:
    """Read a dotted attribute path off GameState; None when missing."""
    value: object = game_state
    for part in path.split("."):
        if not hasattr(value, part):
            return None
        value = getattr(value, part)
    return value


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
