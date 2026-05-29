#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

STANDALONE_TEMPLATE = r'''
# Auto-generated Orbit Wars submission. Pure Python; no local ML package imports.
import math

_MODEL = __MODEL__
_SUN_RADIUS = 10.0
_CENTER = (50.0, 50.0)
_PREV_TOTAL = {}
_COMET_OWNER_BY_ID = {}
_COMET_SHIPS_BY_ID = {}
_COMET_PENDING_BY_PLAYER = {}

def _get(obj, key, default=None):
    if hasattr(obj, "get"):
        value = obj.get(key, default)
        return default if value is None else value
    value = getattr(obj, key, default)
    return default if value is None else value

def _as_planets(obs):
    return _get(obs, "planets", []) or []

def _pid(p): return int(p[0])
def _owner(p): return int(p[1])
def _x(p): return float(p[2])
def _y(p): return float(p[3])
def _radius(p): return float(p[4])
def _ships(p): return int(p[5])
def _production(p): return float(p[6])

def _matvec(x, w, b):
    out = []
    for j in range(len(b)):
        z = float(b[j])
        for i in range(min(len(x), len(w))):
            z += float(x[i]) * float(w[i][j])
        out.append(z)
    return out

def _relu(v):
    return [z if z > 0.0 else 0.0 for z in v]

def _tanh(v):
    return [math.tanh(z) for z in v]

def _sigmoid_scalar(z):
    if z >= 0.0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)

def _clip(v, lo=0.0, hi=1.0):
    value = float(v)
    if value < lo:
        return float(lo)
    if value > hi:
        return float(hi)
    return value

def _apply_activation(values, name):
    if name == "relu":
        return _relu(values)
    if name == "tanh":
        return _tanh(values)
    if name in (None, "identity"):
        return values
    raise RuntimeError("unsupported embedded activation: %r" % (name,))

def _forward_layers(obs, layers):
    x = [float(v) for v in obs[:15]]
    for layer in layers:
        x = _matvec(x, layer["w"], layer["b"])
        x = _apply_activation(x, layer.get("activation", "identity"))
    return x

def _argmax(values):
    best_idx = 0
    best_value = float(values[0])
    for idx in range(1, len(values)):
        value = float(values[idx])
        if value > best_value:
            best_idx = idx
            best_value = value
    return best_idx

def _predict(obs):
    kind = _MODEL["kind"]
    if kind == "numpy":
        hidden = _tanh(_matvec([float(v) for v in obs[:15]], _MODEL["w1"], _MODEL["b1"]))
        raw = _matvec(hidden, _MODEL["w2"], _MODEL["b2"])
        return [_sigmoid_scalar(v) for v in raw]

    latent = _forward_layers(obs, _MODEL.get("hidden", []))
    raw = _matvec(latent, _MODEL["action"]["w"], _MODEL["action"]["b"])
    if kind == "continuous":
        lows = _MODEL.get("low", [0.0] * len(raw))
        highs = _MODEL.get("high", [1.0] * len(raw))
        return [_clip(raw[i], float(lows[i]), float(highs[i])) for i in range(len(raw))]
    if kind == "multidiscrete":
        out = []
        offset = 0
        for count in _MODEL["nvec"]:
            count = int(count)
            out.append(float(_argmax(raw[offset:offset + count])))
            offset += count
        return out
    raise RuntimeError("unsupported embedded model kind: %r" % (kind,))

def _dist_xy(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)

def _dist(a, b):
    return _dist_xy(_x(a), _y(a), _x(b), _y(b))

def _dist_center(p):
    return _dist_xy(_x(p), _y(p), _CENTER[0], _CENTER[1])

def _orbiting(p):
    return _dist_center(p) + _radius(p) < 50.0

def _quad_xy(x, y):
    if x >= 50.0 and y < 50.0:
        return 1
    if x < 50.0 and y < 50.0:
        return 2
    if x < 50.0 and y >= 50.0:
        return 3
    return 4

def _quad(p):
    return _quad_xy(_x(p), _y(p))

def _ccw(q):
    return {1: 2, 2: 3, 3: 4, 4: 1}[q]

def _ownership_encoding(p, player):
    if _owner(p) == int(player):
        return 1
    if _owner(p) == -1:
        return 0
    return -1

def _owner_priority(p, player):
    return {-1: 0, 0: 1, 1: 2}[_ownership_encoding(p, player)]

def _comet_ids(obs):
    ids = set(int(pid) for pid in (_get(obs, "comet_planet_ids", []) or []))
    for group in _get(obs, "comets", []) or []:
        if hasattr(group, "get") or hasattr(group, "planet_ids"):
            ids.update(int(pid) for pid in (_get(group, "planet_ids", []) or []))
        elif isinstance(group, (list, tuple)) and group:
            first = group[0]
            if isinstance(first, (list, tuple, set)):
                ids.update(int(pid) for pid in first)
    return ids

def _total_production(planets, player):
    return sum(_production(p) for p in planets if _owner(p) == int(player))

def _select_candidates(src, planets, player, comets):
    src_q = _quad(src)
    wanted_q = _ccw(src_q)
    source_orbiting = _orbiting(src)
    static = []
    orbiting = []
    seen = set()
    for p in planets:
        pid = _pid(p)
        if pid == _pid(src) or pid in seen or pid in comets:
            continue
        orbiting_p = _orbiting(p)
        if not orbiting_p:
            if source_orbiting:
                if _quad(p) == src_q:
                    static.append(p)
                    seen.add(pid)
            elif _dist(src, p) <= 25.0:
                static.append(p)
                seen.add(pid)
        else:
            orbit_q = _quad(p)
            if orbit_q == wanted_q or (source_orbiting and orbit_q == src_q):
                orbiting.append(p)
            seen.add(pid)
    combined = static + orbiting
    group = {}
    for p in static:
        group[_pid(p)] = 0
    for p in orbiting:
        group[_pid(p)] = 1
    combined.sort(
        key=lambda p: (
            _owner_priority(p, player),
            -_production(p),
            _ships(p),
            group[_pid(p)],
            _dist(src, p),
            _pid(p),
        )
    )
    return combined[:4]

def _segment_intersects_circle(ax, ay, bx, by, cx, cy, radius):
    if radius < 0.0:
        radius = 0.0
    abx = bx - ax
    aby = by - ay
    den = abx * abx + aby * aby
    if den == 0.0:
        return _dist_xy(ax, ay, cx, cy) <= radius
    u = ((cx - ax) * abx + (cy - ay) * aby) / den
    u = max(0.0, min(1.0, u))
    return _dist_xy(ax + u * abx, ay + u * aby, cx, cy) <= radius

def _crosses_sun(source_xy, target_xy, radius):
    return _segment_intersects_circle(
        source_xy[0], source_xy[1], target_xy[0], target_xy[1], 50.0, 50.0, radius
    )

def _predicted_planet_position(p, t, angular_velocity):
    if angular_velocity == 0.0 or not _orbiting(p):
        return (_x(p), _y(p))
    dx = _x(p) - 50.0
    dy = _y(p) - 50.0
    current_angle = math.atan2(dy, dx)
    radius = _dist_center(p)
    predicted_angle = current_angle + angular_velocity * t
    return (50.0 + radius * math.cos(predicted_angle), 50.0 + radius * math.sin(predicted_angle))

def _predict_launch(source, target, angular_velocity, fleet_speed):
    source_xy = (_x(source), _y(source))
    if (not _orbiting(target)) or angular_velocity == 0.0 or fleet_speed <= 0.0:
        target_xy = (_x(target), _y(target))
        return (math.atan2(target_xy[1] - source_xy[1], target_xy[0] - source_xy[0]), source_xy, target_xy)

    def intercept_error(t):
        tx, ty = _predicted_planet_position(target, t, angular_velocity)
        distance = _dist_xy(_x(source), _y(source), tx, ty) - _radius(source) - _radius(target)
        return distance - fleet_speed * t

    scan_dt = 0.25
    max_t = 500.0
    prev_t = 0.0
    prev_f = intercept_error(prev_t)
    t = scan_dt
    while t <= max_t:
        curr_f = intercept_error(t)
        if prev_f * curr_f <= 0.0:
            lo = prev_t
            hi = t
            for _ in range(40):
                mid = (lo + hi) * 0.5
                mid_f = intercept_error(mid)
                if prev_f * mid_f <= 0.0:
                    hi = mid
                    curr_f = mid_f
                else:
                    lo = mid
                    prev_f = mid_f
            target_xy = _predicted_planet_position(target, (lo + hi) * 0.5, angular_velocity)
            return (
                math.atan2(target_xy[1] - _y(source), target_xy[0] - _x(source)),
                source_xy,
                target_xy,
            )
        prev_t = t
        prev_f = curr_f
        t += scan_dt
    target_xy = (_x(target), _y(target))
    return (math.atan2(target_xy[1] - _y(source), target_xy[0] - _x(source)), source_xy, target_xy)

def _get_fleet_speed(num_ships):
    min_speed = 1.0
    max_speed = 6.0
    ships = int(num_ships)
    if ships <= 1:
        return min_speed
    ships = min(ships, 1000)
    speed = min_speed + (max_speed - min_speed) * (math.log(ships) / math.log(1000)) ** 1.5
    return min(max(speed, min_speed), max_speed)

def _sun_radius(obs, config=None):
    for source in (obs, config):
        if source is None:
            continue
        for key in ("sun_collision_radius", "sun_radius"):
            value = _get(source, key, None)
            if value is not None:
                return float(value)
        nested = _get(source, "configuration", None) or _get(source, "config", None)
        if nested is not None:
            for key in ("sun_collision_radius", "sun_radius"):
                value = _get(nested, key, None)
                if value is not None:
                    return float(value)
    return _SUN_RADIUS

def _filter_valid_candidates(src, candidates, angular_velocity, fleet_speed, sun_radius):
    valid = []
    launches = []
    for target in candidates:
        if len(valid) >= 4:
            break
        launch = _predict_launch(src, target, angular_velocity, fleet_speed)
        if _crosses_sun(launch[1], launch[2], sun_radius):
            continue
        valid.append(target)
        launches.append(launch)
    return valid, launches

def _build_obs(src, chosen, planets, player, previous_total):
    total = _total_production(planets, player)
    values = [float(total), float(total - previous_total), float(_ships(src))]
    for i in range(4):
        if i < len(chosen):
            p = chosen[i]
            enc = _ownership_encoding(p, player)
            ships = float(_ships(p) if enc == 1 else -_ships(p))
            values.extend([float(enc), ships, _production(p)])
        else:
            values.extend([0.0, 0.0, 0.0])
    return values

def _build_filtered_for_source(src, planets, player, previous_total, comets, angular_velocity, fleet_speed, sun_radius):
    raw = _select_candidates(src, planets, player, comets)
    chosen, launches = _filter_valid_candidates(src, raw, angular_velocity, fleet_speed, sun_radius)
    return _build_obs(src, chosen, planets, player, previous_total), chosen, launches

_SEND_FRACTIONS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

def _minimum_tactically_valid_send(target, candidate_player, remaining_ships, capture_margin=1):
    if _owner(target) == int(candidate_player):
        return 1 if remaining_ships > 0 else 0
    return max(0, _ships(target) + max(0, capture_margin))

def _decode(src, chosen, outputs, angular_velocity, fleet_speed, sun_radius):
    outputs = [float(v) for v in outputs]
    if len(outputs) == 2:
        if not chosen:
            return []
        remaining = max(0, _ships(src) - 1)
        if remaining <= 0:
            return []
        target_choice = int(outputs[0])
        if target_choice <= 0:
            return []
        idx = target_choice - 1
        if idx < 0 or idx >= min(4, len(chosen)):
            return []
        target = chosen[idx]
        min_send = _minimum_tactically_valid_send(target, _owner(src), remaining)
        if min_send <= 0:
            return []
        if _owner(target) != _owner(src) and min_send > remaining:
            return []
        amount = outputs[1] / 100.0 if outputs[1] > 1.0 else outputs[1]
        span = max(0, remaining - min_send)
        ships = int(min_send + math.floor(_clip(amount) * span))
        ships = min(max(min_send, ships), remaining)
        launch = _predict_launch(src, target, angular_velocity, _get_fleet_speed(ships))
        if _crosses_sun(launch[1], launch[2], sun_radius):
            return []
        return [[_pid(src), float(launch[0]), int(ships)]]

    output_len = len(outputs)
    if output_len == 9:
        output_len = 8
    if output_len not in (4, 8):
        return []
    remaining = max(0, _ships(src) - 1)
    actions = []
    if output_len == 4:
        weights = [_SEND_FRACTIONS[max(0, min(5, int(v)))] for v in outputs[:min(4, len(chosen))]]
        total_weight = sum(weights)
        if total_weight <= 0.0:
            return []
        requests = [int(math.floor(remaining * (w / total_weight))) if w > 0.0 else 0 for w in weights]
    else:
        requests = []
        for idx in range(min(4, len(chosen))):
            if float(outputs[idx * 2]) <= 0.5:
                requests.append(0)
                continue
            requested = int(math.floor(_ships(src) * _clip(outputs[idx * 2 + 1])))
            requests.append(max(1, requested))

    for idx, requested in enumerate(requests):
        if remaining <= 0:
            break
        ships = min(max(0, int(requested)), remaining)
        if ships <= 0:
            continue
        target = chosen[idx]
        launch = _predict_launch(src, target, angular_velocity, _get_fleet_speed(ships))
        if _crosses_sun(launch[1], launch[2], sun_radius):
            continue
        remaining -= ships
        actions.append([_pid(src), float(launch[0]), int(ships)])
    return actions

def _closest_non_comet_planet(comet, planets, comets):
    candidates = [p for p in planets if _pid(p) != _pid(comet) and _pid(p) not in comets]
    if not candidates:
        return None
    candidates.sort(key=lambda p: (_dist(comet, p), _pid(p)))
    return candidates[0]

def _forced_comet_actions(obs, player, angular_velocity, fleet_speed, sun_radius):
    planets = _as_planets(obs)
    comets = _comet_ids(obs)
    by_id = {_pid(p): p for p in planets}
    player = int(player)
    forced = []
    pending = set(_COMET_PENDING_BY_PLAYER.get(player, set()))
    _COMET_PENDING_BY_PLAYER.setdefault(player, set())
    for cid in sorted(pending):
        comet = by_id.get(cid)
        if comet and _owner(comet) == player and _ships(comet) > 1:
            target = _closest_non_comet_planet(comet, planets, comets)
            if target:
                ships = _ships(comet) - 1
                launch = _predict_launch(comet, target, angular_velocity, fleet_speed)
                if not _crosses_sun(launch[1], launch[2], sun_radius):
                    forced.append([_pid(comet), float(launch[0]), int(ships)])
        _COMET_PENDING_BY_PLAYER[player].discard(cid)

    for cid in comets:
        comet = by_id.get(cid)
        if comet is None:
            _COMET_OWNER_BY_ID.pop(cid, None)
            _COMET_SHIPS_BY_ID.pop(cid, None)
            continue
        old_owner = _COMET_OWNER_BY_ID.get(cid)
        old_ships = _COMET_SHIPS_BY_ID.get(cid)
        captured = old_owner is not None and (
            _owner(comet) != old_owner
            or (old_ships is not None and _ships(comet) > old_ships + max(1, _production(comet)))
        )
        if captured and _owner(comet) >= 0:
            _COMET_PENDING_BY_PLAYER.setdefault(_owner(comet), set()).add(cid)
        _COMET_OWNER_BY_ID[cid] = _owner(comet)
        _COMET_SHIPS_BY_ID[cid] = _ships(comet)
    return forced

def agent(obs, config=None):
    planets = _as_planets(obs)
    player = int(_get(obs, "player", 0))
    angular_velocity = float(_get(obs, "angular_velocity", 0.0))
    fleet_speed = float(_get(obs, "fleet_speed", 1.0))
    sun_radius = _sun_radius(obs, config)
    comets = _comet_ids(obs)
    total = _total_production(planets, player)
    previous_total = _PREV_TOTAL.get(player, total)
    _PREV_TOTAL[player] = total
    actions = _forced_comet_actions(obs, player, angular_velocity, fleet_speed, sun_radius)
    enforce_min_source_ships = int(_MODEL.get("output_size", 0)) == 2
    for src in planets:
        if _owner(src) != player or _pid(src) in comets:
            continue
        if enforce_min_source_ships and _ships(src) < 11:
            continue
        model_obs, chosen, _launches = _build_filtered_for_source(
            src, planets, player, previous_total, comets, angular_velocity, fleet_speed, sun_radius
        )
        actions.extend(_decode(src, chosen, _predict(model_obs), angular_velocity, fleet_speed, sun_radius))
    return actions
'''.lstrip()

FALLBACK_MODEL: dict[str, Any] = {
    "kind": "continuous",
    "hidden": [
        {
            "w": [[0.0] * 9 for _ in range(15)],
            "b": [1.0] * 8 + [0.0],
            "activation": "relu",
        }
    ],
    "action": {
        "w": [[0.0] * 9 for _ in range(9)],
        "b": [1.0, 0.2, 1.0, 0.2, 1.0, 0.2, 1.0, 0.2, 0.0],
    },
    "low": [0.0] * 9,
    "high": [1.0] * 9,
    "output_size": 9,
}

STANDALONE = STANDALONE_TEMPLATE.replace(
    "__MODEL__", json.dumps(FALLBACK_MODEL, separators=(",", ":"))
)


def _linear_to_layer(linear: Any, activation: str | None = None) -> dict[str, Any]:
    weight = linear.weight.detach().cpu().numpy().tolist()  # out x in
    bias = linear.bias.detach().cpu().numpy().tolist()
    # Standalone matvec expects in x out.
    transposed = [list(col) for col in zip(*weight)]
    layer: dict[str, Any] = {"w": transposed, "b": bias}
    if activation is not None:
        layer["activation"] = activation
    return layer


def _extract_policy_layers(policy_net: Any) -> list[dict[str, Any]]:
    import torch as th

    layers: list[dict[str, Any]] = []
    for module in policy_net:
        if isinstance(module, th.nn.Linear):
            layers.append(_linear_to_layer(module))
        elif isinstance(module, th.nn.ReLU):
            if not layers:
                raise TypeError("Unsupported policy_net: activation before linear layer")
            layers[-1]["activation"] = "relu"
        elif isinstance(module, th.nn.Tanh):
            if not layers:
                raise TypeError("Unsupported policy_net: activation before linear layer")
            layers[-1]["activation"] = "tanh"
        else:
            raise TypeError(f"Unsupported policy_net module for standalone export: {module!r}")
    return layers


def _space_bounds(value: Any, size: int, default: float) -> list[float]:
    try:
        import numpy as np

        flat = np.asarray(value, dtype=float).reshape(-1)
        if flat.size == 1:
            return [float(flat[0])] * size
        if flat.size == size:
            return [float(v) for v in flat.tolist()]
    except Exception:
        pass
    return [float(default)] * size


def _load_sb3_model(path: str | Path) -> Any:
    from stable_baselines3 import PPO

    try:
        from sb3_contrib import MaskablePPO
    except Exception:
        MaskablePPO = None  # type: ignore[assignment]

    if MaskablePPO is not None:
        try:
            return MaskablePPO.load(str(path), device="cpu")
        except Exception:
            pass
    return PPO.load(str(path), device="cpu")


def _extract_sb3_model(path: str | Path) -> dict[str, Any]:
    model = _load_sb3_model(path)
    action_space = model.action_space
    hidden = _extract_policy_layers(model.policy.mlp_extractor.policy_net)
    action = _linear_to_layer(model.policy.action_net)
    nvec = getattr(action_space, "nvec", None)
    if nvec is not None:
        nvec_list = [int(v) for v in list(nvec)]
        if nvec_list not in ([5, 101], [6, 6, 6, 6]):
            raise ValueError(
                "Unsupported MultiDiscrete action space for standalone export: "
                f"{nvec_list!r}. Supported spaces are [5, 101] and [6, 6, 6, 6]."
            )
        return {
            "kind": "multidiscrete",
            "hidden": hidden,
            "action": action,
            "nvec": nvec_list,
            "output_size": len(nvec_list),
        }

    shape = getattr(action_space, "shape", None)
    if isinstance(shape, tuple) and len(shape) == 1 and int(shape[0]) in (8, 9):
        size = int(shape[0])
        return {
            "kind": "continuous",
            "hidden": hidden,
            "action": action,
            "low": _space_bounds(getattr(action_space, "low", None), size, 0.0),
            "high": _space_bounds(getattr(action_space, "high", None), size, 1.0),
            "output_size": size,
        }

    raise ValueError(
        "Unsupported action space for standalone export: "
        f"{action_space!r}. Supported spaces are Box shape 8/9, "
        "MultiDiscrete([5, 101]), and MultiDiscrete([6, 6, 6, 6])."
    )


def _extract_numpy_model(path: str | Path) -> dict[str, Any]:
    from orbit_wars_rl.models.policy import NumpyPolicy

    policy = NumpyPolicy.load(path)
    output_size = len(policy.b2)
    if output_size not in (8, 9):
        raise ValueError(
            "Unsupported NumpyPolicy output size for standalone export: "
            f"{output_size}. Supported sizes are 8 and 9."
        )
    return {
        "kind": "numpy",
        "w1": policy.w1,
        "b1": policy.b1,
        "w2": policy.w2,
        "b2": policy.b2,
        "output_size": output_size,
    }


def extract_model(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return FALLBACK_MODEL
    model_path = Path(path)
    if zipfile.is_zipfile(model_path):
        return _extract_sb3_model(model_path)
    return _extract_numpy_model(model_path)


def render_submission(model_spec: dict[str, Any]) -> str:
    return STANDALONE_TEMPLATE.replace(
        "__MODEL__", json.dumps(model_spec, separators=(",", ":"))
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a saved Orbit Wars model as a standalone Kaggle submission.py."
    )
    parser.add_argument("--model", default=None, help="Path to a saved policy/model artifact.")
    parser.add_argument("--out", default="submission.py", help="Output submission.py path.")
    args = parser.parse_args()

    try:
        model_spec = extract_model(args.model)
    except Exception as exc:
        parser.exit(2, f"error: could not export model: {exc}\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_submission(model_spec), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
