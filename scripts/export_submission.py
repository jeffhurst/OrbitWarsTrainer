#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

STANDALONE_TEMPLATE = r'''
# Auto-generated Orbit Wars submission. Pure Python; no local, SB3, torch, or numpy imports.
import math

_WEIGHTS = __WEIGHTS__
_SUN_RADIUS = 10.0
_PREV_TOTAL = {}

def _matvec(x, w, b):
    out=[]
    for j in range(len(b)):
        z=b[j]
        for i in range(min(len(x), len(w))):
            z += float(x[i]) * float(w[i][j])
        out.append(z)
    return out

def _relu(v): return [z if z > 0.0 else 0.0 for z in v]
def _clip01(v): return 0.0 if v < 0.0 else 1.0 if v > 1.0 else float(v)
def _predict(obs):
    x=[float(v) for v in obs[:15]]
    for layer in _WEIGHTS["hidden"]:
        x=_relu(_matvec(x, layer["w"], layer["b"]))
    y=_matvec(x, _WEIGHTS["action"]["w"], _WEIGHTS["action"]["b"])
    return [_clip01(v) for v in y]

def _dist(a,b): return math.hypot(a[2]-b[2], a[3]-b[3])
def _comet_ids(obs):
    ids=set(obs.get("comet_planet_ids", []) or [])
    for group in obs.get("comets", []) or []:
        if isinstance(group, dict): ids.update(group.get("planet_ids", []) or [])
        elif isinstance(group, (list, tuple)) and group:
            first=group[0]
            if isinstance(first, (list, tuple, set)): ids.update(first)
    return {int(pid) for pid in ids}
def _orbiting(p): return math.hypot(p[2]-50.0, p[3]-50.0) + p[4] < 50.0
def _future_orbit_xy(p, dt, angular_velocity):
    cx,cy=50.0,50.0; dx,dy=p[2]-cx,p[3]-cy; ang=angular_velocity*dt
    ca,sa=math.cos(ang),math.sin(ang)
    return cx + dx*ca - dy*sa, cy + dx*sa + dy*ca
def _intercept_xy(src, t, angular_velocity, fleet_speed=1.0):
    if angular_velocity == 0.0 or fleet_speed <= 0.0 or not _orbiting(t): return t[2], t[3]
    tx,ty=t[2],t[3]; travel=math.hypot(tx-src[2], ty-src[3]) / fleet_speed
    for _ in range(8):
        tx,ty=_future_orbit_xy(t, travel, angular_velocity)
        next_travel=math.hypot(tx-src[2], ty-src[3]) / fleet_speed
        if abs(next_travel-travel) < 1e-9: break
        travel=next_travel
    return tx,ty
def _intercept_angle(src, t, angular_velocity, fleet_speed=1.0):
    tx,ty=_intercept_xy(src, t, angular_velocity, fleet_speed)
    return math.atan2(ty-src[3], tx-src[2])
def _sun_radius(obs):
    for key in ("sun_collision_radius", "sun_radius"):
        if obs.get(key) is not None: return float(obs[key])
    cfg=obs.get("configuration") or obs.get("config") or {}
    if hasattr(cfg, "get"):
        for key in ("sun_collision_radius", "sun_radius"):
            if cfg.get(key) is not None: return float(cfg[key])
    return _SUN_RADIUS
def _segment_intersects_circle(ax, ay, bx, by, cx, cy, radius):
    abx,aby=bx-ax,by-ay; den=abx*abx+aby*aby
    if den == 0.0: return math.hypot(ax-cx, ay-cy) <= radius
    u=((cx-ax)*abx+(cy-ay)*aby)/den; u=max(0.0,min(1.0,u))
    return math.hypot((ax+u*abx)-cx, (ay+u*aby)-cy) <= radius
def _crosses_sun(src, target_xy, radius):
    return _segment_intersects_circle(src[2], src[3], target_xy[0], target_xy[1], 50.0, 50.0, radius)
def _quad(p):
    x,y=p[2],p[3]
    if x>=50 and y<50: return 1
    if x<50 and y<50: return 2
    if x<50 and y>=50: return 3
    return 4
def _ccw(q): return {1:2,2:3,3:4,4:1}[q]
def _owner_priority(p, player):
    if int(p[1]) == int(player): return 2
    if int(p[1]) == -1: return 1
    return 0
def _select(src, planets, player, comets):
    want=_ccw(_quad(src)); out=[]
    for p in planets:
        if int(p[0]) == int(src[0]) or int(p[0]) in comets: continue
        if (not _orbiting(p) and _dist(src,p) <= 25.0) or (_orbiting(p) and _quad(p) == want): out.append(p)
    out.sort(key=lambda p: (-float(p[6]), _owner_priority(p, player), _dist(src,p), int(p[0])))
    return out[:4]
def _total_production(planets, player): return sum(float(p[6]) for p in planets if int(p[1]) == int(player))
def _build_obs(src, chosen, planets, player, previous_total):
    total=_total_production(planets, player); values=[total, total-previous_total, float(src[5])]
    for i in range(4):
        if i < len(chosen):
            p=chosen[i]; enc=1 if int(p[1]) == int(player) else 0 if int(p[1]) == -1 else -1
            ships=float(p[5]) if enc == 1 else -float(p[5])
            values.extend([float(enc), ships, float(p[6])])
        else: values.extend([0.0,0.0,0.0])
    return values
def _decode(src, chosen, outputs, angular_velocity, fleet_speed, sun_radius):
    if len(outputs) != 9 or float(outputs[8]) > 0.5: return []
    remaining=max(0, int(src[5])-1); actions=[]
    for idx in range(min(4, len(chosen))):
        if float(outputs[idx*2]) <= 0.5 or remaining <= 0: continue
        min_send=min(10, remaining)
        ships=min(max(min_send, int(math.floor(float(src[5]) * _clip01(outputs[idx*2+1])))), remaining)
        if ships <= 0: continue
        target=chosen[idx]; tx,ty=_intercept_xy(src, target, angular_velocity, fleet_speed)
        if _crosses_sun(src, (tx,ty), sun_radius): continue
        remaining -= ships
        actions.append([int(src[0]), float(math.atan2(ty-src[3], tx-src[2])), int(ships)])
    return actions

def agent(obs, config=None):
    del config
    planets=obs.get("planets", []) or []; player=int(obs.get("player", 0))
    angular_velocity=float(obs.get("angular_velocity", 0.0)); fleet_speed=float(obs.get("fleet_speed", 1.0))
    sun_radius=_sun_radius(obs); comets=_comet_ids(obs); total=_total_production(planets, player)
    previous_total=_PREV_TOTAL.get(player, total); _PREV_TOTAL[player]=total
    actions=[]
    for src in planets:
        if int(src[1]) != player or int(src[0]) in comets: continue
        chosen=_select(src, planets, player, comets)
        model_obs=_build_obs(src, chosen, planets, player, previous_total)
        actions.extend(_decode(src, chosen, _predict(model_obs), angular_velocity, fleet_speed, sun_radius))
    return actions
'''.lstrip()

FALLBACK_WEIGHTS = {"hidden": [{"w": [[0.0] * 9 for _ in range(15)], "b": [1.0] * 8 + [0.0]}], "action": {"w": [[0.0] * 9 for _ in range(9)], "b": [1.0, 0.2, 1.0, 0.2, 1.0, 0.2, 1.0, 0.2, 0.0]}}
STANDALONE = STANDALONE_TEMPLATE.replace("__WEIGHTS__", json.dumps(FALLBACK_WEIGHTS, separators=(",", ":")))


def _linear_to_layer(linear):
    weight = linear.weight.detach().cpu().numpy().tolist()  # out x in
    bias = linear.bias.detach().cpu().numpy().tolist()
    # Standalone matvec expects in x out.
    transposed = [list(col) for col in zip(*weight)]
    return {"w": transposed, "b": bias}


def _extract_ppo_weights(path: str | Path) -> dict:
    from stable_baselines3 import PPO
    import torch as th

    model = PPO.load(str(path), device="cpu")
    hidden = []
    for module in model.policy.mlp_extractor.policy_net:
        if isinstance(module, th.nn.Linear):
            hidden.append(_linear_to_layer(module))
        elif isinstance(module, th.nn.ReLU):
            continue
        else:
            raise TypeError(f"Unsupported policy_net module for standalone export: {module!r}")
    return {"hidden": hidden, "action": _linear_to_layer(model.policy.action_net)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", default="submission.py")
    args = parser.parse_args()
    if args.model and str(args.model).endswith(".zip"):
        weights = _extract_ppo_weights(args.model)
    else:
        weights = FALLBACK_WEIGHTS
    out = Path(args.out)
    out.write_text(STANDALONE_TEMPLATE.replace("__WEIGHTS__", json.dumps(weights, separators=(",", ":"))))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
