#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import argparse
from pathlib import Path

STANDALONE = r'''
# Auto-generated Orbit Wars submission. Fallback starter policy, no local imports.
import math

def _dist(a,b): return math.hypot(a[2]-b[2], a[3]-b[3])
def _comet_ids(obs):
    ids = set(obs.get("comet_planet_ids", []) or [])
    for group in obs.get("comets", []) or []:
        if isinstance(group, dict):
            ids.update(group.get("planet_ids", []) or [])
        elif isinstance(group, (list, tuple)) and group:
            first = group[0]
            if isinstance(first, (list, tuple, set)):
                ids.update(first)
    return {int(pid) for pid in ids}
def _orbiting(p): return math.hypot(p[2]-50.0, p[3]-50.0) + p[4] < 50.0
def _quad(p):
    x,y=p[2],p[3]
    if x>=50 and y<50: return 1
    if x<50 and y<50: return 2
    if x<50 and y>=50: return 3
    return 4
def _ccw(q): return {1:2,2:3,3:4,4:1}[q]
def _owner_priority(p, player):
    if p[1] == player: return 2
    if p[1] == -1: return 1
    return 0
def _select(src, planets, player, comets):
    want=_ccw(_quad(src)); out=[]
    for p in planets:
        if p[0] == src[0] or p[0] in comets: continue
        if (not _orbiting(p) and _dist(src,p) <= 25.0) or (_orbiting(p) and _quad(p) == want):
            out.append(p)
    out.sort(key=lambda p: (-p[6], _owner_priority(p, player), _dist(src,p), p[0]))
    return out[:4]
def agent(obs, config=None):
    planets=obs.get("planets", []) or []
    player=obs.get("player", 0)
    comets=_comet_ids(obs)
    actions=[]
    for src in planets:
        if src[1] != player or src[0] in comets: continue
        chosen=_select(src, planets, player, comets)
        if src[5] < len(chosen): continue
        for t in chosen:
            actions.append([int(src[0]), math.atan2(t[3]-src[3], t[2]-src[2]), 1])
    return actions
'''.lstrip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Accepted for pipeline compatibility; fallback starter code is embedded in v1.")
    parser.add_argument("--out", default="submission.py")
    args = parser.parse_args()
    out = Path(args.out)
    out.write_text(STANDALONE)
    print(f"wrote {out}")
