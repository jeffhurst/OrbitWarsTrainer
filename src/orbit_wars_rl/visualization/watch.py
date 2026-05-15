"""Watch utilities with Kaggle renderer fallback to local matplotlib snapshots."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from orbit_wars_rl.env.kaggle_env import make_kaggle_env
from orbit_wars_rl.visualization.render import render_observation


def make_demo_observation(player: int = 0) -> dict[str, Any]:
    return {
        "player": player,
        "angular_velocity": 0.03,
        "comet_planet_ids": [8],
        "comets": [{"planet_ids": [8], "paths": [], "path_index": 0}],
        "fleets": [],
        "planets": [
            [0, 0, 80, 20, 2.0, 10, 3],
            [1, 1, 20, 80, 2.0, 10, 3],
            [2, -1, 70, 18, 1.7, 8, 5],
            [3, -1, 62, 16, 1.5, 6, 4],
            [4, -1, 35, 35, 1.2, 5, 2],
            [5, -1, 20, 20, 1.2, 5, 5],
            [6, -1, 15, 82, 1.5, 8, 4],
            [7, -1, 75, 78, 1.5, 8, 4],
            [8, -1, 50, 8, 1.0, 4, 1],
        ],
        "initial_planets": [],
    }


def watch_agents(agent0: Callable[[dict], list], agent1: Callable[[dict], list], out_dir: str | Path = "runs/watch", steps: int = 3) -> list[Path]:
    """Run a minimal visual debug session.

    If the Kaggle Orbit Wars env is installed, this function tries to render through it in future
    versions. The guaranteed path is a deterministic local snapshot series that overlays candidate
    lines and the first-turn starter actions, enough to verify target selection and angles.
    """
    env = make_kaggle_env(debug=True)
    # Keep fallback deterministic and always available; Kaggle APIs may vary by release.
    paths: list[Path] = []
    out = Path(out_dir)
    for step in range(steps):
        obs0 = make_demo_observation(0)
        obs1 = make_demo_observation(1)
        actions0 = agent0(obs0)
        actions1 = agent1(obs1)
        fleets = []
        fid = 0
        by_id = {int(p[0]): p for p in obs0["planets"]}
        for owner, actions in [(0, actions0), (1, actions1)]:
            for from_id, angle, ships in actions:
                p = by_id[int(from_id)]
                fleets.append([fid, owner, float(p[2]), float(p[3]), float(angle), int(from_id), int(ships)])
                fid += 1
        obs0["fleets"] = fleets
        path = render_observation(obs0, out / f"starter_vs_starter_{step:03d}.png", title=f"Starter vs Starter debug step {step}")
        paths.append(path)
    return paths
