#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orbit_wars_rl.agents.starter_agent import StarterAgent
from orbit_wars_rl.visualization.watch import watch_agents


def main() -> None:
    a0 = StarterAgent()
    a1 = StarterAgent()
    paths = watch_agents(a0.act, a1.act, out_dir="runs/watch", steps=3)
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
