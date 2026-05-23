#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orbit_wars_rl.env.ppo_planet_env import MAP_SEEDS
from orbit_wars_rl.evaluation.evaluate import evaluate_map_seeds_deterministic
from orbit_wars_rl.models.save_load import load_any_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic PPO eval over fixed map seeds.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", default="runs/eval")
    parser.add_argument("--seed-limit", type=int, default=None)
    parser.add_argument(
        "--opponent",
        choices=("starter", "random", "greedy", "hard", "model"),
        default="starter",
    )
    parser.add_argument("--opponent-model", default=None)
    parser.add_argument("--candidate-player", type=int, default=0)
    parser.add_argument("--max-episode-turns", type=int, default=500)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--require-kaggle", dest="require_kaggle", action="store_true", default=True)
    group.add_argument("--no-require-kaggle", dest="require_kaggle", action="store_false")
    args = parser.parse_args()

    seeds = MAP_SEEDS[: args.seed_limit] if args.seed_limit is not None else MAP_SEEDS
    policy = load_any_policy(args.model)
    metrics, results = evaluate_map_seeds_deterministic(
        policy,
        seeds,
        require_kaggle=args.require_kaggle,
        opponent=args.opponent,
        opponent_model=args.opponent_model,
        candidate_player=args.candidate_player,
        max_episode_turns=args.max_episode_turns,
        verbose=True,
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "map_seed_deterministic_eval.json"
    path.write_text(json.dumps({"metrics": metrics, "results": results}, indent=2) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
