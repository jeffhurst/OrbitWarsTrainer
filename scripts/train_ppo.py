#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orbit_wars_rl.training.ppo_config import PPOTrainConfig
from orbit_wars_rl.training.ppo_train import parse_net_arch, train_ppo


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an SB3 PPO policy for Orbit Wars.")
    parser.add_argument("--out", default="runs/models/ppo_orbit_wars.zip")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--net-arch", default="256,256,128")
    parser.add_argument("--opponent", choices=("starter", "model"), default="starter")
    parser.add_argument("--opponent-model", default=None)
    parser.add_argument("--candidate-player", type=int, default=0)
    parser.add_argument("--max-episode-turns", type=int, default=400)
    parser.add_argument("--tensorboard-log", default="runs/tensorboard")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--require-kaggle", dest="require_kaggle", action="store_true", default=True)
    group.add_argument("--no-require-kaggle", dest="require_kaggle", action="store_false")
    args = parser.parse_args()

    if not args.require_kaggle:
        print(
            "WARNING: using the deterministic fake smoke backend; "
            "the resulting PPO model is not meaningful for real Orbit Wars training."
        )
    config = PPOTrainConfig(
        out=args.out,
        timesteps=args.timesteps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        net_arch=parse_net_arch(args.net_arch),
        opponent=args.opponent,
        opponent_model=args.opponent_model,
        candidate_player=args.candidate_player,
        max_episode_turns=args.max_episode_turns,
        require_kaggle=args.require_kaggle,
        tensorboard_log=args.tensorboard_log if args.tensorboard_log else None,
    )
    print(f"wrote {train_ppo(config)}")


if __name__ == "__main__":
    main()
