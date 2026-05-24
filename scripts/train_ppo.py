#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orbit_wars_rl.training.ppo_config import PPOTrainConfig
from orbit_wars_rl.training.ppo_train import parse_net_arch, train_ppo


def main() -> None:
    defaults = PPOTrainConfig()
    parser = argparse.ArgumentParser(description="Train an SB3 PPO policy for Orbit Wars.")
    parser.add_argument("--out", default=str(defaults.out))
    parser.add_argument("--timesteps", type=int, default=defaults.timesteps)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--n-steps", type=int, default=defaults.n_steps)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--n-epochs", type=int, default=defaults.n_epochs)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--gae-lambda", type=float, default=defaults.gae_lambda)
    parser.add_argument("--clip-range", type=float, default=defaults.clip_range)
    parser.add_argument("--ent-coef", type=float, default=defaults.ent_coef)
    parser.add_argument("--vf-coef", type=float, default=defaults.vf_coef)
    parser.add_argument("--max-grad-norm", type=float, default=defaults.max_grad_norm)
    parser.add_argument("--net-arch", default=",".join(str(width) for width in defaults.net_arch))
    parser.add_argument(
        "--opponent",
        choices=("starter", "random", "greedy", "hard", "model"),
        default=defaults.opponent,
    )
    parser.add_argument("--opponent-model", default=defaults.opponent_model)
    parser.add_argument("--candidate-player", type=int, default=defaults.candidate_player)
    parser.add_argument("--max-episode-turns", type=int, default=defaults.max_episode_turns)
    parser.add_argument("--tensorboard-log", default=defaults.tensorboard_log)
    parser.add_argument(
        "--eval-freq-rollouts",
        type=int,
        default=defaults.eval_freq_rollouts,
        help="Run deterministic eval every N completed rollouts. Omit for final-only evaluation.",
    )
    parser.add_argument(
        "--eval-seed-limit",
        type=int,
        default=defaults.eval_seed_limit,
        help="Limit deterministic eval to the first N fixed map seeds. Omit to evaluate all seeds.",
    )
    eval_group = parser.add_mutually_exclusive_group()
    eval_group.add_argument(
        "--deterministic-eval",
        dest="deterministic_eval",
        action="store_true",
        default=defaults.deterministic_eval,
    )
    eval_group.add_argument(
        "--no-deterministic-eval",
        dest="deterministic_eval",
        action="store_false",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=defaults.verbose,
        choices=(0, 1, 2),
        help="SB3 verbosity: 0=no output, 1=training stats table, 2=debug-level details.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--require-kaggle",
        dest="require_kaggle",
        action="store_true",
        default=defaults.require_kaggle,
    )
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
        verbose=args.verbose,
        deterministic_eval=args.deterministic_eval,
        eval_freq_rollouts=args.eval_freq_rollouts,
        eval_seed_limit=args.eval_seed_limit,
    )
    print(f"wrote {train_ppo(config)}")


if __name__ == "__main__":
    main()
