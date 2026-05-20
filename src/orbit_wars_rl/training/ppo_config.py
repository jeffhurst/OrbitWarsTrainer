"""Configuration for Stable-Baselines3 PPO training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PPOTrainConfig:
    out: str | Path = "runs/models/ppo_orbit_wars.zip"
    timesteps: int = 100_000
    seed: int = 0

    # Single env for now.
    n_envs: int = 1

    # PPO hyperparameters for per-planet-action Orbit Wars env.
    learning_rate: float = 7.5e-5
    n_steps: int = 18432
    batch_size: int = 512
    n_epochs: int = 10
    gamma: float = 0.999
    gae_lambda: float = 0.97
    clip_range: float = 0.15
    ent_coef: float = 0.003
    vf_coef: float = 0.75
    max_grad_norm: float = 0.5

    # Network.
    net_arch: tuple[int, ...] = (256, 256, 128)

    # Orbit Wars env behavior.
    opponent: str = "starter"
    opponent_model: str | Path | None = None
    candidate_player: int = 0
    max_episode_turns: int = 500
    require_kaggle: bool = True

    # Logging.
    tensorboard_log: str | None = "runs/tensorboard"
    verbose: int = 0

    def validate(self) -> None:
        """Raise ``ValueError`` if this config requests unsupported PPO training."""
        if self.timesteps <= 0:
            raise ValueError("timesteps must be > 0")
        if self.n_envs != 1:
            raise ValueError("only n_envs == 1 is supported for now")
        if self.candidate_player not in (0, 1):
            raise ValueError("candidate_player must be 0 or 1")
        if self.opponent == "starter":
            if self.opponent_model is not None:
                raise ValueError("opponent_model must be None when opponent='starter'")
        elif self.opponent == "model":
            if self.opponent_model is None:
                raise ValueError("opponent_model is required when opponent='model'")
        else:
            raise ValueError("opponent must be one of: 'starter', 'model'")
        if not self.net_arch:
            raise ValueError("net_arch must be nonempty")
        if any((not isinstance(width, int)) or width <= 0 for width in self.net_arch):
            raise ValueError("net_arch must contain only positive integers")
