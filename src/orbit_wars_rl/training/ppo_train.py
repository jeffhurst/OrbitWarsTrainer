"""Stable-Baselines3 PPO training entrypoints."""
from __future__ import annotations

from pathlib import Path

from orbit_wars_rl.training.ppo_config import PPOTrainConfig


def parse_net_arch(value: str) -> tuple[int, ...]:
    try:
        arch = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError("--net-arch must be comma-separated positive integers") from exc
    if not arch or any(width <= 0 for width in arch):
        raise ValueError("--net-arch must contain at least one positive integer")
    return arch


def train_ppo(config: PPOTrainConfig) -> Path:
    """Train a single-env CPU PPO model and save it as an SB3 .zip artifact."""
    config.validate()
    from stable_baselines3 import PPO
    import torch as th

    from orbit_wars_rl.env.ppo_planet_env import OrbitWarsPlanetStepEnv

    env = OrbitWarsPlanetStepEnv(
        opponent=config.opponent,
        candidate_player=config.candidate_player,
        seed=config.seed,
        max_episode_turns=config.max_episode_turns,
        require_kaggle=config.require_kaggle,
    )
    tensorboard_log = config.tensorboard_log
    if tensorboard_log is not None:
        try:
            import tensorboard  # noqa: F401
        except Exception:
            tensorboard_log = None

    policy_kwargs = dict(
        activation_fn=th.nn.ReLU,
        net_arch=dict(pi=list(config.net_arch), vf=list(config.net_arch)),
    )
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        seed=config.seed,
        verbose=config.verbose,
        device="cpu",
    )
    model.learn(total_timesteps=config.timesteps, progress_bar=False)
    out = Path(config.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    return out
