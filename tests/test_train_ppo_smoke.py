from pathlib import Path

import pytest


def test_train_ppo_fake_smoke(tmp_path):
    pytest.importorskip("stable_baselines3")
    from orbit_wars_rl.training.ppo_config import PPOTrainConfig
    from orbit_wars_rl.training.ppo_train import train_ppo

    out = tmp_path / "ppo_smoke.zip"
    path = train_ppo(
        PPOTrainConfig(
            out=out,
            timesteps=32,
            n_steps=16,
            batch_size=8,
            net_arch=(16,),
            require_kaggle=False,
            tensorboard_log=None,
            verbose=0,
            deterministic_eval=False,
        )
    )
    assert path == out
    assert Path(path).exists()
