import pytest

from orbit_wars_rl.training.ppo_config import PPOTrainConfig
from orbit_wars_rl.training.ppo_train import parse_net_arch


def test_ppo_config_defaults_validate():
    cfg = PPOTrainConfig()
    cfg.validate()
    assert cfg.net_arch == (256, 256, 128)
    assert cfg.n_envs == 1
    assert cfg.opponent_model is None


def test_ppo_config_supported_non_model_opponents_validate_without_model_path():
    for opponent in ("starter", "random", "greedy", "hard"):
        PPOTrainConfig(opponent=opponent).validate()


def test_ppo_config_model_opponent_requires_model_path():
    cfg = PPOTrainConfig(opponent="model", opponent_model="runs/models/some_model.zip")
    cfg.validate()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timesteps": 0},
        {"n_envs": 2},
        {"candidate_player": 2},
        {"opponent": "starter", "opponent_model": "runs/models/some_model.zip"},
        {"opponent": "random", "opponent_model": "runs/models/some_model.zip"},
        {"opponent": "greedy", "opponent_model": "runs/models/some_model.zip"},
        {"opponent": "hard", "opponent_model": "runs/models/some_model.zip"},
        {"opponent": "unknown"},
        {"opponent": "model"},
        {"net_arch": ()},
        {"net_arch": (64, 0)},
        {"eval_freq_rollouts": 0},
        {"eval_seed_limit": 0},
    ],
)
def test_ppo_config_validation_rejects_unsupported_values(kwargs):
    with pytest.raises(ValueError):
        PPOTrainConfig(**kwargs).validate()


def test_parse_net_arch():
    assert parse_net_arch("256, 256,128") == (256, 256, 128)
    with pytest.raises(ValueError):
        parse_net_arch("64,0")
