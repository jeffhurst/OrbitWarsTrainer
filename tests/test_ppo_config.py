import pytest

from orbit_wars_rl.training.ppo_config import PPOTrainConfig
from orbit_wars_rl.training.ppo_train import parse_net_arch


def test_ppo_config_defaults_validate():
    cfg = PPOTrainConfig()
    cfg.validate()
    assert cfg.net_arch == (256, 256, 128)
    assert cfg.n_envs == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timesteps": 0},
        {"n_envs": 2},
        {"candidate_player": 2},
        {"opponent": "random"},
        {"net_arch": ()},
        {"net_arch": (64, 0)},
    ],
)
def test_ppo_config_validation_rejects_unsupported_values(kwargs):
    with pytest.raises(ValueError):
        PPOTrainConfig(**kwargs).validate()


def test_parse_net_arch():
    assert parse_net_arch("256, 256,128") == (256, 256, 128)
    with pytest.raises(ValueError):
        parse_net_arch("64,0")
