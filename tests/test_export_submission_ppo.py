import importlib.util
import subprocess
import sys

import pytest


def test_export_ppo_submission_imports_and_returns_actions(tmp_path):
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("sb3_contrib")
    th = pytest.importorskip("torch")
    from sb3_contrib import MaskablePPO

    from orbit_wars_rl.training.ppo_config import PPOTrainConfig
    from orbit_wars_rl.training.ppo_train import train_ppo

    model_path = train_ppo(
        PPOTrainConfig(
            out=tmp_path / "ppo.zip",
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
    model = MaskablePPO.load(str(model_path), device="cpu")
    with th.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.fill_(-10.0)
        model.policy.action_net.bias[1] = 10.0
        model.policy.action_net.bias[5 + 100] = 10.0
    model.save(str(model_path))

    submission = tmp_path / "submission.py"
    subprocess.run(
        [sys.executable, "scripts/export_submission.py", "--model", str(model_path), "--out", str(submission)],
        check=True,
        text=True,
        capture_output=True,
    )
    text = submission.read_text()
    assert "stable_baselines3" not in text
    assert "orbit_wars_rl" not in text
    assert "torch" not in text.lower()
    assert "import numpy" not in text.lower()

    spec = importlib.util.spec_from_file_location("submission", submission)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    obs = {
        "player": 0,
        "planets": [[0, 0, 80, 80, 10, 20, 1], [1, -1, 90, 80, 10, 5, 9]],
        "angular_velocity": 0.0,
    }
    assert module._MODEL["kind"] == "multidiscrete"
    assert module._MODEL["nvec"] == [5, 101]
    assert module._predict([0.0] * 15) == [1.0, 100.0]
    actions = module.agent(obs, None)
    assert isinstance(actions, list)
    assert actions
    for action in actions:
        assert len(action) == 3
        assert int(action[0]) == action[0]
        assert isinstance(float(action[1]), float)
        assert int(action[2]) == action[2]
        assert action[2] > 0
