import zipfile

from orbit_wars_rl.models.policy import NumpyPolicy
from orbit_wars_rl.models.save_load import load_any_policy


def test_load_any_policy_supports_numpy_policy_bootstrap_artifacts(tmp_path):
    model_path = tmp_path / "bootstrap_policy.json"
    saved_policy = NumpyPolicy.random(3)
    saved_policy.save(model_path)

    loaded_policy = load_any_policy(model_path)

    assert isinstance(loaded_policy, NumpyPolicy)
    assert loaded_policy.predict([0.0] * 15) == saved_policy.predict([0.0] * 15)


def test_load_any_policy_routes_sb3_zip_artifacts_through_adapter(monkeypatch, tmp_path):
    from orbit_wars_rl.models import save_load
    from orbit_wars_rl.models.sb3_policy import SB3PolicyAdapter

    model_path = tmp_path / "ppo_orbit_wars.zip"
    with zipfile.ZipFile(model_path, "w") as artifact:
        artifact.writestr("data", "placeholder")

    loaded_policy = object()
    calls = []

    def fake_load(path):
        calls.append(path)
        return loaded_policy

    monkeypatch.setattr(SB3PolicyAdapter, "load", fake_load)

    assert save_load.load_any_policy(model_path) is loaded_policy
    assert calls == [model_path]
