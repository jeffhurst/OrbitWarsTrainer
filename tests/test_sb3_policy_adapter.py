import pytest

np = pytest.importorskip("numpy")

from orbit_wars_rl.models.sb3_policy import SB3PolicyAdapter


class FakeModel:
    def predict(self, obs, deterministic=True):
        assert deterministic is True
        assert obs.shape == (15,)
        return np.array([-1.0, 0.1, 0.2, 1.5, 0.4, 0.5, 0.6, 0.7, 2.0], dtype=np.float32), None


def test_sb3_policy_adapter_predict_clips_and_shapes():
    action = SB3PolicyAdapter(FakeModel()).predict([0.0] * 15)
    assert action == [0.0, 0.10000000149011612, 0.20000000298023224, 1.0, 0.4000000059604645, 0.5, 0.6000000238418579, 0.699999988079071, 1.0]


class FakeMultiDiscreteModel:
    def predict(self, obs, deterministic=True):
        assert deterministic is True
        assert obs.shape == (15,)
        return np.array([5, 0, 3, 1], dtype=np.int64), None


def test_sb3_policy_adapter_accepts_multidiscrete_actions():
    action = SB3PolicyAdapter(FakeMultiDiscreteModel()).predict([0.0] * 15)
    assert action == [5.0, 0.0, 3.0, 1.0]


def test_sb3_policy_adapter_rejects_wrong_obs_shape():
    with pytest.raises(ValueError):
        SB3PolicyAdapter(FakeModel()).predict([0.0] * 14)
