from orbit_wars_rl.training.callbacks import DeterministicMapSeedEvalCallback


class DummyLogger:
    def __init__(self):
        self.records = []
        self.dumps = []

    def record(self, key, value):
        self.records.append((key, value))

    def dump(self, step):
        self.dumps.append(step)


def _attach_runtime(callback):
    callback.model = object()
    callback.logger = DummyLogger()
    callback.num_timesteps = 123
    return callback.logger


def test_deterministic_eval_defaults_to_training_end_only(monkeypatch):
    from orbit_wars_rl.evaluation import evaluate

    calls = []

    def fake_eval(policy, map_seeds, **kwargs):
        del policy
        calls.append((list(map_seeds), kwargs))
        return {"eval/test_metric": 1.0}, []

    monkeypatch.setattr(evaluate, "evaluate_map_seeds_deterministic", fake_eval)
    callback = DeterministicMapSeedEvalCallback(map_seeds=[11, 12], require_kaggle=False)
    logger = _attach_runtime(callback)

    callback._on_rollout_end()
    callback._on_rollout_end()
    assert calls == []

    callback._on_training_end()
    assert len(calls) == 1
    assert calls[0][0] == [11, 12]
    assert logger.records == [("eval/test_metric", 1.0)]
    assert logger.dumps == [123]


def test_deterministic_eval_uses_rollout_frequency_when_set(monkeypatch):
    from orbit_wars_rl.evaluation import evaluate

    calls = []

    def fake_eval(policy, map_seeds, **kwargs):
        del policy, kwargs
        calls.append(list(map_seeds))
        return {"eval/test_metric": 1.0}, []

    monkeypatch.setattr(evaluate, "evaluate_map_seeds_deterministic", fake_eval)
    callback = DeterministicMapSeedEvalCallback(
        map_seeds=[21],
        require_kaggle=False,
        eval_freq_rollouts=2,
    )
    _attach_runtime(callback)

    callback._on_rollout_end()
    assert calls == []

    callback._on_rollout_end()
    assert calls == [[21]]

    callback._on_training_end()
    assert calls == [[21]]
