import json

from orbit_wars_rl.visualization import watch


class AttrObservation:
    def __init__(self, player):
        self.player = player


class FakeEnv:
    def __init__(self):
        self.ran_agents = None

    def run(self, agents):
        self.ran_agents = agents
        obs = {"player": 0}
        assert agents[0](obs, {"unused": True}) == [[1, 0.0, 1]]
        assert agents[1](AttrObservation(1), None) == [[2, 3.14, 1]]

    def render(self, mode, width, height):
        assert mode == "html"
        assert width == 1000
        assert height == 800
        return "<html>replay</html>"

    def toJSON(self):
        return {"steps": [[{"observation": {"player": 0}}]]}


def test_watch_agents_runs_kaggle_env_and_writes_replay(monkeypatch, tmp_path):
    fake = FakeEnv()
    monkeypatch.setattr(watch, "require_kaggle_env", lambda **kwargs: fake)

    paths = watch.watch_agents(
        lambda obs: [[1, 0.0, 1]],
        lambda obs: [[2, 3.14, 1]],
        out_dir=tmp_path,
        name="match",
    )

    assert [p.name for p in paths] == ["match.html", "match.json"]
    assert (tmp_path / "match.html").read_text() == "<html>replay</html>"
    assert json.loads((tmp_path / "match.json").read_text())["steps"][0][0]["observation"]["player"] == 0
    assert fake.ran_agents is not None
