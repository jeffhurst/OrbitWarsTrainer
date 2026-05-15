from orbit_wars_rl.agents.model_agent import ModelAgent


class RecordingPolicy:
    def __init__(self):
        self.observations = []

    def predict(self, obs):
        self.observations.append(obs.tolist())
        return [0.0] * 9


def test_model_agent_reuses_same_production_delta_for_all_sources_in_turn():
    policy = RecordingPolicy()
    agent = ModelAgent(policy=policy)
    first_obs = {
        "player": 0,
        "planets": [
            [0, 0, 80, 80, 10, 5, 1],
            [1, 0, 82, 80, 10, 5, 2],
            [2, -1, 84, 80, 10, 5, 1],
        ],
    }
    second_obs = {
        "player": 0,
        "planets": [
            [0, 0, 80, 80, 10, 5, 3],
            [1, 0, 82, 80, 10, 5, 2],
            [2, -1, 84, 80, 10, 5, 1],
        ],
    }

    agent.act(first_obs)
    policy.observations.clear()

    agent.act(second_obs)

    assert len(policy.observations) == 2
    assert [obs[1] for obs in policy.observations] == [2.0, 2.0]
