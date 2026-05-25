from orbit_wars_rl.agents.model_agent import ModelAgent
from orbit_wars_rl.core.candidates import CandidateConfig


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


def test_model_agent_observes_and_decodes_filtered_candidates():
    class AttackFirstPolicy:
        def __init__(self):
            self.observation = None

        def predict(self, obs):
            self.observation = obs.tolist()
            return [5, 0, 0, 0]

    policy = AttackFirstPolicy()
    agent = ModelAgent(policy=policy, candidate_config=CandidateConfig(static_radius=120))
    obs = {
        "player": 0,
        "planets": [
            [0, 0, 0, 50, 5, 10, 1],
            [1, 1, 100, 50, 5, 1, 9],
            [2, 1, 0, 80, 5, 2, 6],
        ],
    }

    actions = agent.act(obs)

    assert policy.observation[3:6] == [-1.0, -2.0, 6.0]
    assert isinstance(actions, list)


def test_model_agent_accepts_tuple_predict_output():
    class TuplePolicy:
        def predict(self, obs):
            del obs
            return [1, 1], None

    agent = ModelAgent(policy=TuplePolicy(), candidate_config=CandidateConfig(static_radius=120))
    obs = {
        "player": 0,
        "planets": [
            [0, 0, 0, 50, 5, 20, 1],
            [1, 1, 100, 50, 5, 1, 9],
        ],
    }

    actions = agent.act(obs)

    assert isinstance(actions, list)
