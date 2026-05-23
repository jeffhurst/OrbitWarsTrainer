"""Training callback placeholders kept separate from core logic."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    from stable_baselines3.common.callbacks import BaseCallback
except Exception:  # pragma: no cover
    class BaseCallback:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            del args, kwargs


@dataclass(slots=True)
class TrainingLog:
    step: int
    reward: float


class EpisodeComponentLogger(BaseCallback):
    """Logs reward/action/game components at episode boundaries."""

    def _on_step(self) -> bool:
        if not hasattr(self, "locals") or not hasattr(self, "logger"):
            return True
        for info in self.locals.get("infos", []):
            if not isinstance(info, dict):
                continue
            episode = info.get("episode_components") or info.get("episode")
            if not episode:
                continue
            for key, value in episode.items():
                self.logger.record(key, float(value))
            ordered_keys = [
                "reward/terminal",
                "reward/strategic_delta",
                "reward/capture",
                "reward/local_action",
                "reward/total",
                "reward/episode_return",
                "train_rollout/stochastic_win_rate",
                "game/loss_rate",
                "game/timeout_rate",
                "game/avg_turns",
                "game/avg_enemy_captures",
                "game/avg_neutral_captures",
                "game/map_seed",
                "action/invalid_rate",
                "action/ships_sent_mean",
            ]
            print("[episode]")
            matched = False
            for key in ordered_keys:
                if key in episode:
                    print(f"  {key}: {float(episode[key]):.4f}")
                    matched = True
            if not matched:
                for key in sorted(episode):
                    value = episode[key]
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {float(value):.4f}")
                    else:
                        print(f"  {key}: {value}")
        return True


class DeterministicMapSeedEvalCallback(BaseCallback):
    """Run deterministic policy evaluation across fixed map seeds after rollouts."""

    def __init__(
        self,
        *,
        map_seeds: Sequence[int],
        require_kaggle: bool,
        opponent: str = "starter",
        opponent_model: str | Path | None = None,
        candidate_player: int = 0,
        max_episode_turns: int = 500,
        eval_freq_rollouts: int = 1,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.map_seeds = [int(seed) for seed in map_seeds]
        self.require_kaggle = require_kaggle
        self.opponent = opponent
        self.opponent_model = opponent_model
        self.candidate_player = candidate_player
        self.max_episode_turns = max_episode_turns
        self.eval_freq_rollouts = max(1, int(eval_freq_rollouts))
        self.rollout_count = 0

    def _on_rollout_end(self) -> None:
        self.rollout_count += 1
        if self.rollout_count % self.eval_freq_rollouts != 0:
            return

        from orbit_wars_rl.evaluation.evaluate import evaluate_map_seeds_deterministic

        metrics, _results = evaluate_map_seeds_deterministic(
            self.model,
            self.map_seeds,
            require_kaggle=self.require_kaggle,
            opponent=self.opponent,
            opponent_model=self.opponent_model,
            candidate_player=self.candidate_player,
            max_episode_turns=self.max_episode_turns,
            verbose=True,
        )
        for key, value in metrics.items():
            self.logger.record(key, float(value))

    def _on_step(self) -> bool:
        return True
