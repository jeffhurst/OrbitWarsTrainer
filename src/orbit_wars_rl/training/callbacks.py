"""Training callback placeholders kept separate from core logic."""
from __future__ import annotations

from dataclasses import dataclass

try:
    from stable_baselines3.common.callbacks import BaseCallback
except Exception:  # pragma: no cover
    class BaseCallback:  # type: ignore[override]
        pass


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
                "reward/pressure",
                "reward/local_action",
                "reward/waste_penalty",
                "reward/total",
                "game/win_rate",
                "game/loss_rate",
                "game/timeout_rate",
                "game/avg_turns",
                "game/avg_enemy_captures",
                "game/avg_neutral_captures",
                "action/invalid_rate",
                "action/ships_sent_mean",
                "action/enemy_target_rate",
                "action/neutral_target_rate",
                "action/self_target_rate",
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
