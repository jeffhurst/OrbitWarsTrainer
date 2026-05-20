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
        return True
