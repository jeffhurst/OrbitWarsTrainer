"""Agent-local cross-turn state."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ProductionTracker:
    previous_by_player: dict[int, float] = field(default_factory=dict)

    def change(self, player: int, current_total: float) -> float:
        if player not in self.previous_by_player:
            self.previous_by_player[player] = current_total
            return 0.0
        delta = current_total - self.previous_by_player[player]
        self.previous_by_player[player] = current_total
        return float(delta)
