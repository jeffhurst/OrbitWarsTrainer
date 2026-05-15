"""Small 15->9 MLP policy implemented with the Python standard library."""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(slots=True)
class NumpyPolicy:
    """A lightweight MLP policy; name kept for API compatibility with early scripts."""

    w1: list[list[float]]
    b1: list[float]
    w2: list[list[float]]
    b2: list[float]

    @classmethod
    def random(cls, seed: int = 0, hidden: int = 32) -> "NumpyPolicy":
        rng = random.Random(seed)
        return cls(
            [[rng.gauss(0, 0.05) for _ in range(hidden)] for _ in range(15)],
            [0.0 for _ in range(hidden)],
            [[rng.gauss(0, 0.05) for _ in range(9)] for _ in range(hidden)],
            [0.0 for _ in range(9)],
        )

    def predict(self, obs: Sequence[float]) -> list[float]:
        hidden = []
        for j, bias in enumerate(self.b1):
            z = bias + sum(float(obs[i]) * self.w1[i][j] for i in range(min(15, len(obs))))
            hidden.append(math.tanh(z))
        out = []
        for k, bias in enumerate(self.b2):
            z = bias + sum(hidden[j] * self.w2[j][k] for j in range(len(hidden)))
            out.append(1.0 / (1.0 + math.exp(-z)))
        return out

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"w1": self.w1, "b1": self.b1, "w2": self.w2, "b2": self.b2}))

    @classmethod
    def load(cls, path: str | Path) -> "NumpyPolicy":
        data = json.loads(Path(path).read_text())
        return cls(data["w1"], data["b1"], data["w2"], data["b2"])
