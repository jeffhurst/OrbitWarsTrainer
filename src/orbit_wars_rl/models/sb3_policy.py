"""Adapter that lets existing ModelAgent code use saved SB3 PPO policies."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


class SB3PolicyAdapter:
    """Small predict-only wrapper around a Stable-Baselines3 PPO model."""

    def __init__(self, model: Any):
        self.model = model

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "SB3PolicyAdapter":
        from stable_baselines3 import PPO
        from sb3_contrib import MaskablePPO

        model_path = str(path)
        try:
            model = MaskablePPO.load(model_path, device=device)
        except Exception:
            model = PPO.load(model_path, device=device)
        return cls(model)

    def predict(
        self,
        obs: Sequence[float],
        deterministic: bool = True,
        action_masks: Sequence[bool] | None = None,
    ) -> list[float]:
        import numpy as np

        obs_array = np.asarray(list(obs), dtype=np.float32)
        if obs_array.shape != (15,):
            raise ValueError(f"expected observation shape (15,), got {obs_array.shape}")
        predict_kwargs: dict[str, Any] = {"deterministic": deterministic}
        if action_masks is not None:
            predict_kwargs["action_masks"] = action_masks
        try:
            action, _state = self.model.predict(obs_array, **predict_kwargs)
        except TypeError:
            predict_kwargs.pop("action_masks", None)
            action, _state = self.model.predict(obs_array, **predict_kwargs)
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_array.shape in ((2,), (4,)):
            return action_array.astype(float).tolist()
        if action_array.shape in ((8,), (9,)):
            return np.clip(action_array, 0.0, 1.0).astype(float).tolist()
        raise ValueError(
            f"expected action shape (2,), (4,), (8,), or (9,), got {action_array.shape}"
        )
