"""Kaggle-backed watch/replay utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from orbit_wars_rl.env.kaggle_env import require_kaggle_env

AgentCallable = Callable[[dict[str, Any]], list]
KaggleAgentCallable = Callable[[dict[str, Any], Any | None], list]


def _observation_dict(obs: Any) -> dict[str, Any]:
    if isinstance(obs, dict):
        return obs
    if hasattr(obs, "items"):
        return dict(obs.items())
    if hasattr(obs, "__dict__"):
        return vars(obs)
    raise TypeError(f"Unsupported Kaggle observation type: {type(obs)!r}")


def kaggle_agent(agent: AgentCallable) -> KaggleAgentCallable:
    """Wrap a local one-argument agent for Kaggle's ``(observation, config)`` call shape."""

    def wrapped(obs: dict[str, Any], config: Any | None = None) -> list:
        del config
        return agent(_observation_dict(obs))

    return wrapped


def _render_html(env: Any) -> str:
    """Render a completed Kaggle episode to HTML across supported package versions."""
    try:
        rendered = env.render(mode="html", width=1000, height=800)
    except TypeError:
        rendered = env.render(mode="html")
    if rendered is None:
        raise RuntimeError("Kaggle environment did not return HTML from env.render(mode='html').")
    return str(rendered)


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(v) for v in value]
        return repr(value)


def watch_agents(
    agent0: AgentCallable,
    agent1: AgentCallable,
    out_dir: str | Path = "runs/watch",
    name: str = "starter_vs_starter",
    seed: int | None = None,
) -> list[Path]:
    """Run a real Kaggle Orbit Wars episode and save an HTML replay plus episode JSON.

    This intentionally does not use the old synthetic snapshot renderer. It follows the Kaggle
    environment workflow from the public getting-started notebook: create the ``orbit_wars``
    environment, run two agents against each other, and render the completed episode.
    """
    make_kwargs: dict[str, Any] = {"debug": True}
    if seed is not None:
        make_kwargs["configuration"] = {"seed": int(seed)}
    env = require_kaggle_env(**make_kwargs)
    env.run([kaggle_agent(agent0), kaggle_agent(agent1)])

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    html_path = out / f"{name}.html"
    json_path = out / f"{name}.json"
    html_path.write_text(_render_html(env), encoding="utf-8")
    json_path.write_text(json.dumps(_json_safe(env.toJSON()), indent=2) + "\n", encoding="utf-8")
    return [html_path, json_path]
