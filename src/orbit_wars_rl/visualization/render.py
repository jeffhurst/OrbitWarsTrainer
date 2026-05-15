"""Lightweight renderer for Orbit Wars observations.

Uses matplotlib when available. If it is not installed, writes an SVG fallback that still shows
planets, comets, candidate lines, and fleets.
"""
from __future__ import annotations

from pathlib import Path

from orbit_wars_rl.core.candidates import comet_ids_from_obs, select_candidates
from orbit_wars_rl.core.types import parse_fleets, parse_planets

COLORS = {-1: "#9e9e9e", 0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}


def _render_svg(obs: dict, out: Path, title: str, show_candidates: bool) -> Path:
    if out.suffix.lower() != ".svg":
        out = out.with_suffix(".svg")
    planets = parse_planets(obs)
    fleets = parse_fleets(obs)
    player = int(obs.get("player", 0))
    comet_ids = comet_ids_from_obs(obs)
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="700" height="700" viewBox="0 0 100 100">',
        '<rect width="100" height="100" fill="#08111f"/>',
        f'<text x="2" y="5" fill="white" font-size="3">{title}</text>',
        '<circle cx="50" cy="50" r="10" fill="gold" fill-opacity="0.35"/>',
    ]
    if show_candidates:
        for source in [p for p in planets if p.owner == player and p.id not in comet_ids]:
            for target in select_candidates(source, planets, player, comet_ids).candidates:
                lines.append(f'<line x1="{source.x}" y1="{source.y}" x2="{target.x}" y2="{target.y}" stroke="lime" stroke-width="0.3" stroke-dasharray="1,1"/>')
    for p in planets:
        color = "#000000" if p.id in comet_ids else COLORS.get(p.owner, "purple")
        stroke = "#ffffff" if p.id in comet_ids else "#222222"
        lines.append(f'<circle cx="{p.x}" cy="{p.y}" r="{max(0.8, p.radius)}" fill="{color}" fill-opacity="0.75" stroke="{stroke}" stroke-width="0.25"/>')
        lines.append(f'<text x="{p.x}" y="{p.y}" fill="white" font-size="2" text-anchor="middle" dominant-baseline="central">{p.id}</text>')
    for f in fleets:
        color = COLORS.get(f.owner, "purple")
        lines.append(f'<text x="{f.x}" y="{f.y}" fill="{color}" font-size="3" text-anchor="middle">×</text>')
    lines.append("</svg>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out


def render_observation(obs: dict, out: str | Path, title: str = "Orbit Wars", show_candidates: bool = True) -> Path:
    out_path = Path(out)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return _render_svg(obs, out_path, title, show_candidates)

    planets = parse_planets(obs)
    fleets = parse_fleets(obs)
    player = int(obs.get("player", 0))
    comet_ids = comet_ids_from_obs(obs)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title(title)
    ax.set_xlim(0, 100)
    ax.set_ylim(100, 0)
    ax.set_aspect("equal")
    ax.add_patch(plt.Circle((50, 50), 10, color="gold", alpha=0.35))
    if show_candidates:
        for source in [p for p in planets if p.owner == player and p.id not in comet_ids]:
            for target in select_candidates(source, planets, player, comet_ids).candidates:
                ax.plot([source.x, target.x], [source.y, target.y], "--", color="lime", lw=1)
                ax.annotate("", xy=(target.x, target.y), xytext=(source.x, source.y), arrowprops=dict(arrowstyle="->", color="lime", lw=0.8))
    for p in planets:
        color = "black" if p.id in comet_ids else COLORS.get(p.owner, "purple")
        ax.add_patch(plt.Circle((p.x, p.y), max(0.8, p.radius), color=color, alpha=0.65))
        ax.text(p.x, p.y, f"{p.id}\n{p.ships}/{p.production:g}", ha="center", va="center", fontsize=7, color="white")
    for f in fleets:
        ax.scatter([f.x], [f.y], marker="x", color=COLORS.get(f.owner, "purple"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path
