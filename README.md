# Orbit Wars RL Trainer

A complete Python 3.11+ project for training, evaluating, visualizing, and exporting agents for the Kaggle **Orbit Wars** competition. The first milestone is deliberately focused on reliable core logic and visual debugging before any serious PPO/self-play work.

## What this project does

- Parses Orbit Wars observations (`planets`, `fleets`, `player`, `comet_planet_ids`, etc.).
- Selects candidate planets for each owned source planet.
- Builds the exact 15-value per-planet model observation.
- Decodes 9 model outputs into legal fleet launch actions.
- Provides starter, random, heuristic, and model agents.
- Renders real Kaggle Orbit Wars HTML replays for starter-vs-starter and model-vs-starter games.
- Provides runnable scripts for training bootstrap, evaluation, watching, and Kaggle submission export.

## Orbit Wars at a high level

Orbit Wars is played on a continuous 100x100 board with origin at the top-left and the sun at `(50, 50)`. Players own planets, produce ships, and launch fleets using actions of the form `[from_planet_id, direction_angle, num_ships]`. Angles use `0 = right` and `pi/2 = down`, so Python `atan2(target.y - source.y, target.x - source.x)` matches the game coordinate system.

Planets are `[id, owner, x, y, radius, ships, production]`; fleets are `[id, owner, x, y, angle, from_planet_id, ships]`. Owner `-1` is neutral. Comets appear as planets but are identified by `comet_planet_ids` and are excluded from normal candidate selection.

## Per-owned-planet model strategy

Each turn:

1. Find all planets owned by the current player.
2. For each owned planet, select up to four candidate targets.
3. Build a 15-value observation for that source planet.
4. Invoke the policy once.
5. Decode the 9 outputs into zero or more launch actions.
6. Combine actions from all owned planets.

## Observation format

The default observation builder emits 15 floats:

- `obs0`: total production controlled by current player.
- `obs1`: change in production since previous turn (or explicit previous total).
- `obs2`: source planet ships.
- Four repeated target triples: ownership encoding (`-1 enemy`, `0 neutral`, `1 friendly`), signed ships (friendly positive; enemy/neutral negative), and raw production.

Missing candidates are padded with zero triples.

## Output format

The model returns 9 values:

- Four activation/percentage pairs for the selected targets.
- `out8` no-op override.

Activations are active when `> 0.5`; percentages are clamped to `[0, 1]`; a configurable reserve (default 1 ship) prevents emptying source planets.

## Candidate planet selection

For each owned source planet:

- Static planets are included if not orbiting, not comets, and within 25 Euclidean units.
- Orbiting planets are included if they are in the counterclockwise screen-coordinate quadrant from the source.
- Comets and the source planet are excluded.
- Candidates are sorted by production descending, then enemy > neutral > friendly, then distance ascending, then id.
- Top 4 are used.

Quadrants use screen coordinates around `(50, 50)`:

- Q1: `x >= 50`, `y < 50` top-right.
- Q2: `x < 50`, `y < 50` top-left.
- Q3: `x < 50`, `y >= 50` bottom-left.
- Q4: `x >= 50`, `y >= 50` bottom-right.

Counterclockwise mapping is Q1→Q2→Q3→Q4→Q1.

## Watch starter vs starter

```bash
python scripts/watch_starter_vs_starter.py
```

This follows the Kaggle getting-started workflow: it creates the real `orbit_wars` Kaggle environment, runs `StarterAgent` against `StarterAgent`, and writes `runs/watch/starter_vs_starter.html` plus the raw `runs/watch/starter_vs_starter.json` episode. Install the Kaggle extra first if needed:

```bash
pip install -e .[kaggle]
```

## Train

```bash
python scripts/train_model.py --out runs/models/bootstrap_policy.zip
```

The v1 training command creates a valid 15→9 lightweight MLP policy artifact. It is intentionally a bootstrap artifact so core selection and visualization can be proven before investing in PPO/self-play.

## Evaluate

```bash
python scripts/evaluate_model.py --model runs/models/bootstrap_policy.zip
```

Evaluation writes JSON metrics under `runs/eval/` and includes starter, random, and heuristic opponent scaffolding.

## Watch model vs starter

```bash
python scripts/watch_model_vs_starter.py --model runs/models/bootstrap_policy.zip
```

Like starter-vs-starter, this runs a real Kaggle environment episode and writes replay artifacts under `runs/watch/`.

## Export a Kaggle submission

```bash
python scripts/export_submission.py --model runs/models/bootstrap_policy.zip --out submission.py
```

The v1 exporter writes a standalone fallback starter-style submission with no local package imports. Future work can embed serialized model weights directly into the exported file.

## Tests

```bash
pytest
```

The suite covers geometry, quadrants, candidate selection, observation building, action decoding, comet behavior, and the starter agent.

## Known assumptions and TODOs

- Orbiting planet detection uses `distance_from_center + radius < 50` unless replaced by environment constants.
- Kaggle environment APIs are optional and isolated under `env/` because local installations can differ.
- Watch scripts require a Kaggle environment version that includes `orbit_wars`; they no longer use the old synthetic visual fallback.
- The training pipeline currently bootstraps a policy artifact; full PPO/self-play integration should happen only after visual candidate validation.
- Submission export currently embeds the robust starter fallback; model weight embedding is the next submission milestone.
