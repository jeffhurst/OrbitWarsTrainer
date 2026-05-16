#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orbit_wars_rl.evaluation.evaluate import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--out-dir", default="runs/eval")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--require-kaggle", dest="require_kaggle", action="store_true", default=True)
    group.add_argument("--no-require-kaggle", dest="require_kaggle", action="store_false")
    args = parser.parse_args()
    if not args.require_kaggle:
        print("WARNING: using fake smoke evaluation backend; metrics are not meaningful for real play.")
    print(f"wrote {evaluate(args.model, args.games, args.out_dir, require_kaggle=args.require_kaggle)}")
