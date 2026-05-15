#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import argparse
from orbit_wars_rl.evaluation.evaluate import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--games", type=int, default=4)
    args = parser.parse_args()
    print(f"wrote {evaluate(args.model, args.games)}")
