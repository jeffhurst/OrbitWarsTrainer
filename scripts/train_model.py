#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import argparse
from orbit_wars_rl.training.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/models/bootstrap_policy.zip")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(f"wrote {train(args.out, args.seed)}")
