"""Preprocess the five public systems with the recorded split rules."""
import argparse
import json
from pathlib import Path
from gaugeformer.data import preprocess_all

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--output", type=Path, default=Path("data/processed"))
    a = p.parse_args()
    print(json.dumps(preprocess_all(a.raw, a.output), indent=2))
