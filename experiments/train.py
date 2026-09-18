"""Main-paper fitting configurations for the five pretrained comparators."""
import argparse
from pathlib import Path
from .common import PROJECT, SYSTEMS, read
from backbones.backbones import BaselineTrainConfig, train
from gaugeformer.data import load_query_partition

METHODS = ("moirai", "timer_xl", "gtm", "unitime", "cpiri")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument(
        "--family", choices=("joint", "loso", "unseen_target"), required=True
    )
    p.add_argument("--seed", type=int, choices=(17, 29, 43, 71, 101), default=17)
    p.add_argument("--target", choices=SYSTEMS)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if (a.family == "loso") != (a.target in SYSTEMS):
        p.error("Specify --target only for loso")
    if a.output.exists():
        raise FileExistsError(a.output)
    systems = tuple(sorted(s for s in SYSTEMS if s != a.target))
    query = None
    if a.family == "unseen_target":
        query = load_query_partition(
            PROJECT / "configs/queries.json",
            "supervised_forecast",
            "gf-cs-2026-08-12-v5",
        )
    profiles = read(PROJECT / "configs/training.json")
    profile = next(
        r for r in profiles if r["method"] == a.method and r["family"] == a.family
    )
    config = BaselineTrainConfig(seed=a.seed, **profile["config"])
    train(a.method, a.output, config, systems, None, True, True, True, query)


if __name__ == "__main__":
    main()
