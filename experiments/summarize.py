"""Aggregate the main table, or recompute its 45 physical-block comparisons."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import numpy as np
from .common import PROJECT, SYSTEMS, SEEDS
from .bootstrap import bootstrap, holm_adjust

REGIMES = ("loso", "fewshot_0.01", "fewshot_0.05", "unseen_target", "joint")
METHODS = (
    "moirai",
    "gaugeformer_v13",
    "timer_xl",
    "timer_xl_gaugeformer",
    "gtm",
    "gtm_gaugeformer",
    "unitime",
    "cpiri",
    "tefn",
    "tggc",
)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input", type=Path, default=PROJECT / "results/reference/accuracy.csv"
    )
    p.add_argument(
        "--runs",
        type=Path,
        help="Aggregate a complete freshly evaluated pair directory",
    )
    p.add_argument(
        "--bootstrap", type=Path, help="Locally available block-metric JSON archive"
    )
    p.add_argument("--output", type=Path, default=PROJECT / "results/runs/summary")
    args = p.parse_args()
    if args.runs:
        rows = []
        for file in sorted(args.runs.glob("*/complete.json")):
            r = json.loads(file.read_text())
            if r["limited_panel"] is not None:
                continue
            for v, systems in r["metrics"].items():
                method = (
                    r["method"]
                    if v == "raw"
                    else "gaugeformer_v13"
                    if r["method"] == "moirai" and v == "full"
                    else r["method"] + "_gaugeformer"
                    if v == "full"
                    else r["method"] + "_" + v
                )
                for system, m in systems.items():
                    rows.append(
                        dict(
                            regime=f"fewshot_{r['fraction']}"
                            if r["family"] == "fewshot"
                            else r["family"],
                            method=method,
                            seed=r["seed"],
                            system=system,
                            sd_nmae=m["sd_nmae"],
                            sd_nrmse=m["sd_nrmse"],
                        )
                    )
    else:
        rows = read_csv(args.input)
    keys = {(r["regime"], r["method"], int(r["seed"]), r["system"]) for r in rows}
    expected = {
        (r, m, s, d) for r in REGIMES for m in METHODS for s in SEEDS for d in SYSTEMS
    }
    if keys != expected or len(rows) != 1250:
        raise ValueError(
            f"Complete main comparison requires 1250 unique cells; found {len(keys)}"
        )
    output = []
    print("| Setting | Pipeline | SD-NMAE | SD-NRMSE |")
    print("|---|---|---:|---:|")
    for regime in REGIMES:
        for method in METHODS:
            rec = dict(regime=regime, method=method)
            for metric in ("sd_nmae", "sd_nrmse"):
                values = [
                    np.mean(
                        [
                            float(r[metric])
                            for r in rows
                            if r["regime"] == regime
                            and r["method"] == method
                            and int(r["seed"]) == seed
                        ]
                    )
                    for seed in SEEDS
                ]
                rec[metric] = float(np.mean(values))
                rec[metric + "_seed_sd"] = float(np.std(values, ddof=1))
            output.append(rec)
            print(
                f"| {regime} | {method} | {rec['sd_nmae']:.4f} | {rec['sd_nrmse']:.4f} |"
            )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "macro.json").write_text(json.dumps(output, indent=2))
    if args.bootstrap:
        with gzip.open(args.bootstrap, "rt", encoding="utf-8") as f:
            data = json.load(f)["records"]
        grid = {
            (r["family"], r["fraction"], r["method"], r["seed"], r["system"]): r
            for r in data
        }
        assert len(grid) == 1250
        comparisons = []
        pairs = [
            ("gaugeformer_v13", m)
            for m in ("moirai", "unitime", "timer_xl", "gtm", "cpiri", "tefn", "tggc")
        ] + [("timer_xl_gaugeformer", "timer_xl"), ("gtm_gaugeformer", "gtm")]
        for family, fraction in (
            ("loso", None),
            ("fewshot", 0.01),
            ("fewshot", 0.05),
            ("unseen_target", None),
            ("joint", None),
        ):
            for proposed, baseline in pairs:
                comparisons.append(
                    bootstrap(grid, family, fraction, proposed, baseline)
                )
        for r, adjusted in zip(
            comparisons, holm_adjust([r["p_value"] for r in comparisons])
        ):
            r["holm_p_value"] = adjusted
        expected_rows = read_csv(PROJECT / "results/reference/paired_comparisons.csv")
        lookup = {(r["regime"], r["proposed"], r["baseline"]): r for r in expected_rows}
        for r in comparisons:
            ref = lookup[r["regime"], r["proposed"], r["baseline"]]
            for k in ("gain", "ci_lower", "ci_upper", "p_value", "holm_p_value"):
                if not np.isclose(r[k], float(ref[k]), rtol=0, atol=1e-12):
                    raise RuntimeError(
                        f'Reference comparison differs: {r["regime"]}/{r["proposed"]}/{k}'
                    )
        (args.output / "paired_comparisons.json").write_text(
            json.dumps(comparisons, indent=2)
        )
        print("All 45 comparisons reproduce the reference values.")


if __name__ == "__main__":
    main()
