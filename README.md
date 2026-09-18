# GaugeFormer

Code for **GaugeFormer: Historical Validation for Cross-System Forecast Adaptation**.

GaugeFormer uses completed forecasting tasks inside the observed context to select a local trajectory, decide whether it should contribute, and apply a controlled residual correction to a frozen backbone forecast. The requested future is used only for evaluation.

This repository contains the implementation and reproduction assets for the main manuscript. The same decision rule is evaluated around Moirai, Timer-XL and GTM, alongside UniTime, CPiRi, TEFN and TGGC.

## Setup

The experiments use Python 3.12, PyTorch 2.8.0 and CUDA 12.8. The decision module itself requires NumPy. Run the commands below from the repository root.

```bash
git clone https://github.com/ybfo/GaugeFormer.git
cd GaugeFormer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1`. Set `CUDA_VISIBLE_DEVICES` to choose a GPU. `GAUGEFORMER_DEVICE=cpu` selects CPU inference; `GAUGEFORMER_THREADS` controls CPU worker threads.

## Data and checkpoints

Large files are distributed through the `main-v1` GitHub Release. Install the [GitHub CLI](https://cli.github.com/) and run `gh auth login` with an account that can access this private repository.

```bash
python scripts/download.py --group data
python scripts/download.py --group sources
python scripts/download.py --group initialization
python scripts/download.py --group checkpoints --method moirai --seed 17
```

Each model/seed selection downloads the fitted checkpoints for all five regimes: source-only, 1% support, 5% support, withheld-query and joint fitting. Raw/GF pairs share one checkpoint. Downloads are verified against SHA-256 manifests before extraction. Checkpoint packaging preserves every tensor exactly; `original_sha256` retains the original checkpoint identity used by stochastic sampling seeds.

Use `--list` to inspect download sizes, select another model or seed, or use `--group all` for the complete release. Initialization weights and fitted weights are separate. `configs/checkpoints.json` maps each experiment cell to its asset.

The processed data contain Building Energy, Hydraulic, MetroPT, PMSM and Steel Industry with their fixed train/development/test splits. There are 67 forecast queries and 48,455 eligible test windows. The preprocessing implementation is in `gaugeformer/data.py`; `python -m experiments.preprocess --raw data/raw` rebuilds processed arrays from the original source files.

## Evaluation

```bash
# Moirai and Moirai + GaugeFormer, using the same current forecast
python -m experiments.evaluate --method moirai --family loso --target building_energy --seed 17

# Other backbone pairs use the same entry point
python -m experiments.evaluate --method timer_xl --family joint --seed 17
python -m experiments.evaluate --method gtm --family fewshot --fraction .01 --target hydraulic --seed 17

# Independent comparators
python -m experiments.evaluate --method tefn --family loso --target building_energy --seed 17
```

`unitime`, `cpiri` and `tggc` are also supported. Download the corresponding checkpoints first. All eligible test windows are evaluated by default; `--limit 32` runs a small smoke check. Results are written under `results/runs/`. The fitted seeds are 17, 29, 43, 71 and 101. Run each model in a separate process so that its upstream modules remain isolated.

## Main-manuscript analyses

```bash
# Local rules, cumulative local selection, fixed blends and full GF
python -m experiments.evaluate --method moirai --family loso --target building_energy --mode controls

# Five-seed component protocol; repeat for each backbone, seed and source system
python -m experiments.evaluate --method timer_xl --family joint --mode components

# Original system-resolved Moirai component protocol
python -m experiments.evaluate --method moirai --family joint --seed 17 --mode diagnostic-components

# Unit conversion, channel reindexing, deletion and insertion
python -m experiments.stress --method moirai --panel diagnostic --seed 17
python -m experiments.stress --method gtm --panel repeated --seed 17

# Full-path latency and allocated GPU memory; run on an otherwise idle GPU
python -m experiments.efficiency --method moirai

# Recreate the main table from the archived numerical results
python -m experiments.summarize

# Recompute all 45 paired physical-block comparisons and check their values
python scripts/download.py --group results
python -m experiments.summarize --bootstrap results/reference/main-block-metrics.json.gz
```

Component and stress studies use development data. The diagnostic stress panel uses four queries per system and seed 17; the repeated panel uses all forecast queries and five fitted seeds. These populations remain separate. `results/reference/` contains the numerical inputs for the main tables and empirical figures. Archive identifiers use `gaugeformer_v13` for the Moirai + GF pipeline.

The main source-only macro SD-NMAE values are:

| Backbone | Raw | With GaugeFormer | Relative reduction |
|---|---:|---:|---:|
| Moirai | 0.2727 | 0.2690 | 1.36% |
| Timer-XL | 0.3588 | 0.3155 | 12.09% |
| GTM | 0.3861 | 0.3269 | 15.34% |

Outcomes vary across fitting regimes. Under joint fitting, for example, Timer-XL changes from 0.2032 to 0.2096 after correction. The full reference files retain all reported results. Four historical origins add four backbone calls, so latency increases despite unchanged neural parameters.

## Using the decision module

```python
from gaugeformer import GaugeFormer

memory = GaugeFormer()
# context: [channels, 96]; query: [queries]
# current: [queries, 24]; historical: [4, queries, 24]
result = memory.predict(context, query, current, historical)
forecast = result.prediction
```

Historical predictions must be generated from prefixes ending at steps 48, 56, 64 and 72. Their 24-step targets lie inside the observed context. Process windows chronologically and create a new memory for each independent stream or changed query schema. The fixed rule uses a 16-window warm-up, a 0.55 repeated-win threshold and residual weight 0.5. Current evaluation targets are not passed to the decision module.

## Fitting

The original backbone objectives, optimizers and adapters are retained in `backbones/backbones.py`. For example:

```bash
python -m experiments.train --method moirai --family loso --target building_energy --seed 17 --output results/runs/train/moirai_loso_building_seed17
python -m experiments.finetune --method moirai --checkpoint checkpoints/moirai/moirai_loso_seed17_building_energy.pt --target-system building_energy --fraction .01 --seed 17 --output-dir results/runs/finetune/moirai_building_seed17 --max-windows 5000 --learning-rate 5e-8 --batch-size 32
python -m experiments.train_schema --method tefn --family loso --target building_energy --seed 17
```

Source fitting selects checkpoints on source validation data. The pretrained-model fitting configurations are in `configs/training.json`; these were checked against all 425 fitted checkpoints. Few-shot fitting uses deterministic target-training subsets; the selected learning-rate multipliers and budgets are in `configs/fitting.json`. TEFN/TGGC use a 26-slot adapter, randomized slot injection during fitting and canonical injection at evaluation; TGGC evaluates one window per graph. Non-Moirai inference uses target-training location/scale statistics, including in the source-only supervised fitting regime. The Moirai pair uses its observed context.

## Checks and upstream implementations

```bash
python -m pytest -q
```

The release retains the source snapshots used by [Moirai/uni2ts](https://github.com/SalesforceAIResearch/uni2ts), [Timer-XL/OpenLTM](https://github.com/thuml/OpenLTM), [GTM](https://github.com/MMTS4All/GTM), [UniTime](https://github.com/liuxu77/UniTime), [CPiRi](https://github.com/JasonStraka/CPiRi), [TEFN](https://github.com/ztxtech/Time-Evidence-Fusion-Network) and [TGGC](https://github.com/KimMeen/TGGC). File identities are recorded in `configs/upstream.json`. Upstream code and weights retain their respective licenses; copyright and license notices accompany the included sources.
