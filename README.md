# GaugeFormer

Code for **GaugeFormer: Historical Validation for Cross-System Forecast Adaptation**.

GaugeFormer uses completed forecasting tasks within the observed context to select a local trajectory, decide whether it should contribute, and apply a controlled residual correction to a frozen backbone forecast. The requested future is used only for evaluation. The same decision mechanism is implemented around Moirai, Timer-XL and GTM.

This repository contains the code for the main manuscript. Data, model weights, result artifacts and manuscript files are not included.

## Installation

The experimental environment uses Python 3.12, PyTorch 2.8.0 and CUDA 12.8. Run commands from the repository root.

```bash
git clone https://github.com/ybfo/GaugeFormer.git
cd GaugeFormer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1`.

## Decision module

The decision module requires only NumPy.

```python
from gaugeformer import GaugeFormer

memory = GaugeFormer()
# context: [channels, 96]; query: [queries]
# current: [queries, 24]; historical: [4, queries, 24]
result = memory.predict(context, query, current, historical)
forecast = result.prediction
```

Generate historical forecasts from prefixes ending at steps 48, 56, 64 and 72. Their 24-step targets lie inside the observed context. Process windows chronologically and create a new memory for each independent stream or changed query schema. The fixed rule uses a 16-window warm-up, a 0.55 repeated-win threshold and residual weight 0.5. Current evaluation targets are not inputs to the decision module.

## Experiments

The code covers Building Energy, Hydraulic, MetroPT, PMSM and Steel Industry. Prepare the original data under `data/raw/` using the filenames specified in `gaugeformer/data.py`, then run:

```bash
python -m experiments.preprocess --raw data/raw
```

Fitting configurations are in `configs/training.json` and `configs/fitting.json`; query definitions are in `configs/queries.json`. The evaluation protocol uses the local paths and recorded checkpoint identities in `configs/checkpoints.json`. Initialization paths are defined in `backbones/backbones.py`; obtain the required pretrained weights from the upstream projects below.

```bash
# Source-only fitting
python -m experiments.train --method moirai --family loso --target building_energy --seed 17 --output results/runs/train/moirai_loso_building_seed17

# Matched backbone / GaugeFormer evaluation
python -m experiments.evaluate --method moirai --family loso --target building_energy --seed 17
python -m experiments.evaluate --method timer_xl --family joint --seed 17
python -m experiments.evaluate --method gtm --family joint --seed 17

# Local controls and component analysis
python -m experiments.evaluate --method moirai --family loso --target building_energy --mode controls
python -m experiments.evaluate --method timer_xl --family joint --mode components
```

The independent comparators are UniTime, CPiRi, TEFN and TGGC. Additional entry points cover few-shot fitting (`experiments.finetune`), TEFN/TGGC fitting (`experiments.train_schema`), measurement transformations (`experiments.stress`), computational cost (`experiments.efficiency`) and result aggregation (`experiments.summarize`). Use `--help` for existing options. Outputs are written under `results/runs/` and are ignored by Git. Run each model in a separate process to isolate its upstream modules.

## Repository structure

| Path | Contents |
|---|---|
| `gaugeformer/` | Historical decision rule, local hypotheses and data utilities |
| `backbones/` | Backbone interfaces and comparator adapters |
| `experiments/` | Main-manuscript fitting, evaluation and analysis code |
| `configs/` | Query definitions and fixed experimental configurations |
| `third_party/` | Required upstream source snapshots and license notices |
| `tests/` | Decision-rule checks |

## Tests

```bash
python -m pytest -q
```

## Acknowledgments

The implementation uses [Moirai/uni2ts](https://github.com/SalesforceAIResearch/uni2ts), [Timer-XL/OpenLTM](https://github.com/thuml/OpenLTM), [GTM](https://github.com/MMTS4All/GTM), [UniTime](https://github.com/liuxu77/UniTime), [CPiRi](https://github.com/JasonStraka/CPiRi), [TEFN](https://github.com/ztxtech/Time-Evidence-Fusion-Network) and [TGGC](https://github.com/KimMeen/TGGC). Included source files retain their copyright and license notices; snapshot checksums are recorded in `configs/upstream.json`.
