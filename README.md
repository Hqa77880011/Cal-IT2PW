# Cal-IT2PW

Implementation of **Calibrated Interval Type-2 Fuzzy Process Windows under Measurement and Operating Uncertainty**.

## Installation

Use Python 3.11+. Create and activate a virtual environment before installing:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

In PowerShell, replace the activation line with `.venv\Scripts\Activate.ps1`.

## Data and preprocessing

Download [SECOM](https://archive.ics.uci.edu/dataset/179/secom), Tennessee Eastman process records from your simulator, or the PHM 2016 CMP process data under their distribution terms. Aggregate traces into process-setting rows using batch and tool identifiers before fitting. SECOM's binary pass/fail label can be used as binary quality; continuous-quality intervals should be studied on CMP or a simulator. Never invent continuous quality from binary labels.

Create `data/process.csv` with numeric process columns and `quality,batch,regime,split`. Assign entire batches, in chronological order when applicable, to `train`, `interval`, `risk`, `decision`, or `test`. These five roles respectively fit rules, interval coverage, failure risk, decision margin and final evaluation. Missing values must be resolved using training-only statistics before export. Batch leakage and nonfinite values cause errors.

Copy `process.example.json` and replace feature names, sensor repeatability standard deviations in original units, and justified monotonic directions (`1` increasing, `-1` decreasing). The 3×2 decision-cost matrix uses rows feasible/manual-review/infeasible and columns satisfactory/failed quality. Specify physically impossible settings by excluding them during data preparation; no unsupported feasible combinations are synthesized.

## Run and evaluate

```bash
calit2pw data/process.csv process.example.json --minimum-quality 0.8 --output outputs/full --seed 0
```

## Ablations

Supported controls: `fixed`, `anfis`, `no_monotonicity`, `no_calibration`, `gp`, `quantile`. Reuse the same inputs, seed and budget. For retrained controls, evaluate the corresponding saved model.

```bash
calit2pw data/process.csv process.example.json --minimum-quality 0.8 --output outputs/fixed --seed 0 --variant fixed
```

## Outputs

`model.pkl` stores the fitted estimator/calibration. `predictions.csv` contains quality, intervals and decisions; `metrics.json` contains computed scores; `process_window.png` visualizes intervals.

Source inspection: Python 3.13.5, numpy 2.1.3, pandas 2.2.3, scipy 1.15.3, scikit-learn 1.6.1, torch 2.7.1+cu118, matplotlib 3.10.0 (installed versions inspected). Checks covered Python syntax, configuration parsing and CLI arguments. No training, full evaluation or CI was run in this update.
