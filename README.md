# Cal-IT2PW

Implementation of **Calibrated Interval Type-2 Fuzzy Process Windows under Measurement and Operating Uncertainty**.

The method learns Gaussian interval antecedents with separate measurement and batch footprints, local linear consequents and consequent uncertainty. Iterative Karnik–Mendel endpoint reduction produces raw intervals. Batch-disjoint calibration then expands intervals, estimates failure probability, selects a decision margin and applies a release-risk gate.

## Install

```bash
python -m venv .venv
python -m pip install -e .
```

Python 3.11 or later is required. Activate the environment with the command appropriate to your shell.

## Prepare data

Download [SECOM](https://archive.ics.uci.edu/dataset/179/secom), Tennessee Eastman process records from your simulator, or the PHM 2016 CMP process data under their distribution terms. Aggregate traces into process-setting rows using batch and tool identifiers before fitting. SECOM's binary pass/fail label can be used as binary quality; continuous-quality intervals should be studied on CMP or a simulator. Never invent continuous quality from binary labels.

Create `data/process.csv` with numeric process columns and `quality,batch,regime,split`. Assign entire batches, in chronological order when applicable, to `train`, `interval`, `risk`, `decision`, or `test`. These five roles respectively fit rules, interval coverage, failure risk, decision margin and final evaluation. Missing values must be resolved using training-only statistics before export. Batch leakage and nonfinite values cause errors.

Copy `process.example.json` and replace feature names, sensor repeatability standard deviations in original units, and justified monotonic directions (`1` increasing, `-1` decreasing). The 3×2 decision-cost matrix uses rows feasible/manual-review/infeasible and columns satisfactory/failed quality. Specify physically impossible settings by excluding them during data preparation; no unsupported feasible combinations are synthesized.

## Train, calibrate and evaluate

```bash
calit2pw data/process.csv process.example.json --minimum-quality 0.8 --output outputs/full
```

`--minimum-quality` defines failure as quality below the threshold. `--alpha 0.1` requests 90% intervals; `--risk 0.05` gates release at estimated failure probability 5%. `--rules`, `--epochs`, `--batch-size` and `--lr` control the fuzzy estimator. `--minimum-regime 50` requires enough calibration rows before using a regime-specific calibrator, otherwise the global calibrator applies. `--seed` controls initialization.

Run comparable controls on the same CSV:

```bash
calit2pw data/process.csv process.example.json --minimum-quality 0.8 --variant anfis --output outputs/anfis
calit2pw data/process.csv process.example.json --minimum-quality 0.8 --variant fixed --output outputs/fixed
calit2pw data/process.csv process.example.json --minimum-quality 0.8 --variant gp --output outputs/gp
calit2pw data/process.csv process.example.json --minimum-quality 0.8 --variant quantile --output outputs/quantile
calit2pw data/process.csv process.example.json --minimum-quality 0.8 --variant no_monotonicity --output outputs/no_monotonicity
calit2pw data/process.csv process.example.json --minimum-quality 0.8 --variant no_calibration --output outputs/no_calibration
```

ANFIS is a fixed Gaussian normalized Takagi–Sugeno control, `fixed` retains interval antecedents with a constant footprint, GP uses a learned RBF-plus-noise kernel, and quantile regression uses linear 5th/95th percentile models. The GP has cubic training cost. All controls receive the same independent coverage/risk/decision calibration.

## Outputs

`metrics.json` reports empirical coverage, mean interval width, false acceptance among released settings, rejection among satisfactory settings, 10-bin ECE and Brier score for failure probability, expected decision cost, release fraction, rule count, finite-difference monotonicity violations and inference milliseconds per setting. A rate with an empty denominator is `null`. `predictions.csv` records measured quality, interval, failure probability and decision (`0` feasible, `1` review, `2` infeasible). `process_window.png` plots intervals against quality; `model.pkl` stores the trained estimator and calibration state. Load pickle files only from trusted runs.

For five batch folds and seeds, construct five CSV split assignments and rerun the same commands. Run `python rule_stability.py outputs/full0/model.pkl outputs/full1/model.pkl` to compute optimally matched rule-center stability and normalized distance; intervals and acceptance must always be evaluated on untouched test batches.

## Implementation choices

Rule count is selected explicitly rather than by undocumented growth/merge heuristics. Batch footprint uses dispersion of batch feature means; replicated sensor uncertainty is supplied separately. Monotonicity uses a soft finite-difference penalty because mixing monotone local consequents does not ensure a monotone global fuzzy surface. Type reduction uses the converged endpoint switch iteration. Risk calibration is monotone isotonic regression of the calibrated lower-bound margin, separately per sufficiently supported regime; interval width enters the interval estimator rather than an unconstrained probability model. These choices keep fuzzy membership, interval coverage and event probability distinct. Nominal marginal coverage does not by itself establish conditional release safety.
