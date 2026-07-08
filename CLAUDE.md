# biohub-cell-tracking

Kaggle competition project built on the [kitchen platform](https://github.com/rkoren/kitchen-platform).

## Setup

```bash
pip install rkoren-kitchen -e .
# Contributors working from the monorepo: pip install -e ../kitchen-platform/kitchen -e .
cp .env.example .env
# Download competition data to data/raw/
```

## The contract — 3 files to implement

| File | Class | Method |
|---|---|---|
| `src/features/run.py` | `BiohubCellTrackingFeatures(FeatureBuilder)` | `build(raw_or_sources, params) -> df` |
| `src/train/run.py` | `BiohubCellTrackingTrainer(Trainer)` | `fit(df, params) -> model` |
| `src/evaluate/run.py` | `BiohubCellTrackingEvaluator(Evaluator)` | `evaluate(model, df) -> dict` |

All config lives in `menu.yaml`. File paths resolve from `params["features"].*`;
model hyperparams from `params["model"].*`.

## Running experiments

```bash
kitchen run train                   # features → train → log to MLflow
kitchen run evaluate                # load champion model, compute metrics
kitchen leaderboard                 # rank all runs by primary metric
kitchen promote METRIC              # promote best run to the registry
kitchen ui                          # open MLflow UI in browser

# Experiment variants (edit first, then run)
python experiments/baseline.py
python experiments/challenger.py

# Generate Kaggle submission
kitchen submit
```

## Kitchen modules

- `kitchen.steps` — `FeatureBuilder`, `Trainer` (set `model_flavour`), `Evaluator` ABCs
- `kitchen.tracking` — `Tracker`, `configure_from_env()`, `init_experiment()`
- `kitchen.store` — `DataStore` (wraps `data/raw/`, `data/processed/`, `models/`)
- `kitchen.modeling` — `train_val_split`, `classification_metrics`, `regression_metrics`

## Experiment tagging

Both experiment scripts tag runs with `model_variant=baseline` or `model_variant=challenger`.
`kitchen promote METRIC` promotes the best run to the `champion` alias.
Load the champion with `mlflow.pyfunc.load_model('models:/biohub-cell-tracking-model@champion')`.
