# WT-Transformer Fault Prediction (Gearbox Failure Detection)

This repository implements gearbox failure detection for wind turbines using temperature forecasting with a Transformer-based encoder. The approach forecasts temperature time series and detects anomalies or precursor patterns that indicate imminent gearbox failures.

This README describes the model and the full reproducible pipeline: environment setup, data preprocessing, training, evaluation, and inference. It includes recommended configuration and troubleshooting tips.

---

## Table of contents

- Project overview
- Architecture and approach
- Repository layout (assumed)
- Data: sources & required format
- Reproducible steps
  - Environment setup
  - Data preparation
  - Training the forecasting model
  - Evaluation and failure detection
  - Inference / deployment
- Hyperparameters & typical configs
- Reproducibility & debugging
- Expected outputs
- Contact & license

---

## Project overview

Goal: predict gearbox faults by forecasting temperature-related sensor series and using deviations (residuals) and/or a downstream classifier to flag anomalous behavior preceding failures.

High-level pipeline:
1. Prepare historical SCADA/time-series sensor data.
2. Train a Transformer encoder-based forecasting model on sliding windows.
3. Use the forecasted temperature vs actual to compute residuals or derived features.
4. Optionally train a classifier or threshold residuals to detect failure events.
5. Evaluate using forecasting metrics (MSE/MAE) and detection metrics (precision/recall/F1, lead time to failure).

---

## Architecture and approach

- Model: Transformer encoder-style model adapted for time-series forecasting (positional encoding, multi-head self-attention, feed-forward blocks). Can use either a simple encoder or encoder-decoder depending on configuration.
- Inputs: multivariate time-series windows (e.g., temperatures, rotational speed, ambient temperature).
- Output: forecasted horizon for target temperature sensor(s).
- Detection: residual-based thresholding or a supervised classifier trained on forecast error patterns and labeled failure windows.

Notes:
- Use normalization per sensor (e.g., standard scaling) using training set statistics.
- Use sliding windows for both training and inference; ensure no data leakage across time (use strict train/val/test splits by time).

---

## Repository layout

This README assumes the repository follows a typical structure. If your repo differs, replace paths in reproduction steps with the actual paths.

- README.md
- requirements.txt
- src/
  - src/train.py
  - src/evaluate.py
  - src/infer.py
  - src/models/transformer.py
  - src/data/preprocess.py
  - src/utils.py
- configs/
  - configs/forecasting.yaml
- data/
  - data/raw/
  - data/processed/
- models/
  - models/checkpoints/
- results/
  - results/metrics/
  - results/figures/

---

## Data: sources & required format

Expected input:
- CSV or Parquet time-series file(s) with a timestamp column and sensor columns.

---

## Reproducible steps

1) Clone the repo
```bash
git clone https://github.com/bryanjoao/wt-transformer-fault-prediction.git
cd wt-transformer-fault-prediction
git checkout 08ac588f1b4e87fe0d7557e1979a387250c3ee29
```

2) Create environment & install dependencies
- Using conda (recommended):
```bash
conda create -n wt-transformer python=3.9 -y
conda activate wt-transformer
pip install -r requirements.txt
```
- Or using virtualenv:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3) Prepare data
- Place raw files in `data/raw/`.
- Run preprocessing to produce normalized windows:
```bash
python src/data/preprocess.py \
  --input-dir data/raw/ \
  --output-dir data/processed/ \
  --freq 1H \
  --target-column t_gearbox \
  --window-size 168 \
  --horizon 24
```
Notes:
- `--freq` is the resampling frequency (example: 1H for hourly).
- `--window-size` and `--horizon` are example values (one week in hours vs 24-hour forecast).

4) Train forecasting model
```bash
python src/train.py \
  --config configs/forecasting.yaml \
  --data-dir data/processed/ \
  --save-dir models/checkpoints/ \
  --device cuda:0
```
Key config options:
- model: transformer
- input_window: 168
- forecast_horizon: 24
- batch_size: 64
- lr: 1e-4
- epochs: 100
- seed: 42

5) Evaluate forecasting and detection
Forecast evaluation:
```bash
python src/evaluate.py \
  --checkpoint models/checkpoints/best.ckpt \
  --data-dir data/processed/ \
  --metrics-out results/metrics/forecast_metrics.json
```

Failure detection (residual thresholding example):
```bash
python src/infer.py \
  --checkpoint models/checkpoints/best.ckpt \
  --data-dir data/processed/ \
  --out results/inference/ \
  --detection-method residual_threshold \
  --threshold 3.0
```
- Optionally train a detection classifier using residuals as features:
```bash
python src/train_detector.py \
  --residuals results/inference/residuals.csv \
  --labels data/processed/labels.csv \
  --save results/detector/model.pkl
```

6) Visualize results
Generate plots for forecast vs actual and residuals:
```bash
python scripts/plot_results.py --input results/inference/predictions.csv --out results/figures/
```

---

## Hyperparameters & typical config

Example (configs/forecasting.yaml)
```yaml
model:
  type: transformer
  d_model: 128
  n_heads: 8
  n_layers: 4
  d_ff: 512
train:
  batch_size: 64
  lr: 1e-4
  weight_decay: 1e-5
  epochs: 100
  scheduler: cosine
data:
  input_window: 168
  horizon: 24
  features: ["t_gearbox","rpm","ambient_temp"]
seed: 42
```

Tuning tips:
- Increase model d_model and n_layers for larger datasets.
- Use learning rate warmup for Transformer training.
- Use early stopping on validation MSE to avoid overfitting.

---

## Expected outputs

- `models/checkpoints/best.ckpt` — trained forecasting model
- `results/metrics/forecast_metrics.json` — forecasting MSE/MAE/RMSE
- `results/inference/predictions.csv` — timestamps, actuals, forecasts, residuals
- `results/detector/predictions.csv` — detection labels & scores
- `results/figures/` — visualizations (forecasts, residuals, detection timelines)

Evaluation metrics:
- Forecasting: MSE, MAE, RMSE
- Detection: precision, recall, F1, lead time to failure (average time between detection and actual failure event)

---

## Troubleshooting

- "CUDA out of memory": reduce batch size or model size; enable gradient accumulation.
- Poor forecast accuracy: verify input features, check normalization, ensure no leakage across train/val/test.
- Imbalanced failure labels: use class weighting or resampling for detector training.



