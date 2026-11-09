# LSTM experiment - 10/28

This folder extends the tag-to-sensor sequence-prediction experiment to include
both straight-line and turning trials. The IMU targets include gravity.

## What is set up here

- `train_rnn.py` trains stateless RNN, GRU, or LSTM models from pre-cut CSV
  windows. It uses validation data to select the best epoch and can score a
  held-out test set after training.
- `make_windows.py` cuts exported recordings into overlapping windows. The
  default is 130 rows (about two seconds at 65 fps), with a 16-row step.
- `evaluate.py` reports per-output RMSE, a training-mean baseline, skill, and
  correlation, and saves predicted-versus-true scatter plots.
- `tags_to_imu_data/raw_data/` holds source recordings; prepared windows and
  splits belong under `tags_to_imu_data/window_data/`.
- The top level contains trained checkpoints and their RMSE-curve plots; the
  available plots also include validation/test scatter plots for a run.

Inputs are tag measurements and other eligible export columns, excluding time
columns, target columns, and the target sensor topics to prevent target leakage.
The regression targets are three linear-acceleration axes, three angular-
velocity axes, and depth. Missing output values are masked out of the loss.

*** Keep all windows from any one recording in the same data split to avoid leakage between
overlapping windows.

## Typical workflow

Create windows for each recording and sort them into recording-level
train/validation/test splits before training:

```bash
python make_windows.py tags_to_imu_data/raw_data/recording.csv
python train_rnn.py \
  --train_dir tags_to_imu_data/window_data/train \
  --val_dir tags_to_imu_data/window_data/val \
  --test_dir tags_to_imu_data/window_data/test \
  --model lstm \
  --out lstm_model.pt
python evaluate.py \
  --checkpoint lstm_model.pt \
  --data_dir tags_to_imu_data/window_data/test
```

The folder arguments in the example are expected split directories; organize
the generated windows into those splits before training.

## Changes from the previous model folder

Compared with [LSTM_10_26](../LSTM_10_26/), this experiment adds turning trials
alongside straight-line trials, allowing the regression model to be trained
and evaluated across both motion types. The IMU acceleration targets still
include gravity. The default window grows from 65 rows (about one second) to
130 rows (about two seconds). 

The option to train the model using Pytorch's transformer module has been removed. Models in runs were 
trained only using an LSTM rather than an RNN or GRU, as the LSTM was found to learn the best during 10_26 
experiments. 
