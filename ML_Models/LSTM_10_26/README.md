# LSTM experiment - 10/26

This folder contains the first sequence-model experiment for predicting vehicle
sensor measurements from AprilTag (whisker) deformation data. The recordings in
this experiment are straight-line trials only. The IMU linear-acceleration
targets still include gravity.

## What is set up here

- `train_rnn.py` trains stateless sequence models from pre-cut CSV windows.
  Supported architectures are RNN, GRU, LSTM, and a causal Transformer using Pytorch's built in library.
- `make_windows.py` cuts recordings into overlapping windows. Its defaults are
  65 rows per window (about one second at 65 fps) and a 16-row step.
- `evaluate.py` evaluates a saved checkpoint and reports RMSE, a training-mean
  baseline, skill relative to that baseline, and correlation. It also writes
  predicted-versus-true scatter plots.
- `tags_to_imu_data/raw_data/` contains the source recordings and
  `tags_to_imu_data/window_data/` is the location for prepared windows and
  splits.
- `runs/` contains nine saved checkpoints and matching training logs, covering
  baseline and input-difference variants across three random seeds.

The model inputs are tag measurements plus elapsed time. By default, the
targets include IMU measurements (including the orientation quaternion) and
depth; the orientation can optionally be represented as roll and pitch instead
of a quaternion. Missing output values are excluded from the loss.

*** Keep all windows from any one recording in the same data split to avoid leakage between
overlapping windows.

## Typical workflow

From this directory, prepare windows from each source CSV, then place the
resulting windows into separate training, validation, and test folders:

```bash
python make_windows.py tags_to_imu_data/raw_data/recording.csv
python train_rnn.py \
  --train_dir tags_to_imu_data/window_data/train \
  --val_dir tags_to_imu_data/window_data/val \
  --test_dir tags_to_imu_data/window_data/test \
  --model lstm \
  --out runs/model.pt
python evaluate.py \
  --checkpoint runs/model.pt \
  --data_dir tags_to_imu_data/window_data/test
```

The folder arguments in the example are expected split directories; organize
the generated windows into those splits before training.

## Changes from the previous train_rnn setup and parsing procedure:

This is the first folder in this experiment sequence, so there is no previous
model folder to compare against. It establishes the straight-line-only baseline
with gravity present in the IMU acceleration targets.
