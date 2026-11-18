# LSTM experiment - 11/01

This folder sets up tag-to-sensor learning on full `tag_to_imu_format` CSV
recordings. It covers IMU regression with gravity removed from the IMU fields
and supports turning/manoeuvre classification from labels stored in the
recordings.

## What is set up here

- `train_rnn.py` trains stateless RNN, GRU, or LSTM models. It can split
  recordings automatically, or accept explicit train/validation/test folders.
  When it splits automatically, it makes a label-balanced split by whole
  recording before generating overlapping windows.
- The default regression target (`--target imu`) is the three
  gravity-removed linear-acceleration axes, three angular-velocity axes, and
  depth. Additional selectable targets include the measured average turn rate,
  action classes (`straight`, `right`, `left`), and approximate circle-diameter
  classes (`diam`). A run can combine multiple targets. Turn tests and wall tests
  go in separate folders (see below).
- `data_check.py` summarizes labels, turn rates, tag visibility, and straight-
  run tag offsets to help check the input recordings.
- `evaluate.py` evaluates saved models. `plot_predictions.py` writes
  per-recording time-series plots for regression predictions and classification
  probabilities.
- `data/` contains full recording CSVs in `tag_to_imu_format`; `removed/`
  keeps a recording separate from the active data set.
- `models_imu/` and `models_turn/` contain model checkpoints or run outputs,
  split manifests, learning curves, and evaluation plots for the respective
  experiment groups.

Windows are generated on the fly from full recordings, with a default 32-row
buffer and 16-row warmup at 65 fps. The warmup rows provide history but are not
scored. By default, `t_s` and `dt_s` are excluded from the inputs; target-topic
columns and label columns are excluded as well to prevent leakage. Tags that
are not detected hold their last observed value, with a `seen` indicator
provided to the model.

### Export and timestamp setup

The parser/export setup for these trials adds the manoeuvre labels to the
`tags_to_imu` export type. Autopilot timestamp correction is selected: during
post-processing, the FCU/MAVLink sensor timestamp is used to restamp sensor
data instead of the MAVROS reception/delivery-dump timestamp. This improves
sensor-message spacing and removes the need to interpolate the sensor timeline.
* Future work is focused on trying to correct the sensor timestamp in realtime, but will continue to work in post-processing for data analysis and predictive model validation.

## Typical workflow

From this directory, inspect the recordings, then train using the full CSVs.
The following illustrates the IMU regression plus action/diameter
classification configuration:

```bash
python data_check.py data
python train_rnn.py \
  --data_dir data \
  --target imu action diam \
  --model lstm \
  --out models_turn/example.pt
python plot_predictions.py models_turn/example.pt --split test
```

### Turn tests and wall tests: keep them apart

The two tests use `action` differently, so **they can't be trained together**.
Keep them in separate folders; `train_rnn.py` stops with an error if one data
set holds both.

| Test | Last column (bagsync export) | `action` left / right means | Targets |
|---|---|---|---|
| Turn test | `approx_diam` (circle diameter, m; `-1` = straight) | turning left / right | `imu`, `turn_rate_avg`, `action`, `diam` |
| Wall test | `dist_to_wall` (distance to the wall, m) | wall on the left / right, driving straight | `imu`, `turn_rate_avg`, `action` |

`dist_to_wall` is only a label in the CSV: it is never used as an input, and it
isn't a training target. Wall-test example:

```bash
python train_rnn.py --data_dir data_wall --target action --model lstm --balance_classes \
  --out models_wall/lstm_action_wall.pt
```

`wall_side_check.py data_wall` checks whether the tags sit differently with the
wall on the left vs on the right, before training a wall-side classifier.

The split is saved next to the checkpoint as a JSON manifest. Reuse it with
`--split_file` when comparing model configurations so they use the same
recording split.

## Changes from the previous model folder

Compared with [LSTM_10_28](../LSTM_10_28/), this version changes the data
workflow from pre-cut window CSVs to full recordings, splitting by recording
before windowing. The IMU target is now gravity-removed. It adds label-based
turn/manoeuvre classification alongside regression, and adds labels to the
`tags_to_imu` export. Autopilot/FCU-MAVLink sensor timestamps are used during
post-processing instead of MAVROS reception timestamps, improving sample
spacing and avoiding interpolation.