# LSTM flow experiment - 11/16

This folder tests whether an outside object can be sensed in the flow. The
model is fed the whisker (AprilTag) movements **and** the IMU/gyro as inputs.
The IMU/gyro tell the model how the ROV itself is moving, so it can discount
the part of the tag signal its own motion explains and read the object from
what is left.

It works on `tag_imu_to_flow_class` exports only. Their last five columns are
the labels: `test`, `wall_side`, `dist_to_wall`, `angle_to_head`,
`dist_to_head`. One script runs two tests, chosen with `--test`; each has its
own pair of classifiers, named after the label columns they read:

| `--test` | `test` column says | Head 1 | Head 2 |
|---|---|---|---|
| `wall` | `dist_to_wall` | `wall_side`: which side the wall is on | `dist_to_wall`: distance to the wall |
| `angle` | `angle_to_head` | `angle_to_head`: angle of the flow disturbance | `dist_to_head`: distance of the disturbance from the head |

Only the chosen test's recordings are used, so one data folder can hold both.
Files without a `test` column (for example `tag_to_imu_format` exports, which
belong to `train_rnn.py` in the previous folder) are skipped and listed.

Both heads of the test are trained by default. A head needs at least two
different label values in the training recordings: with only one (for
example, every recording at the same distance) the script warns, leaves that
head out and trains the other.

## What is set up here

- `train_lstm_flow.py` trains the segment classifier (LSTM by default; GRU and
  plain RNN are selectable). It takes one folder of full recordings, splits it
  by whole recording into train / val / test (balanced by label), and only then
  cuts each recording into fixed-length segments.
- `data_check.py` summarizes labels, segments per class, ROV motion per
  class, per-day tag offsets, and tags that drop out or glitch, to check the
  recordings before training.
- `fft_whiskers.py` plots the whisker and gyro spectra per class (left and
  right tags are given with `--left` / `--right`).
- `evaluate.py` scores a saved model per segment and per recording.
  `plot_predictions.py` writes a per-recording time plot of the class
  probabilities.
- `data/` holds the `tag_imu_to_flow_class` recordings of both tests;
  `models_wall/` and `models_angle/` hold their checkpoints, split manifests,
  learning curves and evaluation plots.

### No sliding window

There is no running window, warmup or stride any more. After the split, each
recording is cut into consecutive, non-overlapping segments of `--seg_len`
frames (default 16, about 0.25 s at 65 fps): rows 0-15, 16-31, 32-47, and so
on. Each segment is one training sample. The model reads it from a zero state
and gives one prediction per head for the whole segment; nothing is carried
from one segment to the next. Leftover rows at the end of a recording that
don't fill a segment are dropped, and recordings shorter than one segment are
skipped. Both are reported.

### Inputs and outputs

| | Columns |
|---|---|
| Inputs: `tags` | every `derived/tf[tag_*...]` column: translation, rotation and the `seen` flag per tag |
| Inputs: `imu` | gravity-removed linear acceleration x/y/z and angular velocity x/y/z |
| Outputs | the test's two heads (table above); classes = the label values found in the training recordings |

Nothing else is fed in: the label columns, `t_s`, `dt_s`, the depth column and
`derived/turn.turn_rate_avg` are left out. Other columns can be added with
`--extra_inputs`, and tags removed with `--drop_tags`.

Labels may be numbers (`0`, `45`, `0.3`) or text (`left`, `right`). A blank
cell means "no label", and the segment is left out of that head's loss. A
no-wall run therefore has to say so in words (for example `wall_side` =
`none`) to count as a class.

Tags that are not detected hold their last observed value, with the `seen`
indicator provided to the model. This is done on the whole recording before it
is cut.

## Typical workflow

Wall test:

```bash
python data_check.py data --test wall --seg_len 16
python train_lstm_flow.py --test wall --data_dir data --seg_len 16 --out models_wall/wall_16.pt
python evaluate.py --checkpoint models_wall/wall_16.pt --plots
python plot_predictions.py models_wall/wall_16.pt --split test
```

Angle test:

```bash
python data_check.py data --test angle --seg_len 16
python train_lstm_flow.py --test angle --data_dir data --seg_len 16 --out models_angle/angle_16.pt
python evaluate.py --checkpoint models_angle/angle_16.pt --plots
```

To train one head only, name it: `--target wall_side`, `--target
angle_to_head`, and so on. `--balance_classes` weights rare classes up.

The split is saved next to the checkpoint as `<out>_split.json`. Reuse it with
`--split_file` when comparing configurations so they see the same recordings:

```bash
# a different segment length on the same split
python train_lstm_flow.py --split_file models_wall/wall_16_split.json --seg_len 32 \
  --out models_wall/wall_32.pt
```

### Is it sensing the object, or the ROV's own motion?

Train the same split three ways:

```bash
python train_lstm_flow.py --split_file models_wall/wall_16_split.json --inputs tags imu --out models_wall/both.pt
python train_lstm_flow.py --split_file models_wall/wall_16_split.json --inputs tags     --out models_wall/tags.pt
python train_lstm_flow.py --split_file models_wall/wall_16_split.json --inputs imu      --out models_wall/imu.pt
```

- `imu` alone should score near its baseline. If it doesn't, the ROV moved
  differently for different classes, and the model can tell the classes apart
  without sensing anything in the flow. `data_check.py` section 2 shows the
  same thing from the data side.
- `tags imu` beating `tags` is the evidence that the IMU/gyro help separate
  the ROV's own motion from the object's effect on the whiskers.

### Reading the scores

Scores are counted per segment. `evaluate.py` also prints, per recording, the
share of its segments that were right and the class the recording gets when
its segments vote. `--vote N` scores each segment from the probabilities
averaged over the last N segments, which shows how much a short running
average would steady the output. `--trim_start` / `--trim_end` leave out the
first / last seconds of every recording.

### Checks for shortcuts

A very high score on a small data set usually means a shortcut. These runs, on
the same split, show which one:

| Run | Reads | Should score near chance if... |
|---|---|---|
| `--inputs imu` | only the IMU/gyro | the ROV moved the same way in every class |
| `--inputs seen` | only which tags were detected | the classes have the same tag dropouts |
| `--inputs tags --center_recordings` | tag movement, with each recording's own mean pose removed | the classes differ only by a constant tag offset |

Tags that are missing in some classes (and so give the class away through their
`seen` flag) should be removed with `--drop_tags`. `data_check.py` section 4
lists them. `fft_whiskers.py` shows the frequency content of the left and
right whiskers per class.

### Keeping tags that sometimes drop out: `--tag_dropout`

Instead of removing every tag that misses sometimes, `--tag_dropout 0.3` makes
each tag look missed (held pose, `seen` = 0) for a whole segment with that
probability, during training only. The model then can't rely on any one tag or
on which tags happened to drop out in a recording. It lowers that shortcut but
doesn't remove it: a tag that misses far more in one class than another still
differs between them. Tags that are never seen in a class (tag 13 here) still
have to go with `--drop_tags`.

### Combining days: `--day_baseline`

Tag poses are measured against a zero pose that can differ between days, so a
second day can't simply be added: if the days hold different classes, the
model learns the day. Per-day normalisation can't fix that either, because
subtracting a day's mean removes the difference between classes along with it.

`--day_baseline` normalises each day against its own **no-object** recordings:
for every day it takes the mean tag pose of that day's training recordings
labelled `none` (`--baseline_label`) and subtracts it from every recording of
that day. A constant offset that differs between days cancels; a steady
deflection caused by the object stays. It needs at least one `none` recording
per day in the training set, otherwise the script stops and says which day
lacks one. Days are read from the file name (`MM_DD_YYYY` or `MM_DD_YY`, or
your own `--day` regex).

```bash
python train_lstm_flow.py --test angle --data_dir data --day_baseline \
  --drop_tags 0,13 --in_dropout 0.2 --dropout 0.2 --patience 15 \
  --out models_angle/angle_daybase.pt
```

It can't be combined with `--center_recordings`. A recording from a day that
wasn't in training can only be scored if its day has a baseline in the model.

## Changes from the previous model folder

Compared with [LSTM_11_01](../LSTM_11_01/):

- `train_rnn.py` is replaced by `train_lstm_flow.py`.
- The sliding window (`--window`, `--warmup`, `--stride`, `--eval_stride`) is
  gone. Recordings are split first, then cut into non-overlapping `--seg_len`
  segments, with one prediction per segment instead of one per frame.
- The gravity-removed IMU/gyro moved from regression target to input, next to
  the tags. There are no regression outputs.
- The data is the new `tag_imu_to_flow_class` export, which adds the `test`,
  `wall_side`, `dist_to_wall`, `angle_to_head` and `dist_to_head` label
  columns in place of `action` and `approx_diam`.
- The outputs are two classification heads per test (`wall_side` +
  `dist_to_wall`, or `angle_to_head` + `dist_to_head`), replacing `imu`,
  `turn_rate_avg`, `action` and `diam`. The classes come from the label
  values in the training recordings.
- Inputs are an explicit list (tags, IMU, `--extra_inputs`) rather than
  "every column except the targets".
- Splits are always made from one folder (`--data_dir`) or a saved
  `--split_file`; the separate `--train_dir` / `--val_dir` / `--test_dir`
  option is removed.
- `evaluate.py`, `plot_predictions.py` and `data_check.py` work per segment
  and report the test's two heads. Checkpoints from `train_rnn.py` do not load in
  the new scripts.
