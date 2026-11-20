# Shortcut checks for the angle test

One folder that runs the same checks on any set of `tag_imu_to_flow_class` angle-test recordings.

## Setup (once)

    cd ~/Sealbot2.0/ML_Models
    unzip ~/Downloads/LSTM_flow_checks.zip
    cd LSTM_flow_checks
    cp -r ../LSTM_11_18/data_11_16_flow data_11_17      # the 11/17 recordings
    cp ../LSTM_11_18/plot_predictions.py .              # only needed for evaluate.py --plots

## Run

    source ~/bagsync-venv/bin/activate
    ./run_checks.sh data_11_17          # 0.25 s windows
    ./run_checks.sh data_11_17 32       # 0.5 s windows (optional)

## What it does

1. `offset_check.py`: does each tag's resting position, and its jitter, change with the angle?
2. `transform_tags.py`: three copies of the data, all with the seen flags removed:
   - `diff`: each tag's change since the previous frame
   - `cmdiff`: that change minus the average change of the whole array in that frame
   - `cmonly`: only that array average
3. Trains and evaluates three models (tags only, tags 0, 7, 20 dropped):
   - `diff_whiskers`: all whisker motion
   - `cmdiff`: how whiskers move relative to each other
   - `cmonly`: motion the whole array shares
4. `summary.txt`: the scores side by side.

## Reading summary.txt (6 classes: guessing ~17%)

| Result | Meaning |
|---|---|
| `cmdiff` well above 17% | Whiskers move differently from each other with the angle: best evidence of flow sensing |
| `cmonly` high, `cmdiff` near 17% | Signal is in motion the whole array shares: camera/rig or whole head, can't tell apart |
| Resting-position F large on most whisker tags, all shifting together | The rig shifted between angle conditions |

Edit `DROP_TAGS` / `EXCLUDE` at the top of `run_checks.sh` for other data sets.

## Window-length sweep

    ./run_windows.sh data_11_17 4 8 16 32

Runs `run_checks.sh` for each window length (skipping any already done), then `plot_windows.py`, which
writes `windows_data_11_17_test.png` / `.csv` (test recordings) and `windows_data_11_17_val+test.png` /
`.csv` (val and test pooled): each class's recall and the balanced accuracy against the window length,
one panel per model. Val picked each model's best epoch, so the test plot is the fair one.
