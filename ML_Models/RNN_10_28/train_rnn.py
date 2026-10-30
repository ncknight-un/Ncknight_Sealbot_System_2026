"""
Stateless sequence model: AprilTag deformations (+ time) -> IMU + depth (PyTorch).

Data = three folders of window CSVs (train / validation / test), one example per
file, as written by make_windows.py from tag_to_imu_format exports (130 rows =
2 s at 65 fps by default; the window length is read from the first training file).
  - train: used to fit the model
  - val:   checked every epoch; the epoch with the lowest val RMSE is saved
  - test:  never touched during training; scored once at the end

Columns used:
  - Inputs  = every "derived/tf[tag_*...]" column (per tag: translation_mm.x/y/z,
              rotation.x/y/z/w and the seen flag) and every other column in the files, EXCEPT the
              outputs, anything else from the output topics, and IGNORE_INPUTS (t_s, dt_s)
  - Outputs = mavros/imu/data.linear_acceleration.x/y/z (m/s², gravity included),
              mavros/imu/data.angular_velocity.x/y/z (rad/s), and depth -> 7
              (depth = mavros/global_position/local.pose.pose.position.z, metres, negative underwater)
Empty output cells are left out of the loss and RMSE.

Missed tag detections: each tag's <tag>.seen column is 1 when the tag was detected
in that row and 0 when it wasn't. A missed tag's values are NOT filled with the
mean: they hold the tag's last detected value in the window (only earlier rows are
used, so it works live), or its zero pose (0 mm, identity rotation) if it hasn't
been seen yet in the window. The seen flag tells the model which rows are real.

Windows cut from the same recording overlap, so each recording's windows
should live in only one folder. The script warns if a recording name (the part
of the filename before _wNNNN_rows...) shows up in more than one folder.

At the end, the train/val RMSE per epoch is saved next to the --out checkpoint
as <out>_rmse_curve.png (needs matplotlib; skipped if it isn't installed).
For per-output scores and predicted-vs-true scatter plots, run evaluate.py.

A progress bar shows while loading and during each epoch (tqdm if installed,
otherwise a simple built-in bar).

Models (--model):
  rnn          plain Elman RNN
  gru          gated recurrent unit
  lstm         simple LSTM

"Stateless" = nothing is carried between windows; every window starts fresh.

Usage:
    python train_rnn.py --train_dir data/train --val_dir data/val --test_dir data/test --model gru
"""

import argparse
import glob
import os
import random
import re
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEQ_LEN = 130                # default only; set from the first training file at startup
# Inputs = every column in the window files EXCEPT the outputs, anything else from the output
# topics (OUTPUT_TOPICS: e.g. IMU orientation would give the answer away), and these. Add names
# here or with --ignore_inputs. t_s (time since the export started) would let the model learn
# "15 s into a run usually means deep" instead of reading the tags; dt_s is frame timing.
IGNORE_INPUTS = ["t_s", "dt_s"]

_I = "mavros/imu/data."
OUTPUT_COLS = ([_I + f"linear_acceleration.{a}" for a in "xyz"]
               + [_I + f"angular_velocity.{a}" for a in "xyz"]
               + ["mavros/global_position/local.pose.pose.position.z"])   # depth
# Topics the outputs come from: none of their columns may be inputs (they'd give the answer away).
OUTPUT_TOPICS = ("mavros/imu/data.", "mavros/global_position/local.")

# Window filenames look like <recording>_w0000_rows0-129.csv; this pulls out <recording>.
WINDOW_NAME = re.compile(r"^(?P<rec>.+)_w\d+_rows\d+-\d+\.csv$")


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm as _tqdm               # nicer bar if tqdm is installed
except ImportError:
    _tqdm = None


def progress(iterable, desc="", leave=False):
    """Wrap a loop in a progress bar: tqdm if installed, otherwise a simple text bar.

    leave=False erases the bar when the loop ends, so only the summary lines
    (e.g. the per-epoch RMSE) stay on screen.
    """
    if _tqdm is not None:
        return _tqdm(iterable, desc=desc, leave=leave, dynamic_ncols=True)
    return _SimpleBar(iterable, desc, leave)


class _SimpleBar:
    """Minimal fallback bar: [#####.....] 12/40. Silent when output isn't a terminal
    (e.g. piped to tee), so log files don't fill up with bar redraws."""

    def __init__(self, iterable, desc, leave):
        self.it, self.desc, self.leave = iterable, desc, leave
        try:
            self.total = len(iterable)
        except TypeError:
            self.total = None

    def __iter__(self):
        if not self.total or not sys.stdout.isatty():
            yield from self.it
            return
        step = max(1, self.total // 50)            # redraw about 50 times per loop
        for i, x in enumerate(self.it, 1):
            yield x
            if i % step == 0 or i == self.total:
                done = int(30 * i / self.total)
                sys.stdout.write(f"\r{self.desc} [{'#' * done}{'.' * (30 - done)}] {i}/{self.total}")
                sys.stdout.flush()
        # Either move to a new line (keep the bar) or blank the line (erase it).
        sys.stdout.write("\n" if self.leave else "\r" + " " * (len(self.desc) + 50) + "\r")
        sys.stdout.flush()


def is_input_col(c):
    """Every column is an input except the outputs, anything else from the output topics, and
    IGNORE_INPUTS."""
    return (c not in OUTPUT_COLS and not c.startswith(OUTPUT_TOPICS) and c not in IGNORE_INPUTS
            and not c.startswith("Unnamed:"))       # (a stray pandas index column)


def short_name(c):
    """'mavros/imu/data.angular_velocity.z' -> 'angular_velocity.z',
    'mavros/global_position/local.pose.pose.position.z' -> 'global_position/local.pose.pose.position.z'."""
    return c.replace("mavros/imu/data.", "").replace("mavros/", "")


def is_output_col(c):
    """Acceleration, gyro and depth (OUTPUT_COLS) are what the model predicts."""
    return c in OUTPUT_COLS


def is_seen_col(c):
    return c.startswith("derived/tf[") and c.endswith(".seen")


def tag_of(c):
    """'derived/tf[tag_3_zero->tag_3].translation_mm.x' -> 'derived/tf[tag_3_zero->tag_3]'."""
    return c[: c.index("]") + 1] if c.startswith("derived/tf[") else None


def fill_tag_gaps(X, in_cols):
    """Fill missed tag detections in raw units, in place. X: (N, T, n_in) with NaN gaps.

    A missed tag holds its last detected value in the window (only earlier rows are
    used, so this works live). Before its first detection in the window it gets its
    zero pose: 0 mm and the identity rotation (0, 0, 0, 1). The tag's seen column
    (1/0) tells the model which rows are real, so these values are placeholders.
    """
    N, T, _ = X.shape
    for j, c in enumerate(in_cols):
        if not c.startswith("derived/tf[") or is_seen_col(c):
            continue
        v = X[:, :, j]
        ok = ~np.isnan(v)
        idx = np.where(ok, np.arange(T)[None, :], 0)
        np.maximum.accumulate(idx, axis=1, out=idx)        # row of the last detection so far
        held = np.take_along_axis(v, idx, axis=1)
        held[~np.maximum.accumulate(ok, axis=1)] = 1.0 if c.endswith("rotation.w") else 0.0
        X[:, :, j] = held
    return X



# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_csv(path, in_cols=None, out_cols=None):
    """Read one window file.

    Returns X (SEQ_LEN, n_in), Y (SEQ_LEN, n_out), and the column lists. Both keep
    NaN where a value is missing: missed tags are filled later by fill_tag_gaps (so
    the normalization uses real readings only), empty outputs are masked out of
    the loss. Without in_cols the first file decides the column lists.
    A tag that is missing from this file entirely (never detected in that recording, so the
    export left it out) counts as never seen: its columns are empty and its seen flag is 0.
    Any other missing column (an output, or t_s) means the file can't be used.
    """
    df = pd.read_csv(path)
    if in_cols is None:
        in_cols = [c for c in df.columns if is_input_col(c)]
        out_cols = [c for c in df.columns if is_output_col(c)]
    if len(df) != SEQ_LEN:
        raise ValueError(f"{os.path.basename(path)}: expected {SEQ_LEN} rows, got {len(df)}")
    absent = [c for c in in_cols if c not in df.columns]
    missing = [c for c in absent if not c.startswith("derived/tf[")] + [c for c in out_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(path)}: missing columns, e.g. {missing[:3]}")
    if absent:                                      # whole tags absent from this recording
        df = pd.concat([df, pd.DataFrame({c: (0.0 if is_seen_col(c) else np.nan) for c in absent},
                                         index=df.index)], axis=1)

    # Convert to numbers; anything unreadable or infinite becomes NaN.
    X = df[in_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    Y = df[out_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return X.to_numpy(np.float32), Y.to_numpy(np.float32), in_cols, out_cols


def all_input_cols(files):
    """Input columns found in any of the files (see is_input_col): tags in tag-number order (each
    tag's columns in the export's order), then the other inputs in the order first seen. Reports
    tags that are missing from some recordings."""
    per_tag, extra, where = {}, [], {}
    for f in progress(files, desc="reading headers"):
        cols = pd.read_csv(f, nrows=0).columns
        tags_here = set()
        for c in cols:
            if c.startswith("derived/tf["):
                t = tag_of(c)
                tags_here.add(t)
                lst = per_tag.setdefault(t, [])
                if c not in lst and is_input_col(c):
                    lst.append(c)
            elif is_input_col(c) and c not in extra:
                extra.append(c)
        m = WINDOW_NAME.match(os.path.basename(f))
        rec = m["rec"] if m else os.path.basename(f)
        where.setdefault(rec, set()).update(tags_here)
    num = lambda t: int(re.search(r"(\d+)\]$", t).group(1)) if re.search(r"(\d+)\]$", t) else 10**9  # noqa: E731
    tags = sorted(per_tag, key=lambda t: (num(t), t))
    gaps = {t: sorted(r for r, have in where.items() if t not in have) for t in tags}
    gaps = {t: r for t, r in gaps.items() if r}
    if gaps:
        print("Tags missing from some recordings (treated as never seen there, seen flag = 0):")
        for t, recs in gaps.items():
            print(f"  {t}: {len(recs)} recording(s), e.g. {', '.join(recs[:3])}")
    return [c for t in tags for c in per_tag[t]] + extra


def list_csvs(folder):
    """Sorted list of CSV paths in a folder ([] if no folder was given)."""
    if folder is None:
        return []
    folder = os.path.expanduser(folder)
    if not os.path.isdir(folder):
        raise SystemExit(f"Folder not found: {folder}")
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found in {folder}")
    return files


def check_overlap(sets):
    """Warn when the same recording has windows in more than one folder.

    `sets` maps a label ("train", "val", "test") to its list of files.
    """
    where = defaultdict(set)                       # recording name -> folders it appears in
    for label, files in sets.items():
        for f in files:
            m = WINDOW_NAME.match(os.path.basename(f))
            if m:
                where[m["rec"]].add(label)
    shared = {r: sorted(l) for r, l in where.items() if len(l) > 1}
    if shared:
        print(f"WARNING: {len(shared)} recording(s) have windows in more than one folder. "
              f"Overlapping windows make val/test scores look better than they are:")
        for r, labels in sorted(shared.items())[:10]:
            print(f"  {r}: {', '.join(labels)}")
        if len(shared) > 10:
            print(f"  ... and {len(shared) - 10} more")


def load_files(files, in_cols=None, out_cols=None, label="files"):
    """Load a list of window CSVs and stack them into arrays.

    Returns X (N, SEQ_LEN, n_in), Y (N, SEQ_LEN, n_out), in_cols, out_cols.
    Returns None for X and Y if the list is empty.
    """
    if not files:
        return None, None, in_cols, out_cols
    xs, ys = [], []
    for f in progress(files, desc=f"loading {label}"):
        try:
            x, y, in_cols, out_cols = load_csv(f, in_cols, out_cols)
        except ValueError as e:
            print(f"  skipping {e}")
            continue
        xs.append(x)
        ys.append(y)
    if not xs:
        raise SystemExit(f"No usable files in {os.path.dirname(files[0])}")
    return np.stack(xs), np.stack(ys), in_cols, out_cols


def report_missing(label, X, Y, in_cols, out_cols):
    """Print how often tags were missed and how many cells are empty, so data
    problems are visible up front. Call it before fill_tag_gaps."""
    seen = [j for j, c in enumerate(in_cols) if is_seen_col(c)]
    if seen:
        S = X[:, :, seen]
        missed = 1.0 - np.nanmean(S, axis=(0, 1))      # fraction of rows each tag was missed
        never = (np.nan_to_num(S) == 0).all(axis=1).sum(0)   # windows where a tag is never seen
        print(f"  {label}: tags missed in {1 - np.nanmean(S):.2%} of rows overall "
              f"(held at the last detection or zero pose; seen flag = 0).")
        worst = [k for k in np.argsort(missed)[::-1][:5] if missed[k] > 0]
        if worst:
            print("    Most often missed:")
            for k in worst:
                print(f"      {tag_of(in_cols[seen[k]])}: {missed[k]:.1%} of rows, "
                      f"never seen in {never[k]} window(s)")
    other = [j for j, c in enumerate(in_cols) if not c.startswith("derived/tf[")]
    x_nan = np.isnan(X[:, :, other]) if other else np.zeros((0,), bool)
    if x_nan.any():
        print(f"  {label}: {x_nan.mean():.2%} of non-tag input cells empty "
              f"({', '.join(in_cols[j] for j in other)}); set to the training mean.")
    y_nan = np.isnan(Y)
    if y_nan.any():
        print(f"  {label}: {y_nan.mean():.2%} of output cells empty (left out of the loss)")
    dead = np.where(y_nan.all(axis=(0, 1)))[0]       # outputs with no readings at all
    if len(dead):
        print(f"  {label}: no readings at all for {[out_cols[i] for i in dead]}")
    if not x_nan.any() and not y_nan.any() and not (seen and np.nanmean(X[:, :, seen]) < 1):
        print(f"  {label}: no empty cells")


def prepare_inputs(X, stats):
    """Raw inputs (N, T, n_in) with NaN gaps -> filled, normalized float32 array.

    Missed tags: last detection / zero pose (fill_tag_gaps), never the mean.
    Anything else still empty (e.g. a blank t_s) becomes 0 after normalization."""
    X = fill_tag_gaps(np.array(X, np.float32, copy=True), stats["in_cols"])
    Xn = (X - stats["x_mean"]) / stats["x_std"]
    return np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class InputPrep(nn.Module):
    """Optional input preprocessing shared by all models (no learned weights).

    use_diff:   append the change in every input since the previous row, so the
                model sees how fast each tag is deforming. Only looks backwards,
                so it still works live. The first row has no previous row -> 0.
    in_dropout: during training, randomly zero this fraction of input values,
                so the model can't lean on a few particular tags. Off at eval.
    """

    def __init__(self, n_in, use_diff=False, in_dropout=0.0):
        super().__init__()
        self.use_diff = use_diff
        self.drop = nn.Dropout(in_dropout)
        self.n_feat = n_in * 2 if use_diff else n_in   # width after preprocessing

    def forward(self, x):                              # x: (B, T, n_in)
        if self.use_diff:
            d = torch.diff(x, dim=1, prepend=x[:, :1])  # row t minus row t-1
            x = torch.cat([x, d], dim=-1)              # (B, T, 2 * n_in)
        return self.drop(x)


class RecurrentModel(nn.Module):
    """RNN / GRU / LSTM followed by a linear layer applied at every time step."""

    def __init__(self, n_in, n_out, hidden, layers, kind, dropout,
                 use_diff=False, in_dropout=0.0):
        super().__init__()
        self.prep = InputPrep(n_in, use_diff, in_dropout)
        rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[kind]
        # PyTorch's built-in RNN dropout only acts BETWEEN stacked layers,
        # so it does nothing when layers == 1.
        self.rnn = rnn_cls(self.prep.n_feat, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        # This extra dropout on the RNN output works for any number of layers.
        # It is only active during training (model.train()), not at eval time.
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x):                # x: (B, T, n_in)
        x = self.prep(x)                 # (B, T, n_in or 2*n_in)
        # No initial hidden state is passed, so PyTorch starts from zeros on
        # every call. That is what makes the model stateless.
        out, _ = self.rnn(x)             # (B, T, hidden)
        return self.head(self.drop(out)) # (B, T, n_out)


def build_model(n_in, n_out, a):
    """Create the model chosen by --model, using the other args for its size."""
    # .get(..., default) so checkpoints saved before these flags existed still load.
    extra = dict(use_diff=a.get("use_diff", False), in_dropout=a.get("in_dropout", 0.0))
    return RecurrentModel(n_in, n_out, a["hidden"], a["layers"], a["model"],
                          a["dropout"], **extra)


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------
def run_epoch(model, loader, device, y_std, optimizer=None, clip=1.0, desc=None, loss_kind="mse",
              huber_delta=1.0):
    """One pass over a loader. Trains if an optimizer is given, otherwise just scores.

    Training minimizes loss_kind on the scaled outputs (each output divided by its training std, so
    every output counts equally): "mse", or "huber" (squared below huber_delta std, linear above,
    so a few extreme samples weigh less). The returned scores are always MSE-based.
    Shows a progress bar over the batches when `desc` is given.
    Returns (scaled RMSE: all outputs in units of their std, RMSE in original units).
    """
    training = optimizer is not None
    model.train(training)
    sum_sq_norm, sum_sq_orig, n_vals = 0.0, 0.0, 0
    batches = progress(loader, desc=desc) if desc else loader
    with torch.set_grad_enabled(training):
        for x, y, m in batches:
            x, y, m = x.to(device), y.to(device), m.to(device)
            pred = model(x)
            sq = ((pred - y) ** 2) * m              # mask: only real readings count
            if loss_kind == "huber":
                el = nn.functional.huber_loss(pred, y, reduction="none", delta=huber_delta) * m
                loss = el.sum() / m.sum().clamp(min=1)
            else:
                loss = sq.sum() / m.sum().clamp(min=1)
            if not torch.isfinite(loss):
                raise SystemExit("Loss became NaN/inf. Try a lower --lr (e.g. 0.0003).")
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)   # keeps updates stable
                optimizer.step()
            sum_sq_norm += sq.sum().item()
            # Multiply by std^2 to convert squared error back to original units.
            sum_sq_orig += (sq.detach() * y_std ** 2).sum().item()
            n_vals += m.sum().item()
    return (sum_sq_norm / n_vals) ** 0.5, (sum_sq_orig / n_vals) ** 0.5


@torch.no_grad()
def per_output_rmse(model, loader, device, y_std):
    """RMSE for each output column separately, in original units."""
    model.eval()
    sq, n = 0, 0
    for x, y, m in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        e = ((model(x) - y) * y_std) ** 2 * m
        sq = sq + e.sum((0, 1))                     # sum over windows and time steps
        n = n + m.sum((0, 1))
    return (sq / n.clamp(min=1)).sqrt().cpu().numpy()


def save_curve(history, path="rmse_curve.png"):
    """Plot train (and val) RMSE per epoch and save it as a PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")                        # draw to a file, no window needed
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping the RMSE plot")
        return
    ep = range(1, len(history["train"]) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(ep, history["train"], label="train")
    if history["val"]:
        plt.plot(ep, history["val"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("scaled RMSE (each output in units of its std)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"\nScaled RMSE curve saved to {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", required=True, help="folder of training window CSVs")
    p.add_argument("--val_dir", help="folder of validation CSVs (picks the best epoch)")
    p.add_argument("--test_dir", help="folder of test CSVs (scored once at the end)")
    p.add_argument("--model", choices=["rnn", "gru", "lstm"], default="rnn",
                   help="which architecture to train")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    p.add_argument("--hidden", type=int, default=128,
                   help="hidden size")
    p.add_argument("--layers", type=int, default=1, help="number of stacked layers")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="fraction of hidden units randomly zeroed during training "
                        "(e.g. 0.2); fights overfitting")
    p.add_argument("--weight_decay", type=float, default=0.0,
                   help="pulls weights toward zero each step (e.g. 1e-4 to 1e-2); "
                        "fights overfitting")
    p.add_argument("--use_diff", action="store_true",
                   help="also feed the change in each input since the previous row "
                        "(how fast the tags are deforming); works live")
    p.add_argument("--in_dropout", type=float, default=0.0,
                   help="fraction of input values randomly zeroed during training "
                        "(e.g. 0.1); stops the model leaning on a few tags")
    p.add_argument("--ignore_inputs", nargs="+", default=[], metavar="NAME",
                   help="more columns NOT to use as inputs (besides IGNORE_INPUTS: t_s, dt_s), e.g. "
                        "--ignore_inputs 'derived/tf[tag_0_zero->tag_0].seen'")
    p.add_argument("--loss", choices=["mse", "huber"], default="mse",
                   help="training loss on the scaled outputs: mse (default), or huber (squared for small "
                        "errors, linear for large ones, so a few extreme samples weigh less)")
    p.add_argument("--huber_delta", type=float, default=1.0,
                   help="huber only: where the loss turns linear, in units of each output's std (default 1)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="checkpoint.pt", help="where to save the best model")
    args = p.parse_args()

    # Fix random seeds so runs are repeatable.
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Find files and warn if any recording is split across folders. ---
    sets = {"train": list_csvs(args.train_dir),
            "val": list_csvs(args.val_dir),
            "test": list_csvs(args.test_dir)}
    check_overlap(sets)

    # --- Window length: every file must have as many rows as the first training file. ---
    global SEQ_LEN
    SEQ_LEN = len(pd.read_csv(sets["train"][0], usecols=[0]))
    print(f"Window length: {SEQ_LEN} rows")

    # --- Load data. Column lists come from the training files. ---
    # Input columns: every tag that appears in ANY training file (a recording where a tag was
    # never detected has no columns for it; load_csv then treats it as never seen).
    IGNORE_INPUTS.extend(args.ignore_inputs)
    if args.ignore_inputs:
        print(f"Also ignoring inputs: {', '.join(args.ignore_inputs)}")
    in_cols = all_input_cols(sets["train"])
    out_cols = list(OUTPUT_COLS)
    X_tr, Y_tr, in_cols, out_cols = load_files(sets["train"], in_cols, out_cols, label="train")
    X_va, Y_va, _, _ = load_files(sets["val"], in_cols, out_cols, label="val")
    X_te, Y_te, _, _ = load_files(sets["test"], in_cols, out_cols, label="test")
    n_in, n_out = len(in_cols), len(out_cols)

    # --- Report empty cells in each set. ---
    print("Checking for empty cells:")
    for label, X, Y in (("train", X_tr, Y_tr), ("val", X_va, Y_va), ("test", X_te, Y_te)):
        if X is not None:
            report_missing(label, X, Y, in_cols, out_cols)

    # --- Normalize each column using training data only, from real readings
    #     (missed tags are still NaN here, so they don't skew the mean/std). ---
    # Columns that barely change (or are always empty) get mean 0 / std 1.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-empty columns
        x_mean = np.nanmean(X_tr.reshape(-1, n_in), 0)
        x_std = np.nanstd(X_tr.reshape(-1, n_in), 0)
        y_mean = np.nanmean(Y_tr.reshape(-1, n_out), 0)
        y_std = np.nanstd(Y_tr.reshape(-1, n_out), 0)
    x_mean = np.nan_to_num(x_mean).astype(np.float32)
    x_std = np.where(~np.isfinite(x_std) | (x_std < 1e-6), 1.0, x_std).astype(np.float32)
    y_mean = np.nan_to_num(y_mean).astype(np.float32)
    y_std = np.where(~np.isfinite(y_std) | (y_std < 1e-8), 1.0, y_std).astype(np.float32)
    stats = dict(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
                 in_cols=in_cols, out_cols=out_cols, seq_len=SEQ_LEN)

    def make_loader(X, Y, shuffle):
        """Normalized inputs, normalized targets, and a mask (1 = real reading)."""
        if X is None:
            return None
        mask = ~np.isnan(Y)
        Xn = prepare_inputs(X, stats)
        Yn = np.nan_to_num((Y - y_mean) / y_std).astype(np.float32)
        ds = TensorDataset(torch.from_numpy(Xn), torch.from_numpy(Yn),
                           torch.from_numpy(mask.astype(np.float32)))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    train_dl = make_loader(X_tr, Y_tr, True)
    val_dl = make_loader(X_va, Y_va, False)
    test_dl = make_loader(X_te, Y_te, False)
    print(f"{len(X_tr)} train / {len(X_va) if val_dl else 0} val / "
          f"{len(X_te) if test_dl else 0} test windows, "
          f"{n_in} inputs -> {n_out} outputs, model={args.model}, "
          f"dropout={args.dropout}, weight_decay={args.weight_decay}, "
          f"use_diff={args.use_diff}, in_dropout={args.in_dropout}, loss={args.loss}, device={device}")

    # --- Build model and optimizer. ---
    model = build_model(n_in, n_out, vars(args)).to(device)
    # AdamW = Adam with weight decay applied directly to the weights (the
    # recommended way to combine the two). With --weight_decay 0 it is plain Adam.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    y_std_t = torch.from_numpy(y_std).to(device)

    # --- Training loop: keep the epoch with the lowest val SCALED RMSE (train if no val). Scaled =
    #     each output divided by its training std, so acceleration (~m/s²), gyro (~rad/s) and depth
    #     (m) count equally when picking the best epoch. 1.0 ~ no better than predicting the mean. ---
    history = {"train": [], "val": []}               # scaled RMSE per epoch, for the plot
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr_sc, tr_rmse = run_epoch(model, train_dl, device, y_std_t, optimizer,
                                   desc=f"epoch {epoch}/{args.epochs} train",
                                   loss_kind=args.loss, huber_delta=args.huber_delta)
        history["train"].append(tr_sc)
        msg = f"epoch {epoch:3d}/{args.epochs}  train scaled {tr_sc:.4f} (RMSE {tr_rmse:.4f})"
        score = tr_sc
        if val_dl:
            va_sc, va_rmse = run_epoch(model, val_dl, device, y_std_t,
                                       desc=f"epoch {epoch}/{args.epochs} val")
            history["val"].append(va_sc)
            msg += f"  val scaled {va_sc:.4f} (RMSE {va_rmse:.4f})"
            score = va_sc
        if score < best:
            best = score
            torch.save({"model": model.state_dict(), "stats": stats, "args": vars(args)}, args.out)
            msg += "  *saved"
        print(msg)

    # --- Final scores of the best saved model on every set. ---
    model, _ = load_model(args.out, device)
    loaders = [(n, dl) for n, dl in (("train", train_dl), ("val", val_dl), ("test", test_dl)) if dl]
    print(f"\nFinal scores of the saved model (scaled = in units of each output's std; RMSE = original units):")
    per_out = {}
    for name, dl in loaders:
        sc, rmse = run_epoch(model, dl, device, y_std_t)
        per_out[name] = per_output_rmse(model, dl, device, y_std_t)
        print(f"  {name:5s} scaled {sc:.4f}   RMSE {rmse:.4f}")

    print("\nRMSE per output (original units; in brackets: scaled, 1.0 ~ no better than the mean):")
    print(f"  {'output':52s}" + "".join(f"{n:>18s}" for n, _ in loaders))
    for i, col in enumerate(out_cols):
        print(f"  {col:52s}" + "".join(f"{per_out[n][i]:10.4f} ({per_out[n][i] / y_std[i]:4.2f})" for n, _ in loaders))

    # --- Plot train/val RMSE over the epochs, named after --out so each run
    #     keeps its own (runs/lstm64_diff_s0.pt -> runs/lstm64_diff_s0_rmse_curve.png). ---
    base = os.path.splitext(args.out)[0]
    save_curve(history, f"{base}_rmse_curve.png")


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def load_model(path, device="cpu"):
    """Rebuild a saved model (whatever type it was) and return it with its normalization stats."""
    global SEQ_LEN
    ckpt = torch.load(path, map_location=device, weights_only=False)
    s = ckpt["stats"]
    SEQ_LEN = s.get("seq_len", 65)           # checkpoints from before 2 s windows were 65 rows
    model = build_model(len(s["in_cols"]), len(s["out_cols"]), ckpt["args"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), s


@torch.no_grad()
def predict_file(model, stats, csv_path, device="cpu"):
    """Predict one window file (SEQ_LEN rows). Returns a DataFrame (rows x outputs) in original units."""
    x, _, _, _ = load_csv(csv_path, stats["in_cols"], stats["out_cols"])
    xt = torch.from_numpy(prepare_inputs(x[None], stats)).to(device)
    pred = model(xt).squeeze(0).cpu().numpy()
    return pd.DataFrame(pred * stats["y_std"] + stats["y_mean"], columns=stats["out_cols"])


if __name__ == "__main__":
    main()
