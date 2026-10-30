"""
Stateless sequence model: AprilTag deformations (+ time) -> IMU + depth (PyTorch).

Data = three folders of 65-row window CSVs (train / validation / test), one
example per file (as written by make_windows.py).
  - train: used to fit the model
  - val:   checked every epoch; the epoch with the lowest val RMSE is saved
  - test:  never touched during training; scored once at the end

Columns used:
  - Inputs  = every "derived/tf[tag_*...]" column + "t_s"     -> 141 inputs
  - Outputs = the 10 "mavros/imu/data.*" columns + "bluerov/depth.data" -> 11
Empty output cells are left out of the loss and RMSE.
With --orientation rollpitch, the 4 quaternion outputs are replaced by roll and
pitch in degrees (computed from the quaternion) -> 9 outputs.

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
  transformer  Transformer encoder with a causal mask (each step only sees
               itself and earlier steps, like the recurrent models)

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
SEQ_LEN = 65                 # rows per file (1 s at 65 fps)
EXTRA_INPUTS = ["t_s"]       # non-tag input columns (time since zero)
DROP_OUTPUTS = []            # outputs to leave out; filled from --drop_outputs in main()

# How orientation is predicted; set from --orientation in main().
#   quat       the IMU's raw quaternion x, y, z, w (heading leaks into all four)
#   rollpitch  roll and pitch in degrees (heading-free)
ORIENTATION = ["quat"]       # a list so main() can change it in place
_O = "mavros/imu/data.orientation."
QUAT_COLS = [_O + c for c in "xyzw"]
ROLLPITCH_COLS = [_O + "roll_deg", _O + "pitch_deg"]

# Window filenames look like <recording>_w0000_rows0-64.csv; this pulls out <recording>.
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
    """Tag deformation columns and time are model inputs."""
    return c.startswith("derived/tf[") or c in EXTRA_INPUTS


def short_name(c):
    """'mavros/imu/data.orientation.z' -> 'orientation.z', 'bluerov/depth.data' -> 'depth.data'."""
    return c.replace("mavros/imu/data.", "").replace("bluerov/", "")


def is_imu_or_depth(c):
    """True for every IMU and depth column, before any are dropped."""
    return c.startswith("mavros/imu/data.") or c == "bluerov/depth.data"


def is_output_col(c):
    """IMU and depth columns are what the model predicts, minus any in --drop_outputs
    (given either as the full column name or the short name, e.g. orientation.z).
    Only the orientation columns for the chosen --orientation format are used."""
    if not is_imu_or_depth(c) or c in DROP_OUTPUTS or short_name(c) in DROP_OUTPUTS:
        return False
    if c in QUAT_COLS:
        return ORIENTATION[0] == "quat"
    if c in ROLLPITCH_COLS:
        return ORIENTATION[0] == "rollpitch"
    return True


def add_orientation_cols(df):
    """Add roll and pitch columns (degrees) computed from the IMU quaternion.

    They ignore heading (yaw), which the tags can't see, and don't change if the
    quaternion's sign flips (q and -q are the same rotation), because every term
    multiplies two components together. Missing quaternion values stay missing.
    """
    if not all(c in df.columns for c in QUAT_COLS):
        return df
    x, y, z, w = (pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64) for c in QUAT_COLS)
    # Roll / pitch (standard ROS convention). Smooth unless the robot goes
    # upside down (roll wraps at +-180) or points straight up/down (pitch +-90).
    roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
    # Add both columns in one step; adding them one at a time to a wide table
    # makes pandas print a "highly fragmented" PerformanceWarning.
    new = pd.DataFrame({ROLLPITCH_COLS[0]: roll, ROLLPITCH_COLS[1]: pitch}, index=df.index)
    return pd.concat([df.drop(columns=ROLLPITCH_COLS, errors="ignore"), new], axis=1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_csv(path, in_cols=None, out_cols=None):
    """Read one window file.

    Returns X (65, n_in), Y (65, n_out), and the column lists. Y keeps NaN
    where a reading is missing so those cells can be masked out of the loss.
    The first file read decides the column lists; later files must match.
    """
    df = add_orientation_cols(pd.read_csv(path))
    if in_cols is None:
        in_cols = [c for c in df.columns if is_input_col(c)]
        out_cols = [c for c in df.columns if is_output_col(c)]
    if len(df) != SEQ_LEN:
        raise ValueError(f"{os.path.basename(path)}: expected {SEQ_LEN} rows, got {len(df)}")
    missing = [c for c in in_cols + out_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(path)}: missing columns, e.g. {missing[:3]}")

    # Convert to numbers; anything unreadable or infinite becomes NaN.
    X = df[in_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    # Fill short gaps in the inputs by interpolation. A column that is empty for
    # the whole window stays NaN and is filled with the training mean later.
    X = X.interpolate(limit_direction="both").to_numpy(np.float32)
    Y = df[out_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return X, Y.to_numpy(np.float32), in_cols, out_cols


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

    Returns X (N, 65, n_in), Y (N, 65, n_out), in_cols, out_cols.
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
    """Print how many cells are empty, so data problems are visible up front."""
    x_nan = np.isnan(X)
    if x_nan.any():
        win = x_nan.any(axis=(1, 2)).sum()           # windows with at least one empty input
        per_col = x_nan.all(axis=1).sum(0)           # windows where a column is empty throughout
        print(f"  {label}: {x_nan.mean():.2%} of input cells empty, in {win} of {len(X)} windows; "
              f"filled with the training mean.")
        worst = [i for i in np.argsort(per_col)[::-1][:5] if per_col[i]]
        if worst:
            print("    Most often empty for a whole window:")
            for i in worst:
                print(f"      {in_cols[i]}: {per_col[i]} window(s)")
    y_nan = np.isnan(Y)
    if y_nan.any():
        print(f"  {label}: {y_nan.mean():.2%} of output cells empty (left out of the loss)")
    dead = np.where(y_nan.all(axis=(0, 1)))[0]       # outputs with no readings at all
    if len(dead):
        print(f"  {label}: no readings at all for {[out_cols[i] for i in dead]}")
    if not x_nan.any() and not y_nan.any():
        print(f"  {label}: no empty cells")


def fill_inputs(Xn):
    """After normalization 0 = training mean, so empty input cells become the mean."""
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

    def forward(self, x):                # x: (B, 65, n_in)
        x = self.prep(x)                 # (B, 65, n_in or 2*n_in)
        # No initial hidden state is passed, so PyTorch starts from zeros on
        # every call. That is what makes the model stateless.
        out, _ = self.rnn(x)             # (B, 65, hidden)
        return self.head(self.drop(out)) # (B, 65, n_out)


class TransformerModel(nn.Module):
    """Transformer encoder that predicts an output at every time step.

    Steps: project inputs to `hidden` size, add a learned position embedding
    (a Transformer has no built-in sense of order), run the encoder with a
    causal mask, then map each step to the outputs.
    """

    def __init__(self, n_in, n_out, hidden, layers, nhead, dropout,
                 use_diff=False, in_dropout=0.0):
        super().__init__()
        self.prep = InputPrep(n_in, use_diff, in_dropout)
        self.inp = nn.Linear(self.prep.n_feat, hidden)
        self.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, hidden))  # one vector per row
        layer = nn.TransformerEncoderLayer(hidden, nhead, dim_feedforward=4 * hidden,
                                           dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(hidden, n_out)
        # Causal mask: True above the diagonal = "may not look at future steps".
        mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x):                # x: (B, T, n_in)
        T = x.shape[1]
        x = self.prep(x)
        h = self.inp(x) + self.pos[:, :T]
        h = self.encoder(h, mask=self.mask[:T, :T])
        return self.head(h)              # (B, T, n_out)


def build_model(n_in, n_out, a):
    """Create the model chosen by --model, using the other args for its size."""
    # .get(..., default) so checkpoints saved before these flags existed still load.
    extra = dict(use_diff=a.get("use_diff", False), in_dropout=a.get("in_dropout", 0.0))
    if a["model"] == "transformer":
        return TransformerModel(n_in, n_out, a["hidden"], a["layers"], a["nhead"],
                                a["dropout"], **extra)
    return RecurrentModel(n_in, n_out, a["hidden"], a["layers"], a["model"],
                          a["dropout"], **extra)


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------
def run_epoch(model, loader, device, y_std, optimizer=None, clip=1.0, desc=None):
    """One pass over a loader. Trains if an optimizer is given, otherwise just scores.

    Shows a progress bar over the batches when `desc` is given.
    Returns (normalized masked MSE, RMSE in original units).
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
    return sum_sq_norm / n_vals, (sum_sq_orig / n_vals) ** 0.5


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
    plt.plot(ep, history["train"], label="train RMSE")
    if history["val"]:
        plt.plot(ep, history["val"], label="val RMSE")
    plt.xlabel("epoch")
    plt.ylabel("RMSE (original units)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"\nRMSE curve saved to {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", required=True, help="folder of training window CSVs")
    p.add_argument("--val_dir", help="folder of validation CSVs (picks the best epoch)")
    p.add_argument("--test_dir", help="folder of test CSVs (scored once at the end)")
    p.add_argument("--model", choices=["rnn", "gru", "lstm", "transformer"], default="rnn",
                   help="which architecture to train")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    p.add_argument("--hidden", type=int, default=128,
                   help="hidden size (RNN/GRU/LSTM) or model width (Transformer)")
    p.add_argument("--layers", type=int, default=1, help="number of stacked layers")
    p.add_argument("--nhead", type=int, default=4,
                   help="attention heads (Transformer only; must divide --hidden)")
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
    p.add_argument("--orientation", choices=["quat", "rollpitch"], default="quat",
                   help="how to predict orientation: raw quaternion (x, y, z, w) or "
                        "roll + pitch in degrees (ignores heading, which the tags can't see)")
    p.add_argument("--drop_outputs", nargs="+", default=[], metavar="NAME",
                   help="outputs to leave out of training, e.g. "
                        "--drop_outputs orientation.z orientation.w")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="checkpoint.pt", help="where to save the best model")
    args = p.parse_args()
    if args.model == "transformer" and args.hidden % args.nhead:
        raise SystemExit("--hidden must be divisible by --nhead for the Transformer")

    # Fix random seeds so runs are repeatable.
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Find files and warn if any recording is split across folders. ---
    sets = {"train": list_csvs(args.train_dir),
            "val": list_csvs(args.val_dir),
            "test": list_csvs(args.test_dir)}
    check_overlap(sets)

    # --- Outputs to leave out. Check the names against the first training file,
    #     so a typo stops the run instead of silently dropping nothing. ---
    ORIENTATION[0] = args.orientation                # is_output_col() picks the matching columns
    if args.orientation == "rollpitch":
        print("Orientation: predicting roll and pitch (degrees) instead of the quaternion")
    if args.drop_outputs:
        header = add_orientation_cols(pd.read_csv(sets["train"][0], nrows=1)).columns
        known = {n for c in header if is_imu_or_depth(c) for n in (c, short_name(c))}
        unknown = [d for d in args.drop_outputs if d not in known]
        if unknown:
            outs = [short_name(c) for c in header if is_imu_or_depth(c)]
            raise SystemExit(f"Unknown --drop_outputs {unknown}. Choose from: {', '.join(outs)}")
        DROP_OUTPUTS[:] = args.drop_outputs          # is_output_col() now skips these
        print(f"Leaving out outputs: {', '.join(args.drop_outputs)}")

    # --- Load data. Column lists come from the training files. ---
    X_tr, Y_tr, in_cols, out_cols = load_files(sets["train"], label="train")
    X_va, Y_va, _, _ = load_files(sets["val"], in_cols, out_cols, label="val")
    X_te, Y_te, _, _ = load_files(sets["test"], in_cols, out_cols, label="test")
    n_in, n_out = len(in_cols), len(out_cols)

    # --- Report empty cells in each set. ---
    print("Checking for empty cells:")
    for label, X, Y in (("train", X_tr, Y_tr), ("val", X_va, Y_va), ("test", X_te, Y_te)):
        if X is not None:
            report_missing(label, X, Y, in_cols, out_cols)

    # --- Normalize each column using training data only (ignoring empty cells). ---
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
                 in_cols=in_cols, out_cols=out_cols)

    def make_loader(X, Y, shuffle):
        """Normalized inputs, normalized targets, and a mask (1 = real reading)."""
        if X is None:
            return None
        mask = ~np.isnan(Y)
        Xn = fill_inputs((X - x_mean) / x_std)
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
          f"use_diff={args.use_diff}, in_dropout={args.in_dropout}, device={device}")

    # --- Build model and optimizer. ---
    model = build_model(n_in, n_out, vars(args)).to(device)
    # AdamW = Adam with weight decay applied directly to the weights (the
    # recommended way to combine the two). With --weight_decay 0 it is plain Adam.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    y_std_t = torch.from_numpy(y_std).to(device)

    # --- Training loop: keep the epoch with the lowest val RMSE (train RMSE if no val). ---
    history = {"train": [], "val": []}               # RMSE per epoch, for the plot
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        _, tr_rmse = run_epoch(model, train_dl, device, y_std_t, optimizer,
                               desc=f"epoch {epoch}/{args.epochs} train")
        history["train"].append(tr_rmse)
        msg = f"epoch {epoch:3d}/{args.epochs}  train RMSE {tr_rmse:.4f}"
        score = tr_rmse
        if val_dl:
            _, va_rmse = run_epoch(model, val_dl, device, y_std_t,
                                   desc=f"epoch {epoch}/{args.epochs} val")
            history["val"].append(va_rmse)
            msg += f"  val RMSE {va_rmse:.4f}"
            score = va_rmse
        if score < best:
            best = score
            torch.save({"model": model.state_dict(), "stats": stats, "args": vars(args)}, args.out)
            msg += "  *saved"
        print(msg)

    # --- Final scores of the best saved model on every set. ---
    model, _ = load_model(args.out, device)
    loaders = [(n, dl) for n, dl in (("train", train_dl), ("val", val_dl), ("test", test_dl)) if dl]
    print(f"\nFinal RMSE of the saved model (original units):")
    per_out = {}
    for name, dl in loaders:
        _, rmse = run_epoch(model, dl, device, y_std_t)
        per_out[name] = per_output_rmse(model, dl, device, y_std_t)
        print(f"  {name:5s} RMSE {rmse:.4f}")

    print("\nRMSE per output:")
    print(f"  {'output':42s}" + "".join(f"{n:>10s}" for n, _ in loaders))
    for i, col in enumerate(out_cols):
        print(f"  {col:42s}" + "".join(f"{per_out[n][i]:10.4f}" for n, _ in loaders))

    # --- Plot train/val RMSE over the epochs, named after --out so each run
    #     keeps its own (runs/lstm64_diff_s0.pt -> runs/lstm64_diff_s0_rmse_curve.png). ---
    base = os.path.splitext(args.out)[0]
    save_curve(history, f"{base}_rmse_curve.png")


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def load_model(path, device="cpu"):
    """Rebuild a saved model (whatever type it was) and return it with its normalization stats."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    s = ckpt["stats"]
    model = build_model(len(s["in_cols"]), len(s["out_cols"]), ckpt["args"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), s


@torch.no_grad()
def predict_file(model, stats, csv_path, device="cpu"):
    """Predict one 65-row window file. Returns a DataFrame (65 x 11) in original units."""
    x, _, _, _ = load_csv(csv_path, stats["in_cols"], stats["out_cols"])
    xt = torch.from_numpy(fill_inputs((x - stats["x_mean"]) / stats["x_std"])).unsqueeze(0).to(device)
    pred = model(xt).squeeze(0).cpu().numpy()
    return pd.DataFrame(pred * stats["y_std"] + stats["y_mean"], columns=stats["out_cols"])


if __name__ == "__main__":
    main()
