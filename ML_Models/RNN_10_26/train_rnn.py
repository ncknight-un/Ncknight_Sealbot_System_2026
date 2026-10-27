"""
Stateless RNN: AprilTag deformations (+ time) -> IMU + depth (PyTorch).

Data = three folders of 65-row window CSVs (train / validation / test), one
example per file, e.g. sub1_tagged_9.89-49.95s_tag_to_imu_format_w0000_rows0-64.csv
(as written by make_windows.py).
  - train: used to fit the model
  - val:   checked every epoch; the epoch with the lowest val RMSE is saved
  - test:  never touched during training; scored once at the end

Columns used:
  - Inputs  = every "derived/tf[tag_*...]" column (20 tags x 7 = 140 here)
              + "t_s" (time since zero)                     -> 141 inputs
  - Outputs = the 10 "mavros/imu/data.*" columns + "bluerov/depth.data" -> 11
  - "dt_s" is not used.

Any empty output cells are left out of the loss and RMSE.

Windows cut from the same recording overlap, so put each recording's windows
in only one folder. The script warns if a recording name (the part of the file
name before _wNNNN_rows...) shows up in more than one folder.

"Stateless" = the hidden state is reset to zero for every window.

Usage:
    python train_rnn.py --train_dir data/train --val_dir data/val --test_dir data/test
"""

import argparse
import glob
import os
import random
import re
import sys
import time
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

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None


def progress(iterable, desc=None, leave=True, unit="it"):
    """Progress bar: tqdm if installed, otherwise a simple built-in text bar."""
    if _tqdm is not None:
        return _tqdm(iterable, desc=desc, leave=leave, unit=unit, dynamic_ncols=True)
    return _SimpleBar(iterable, desc, leave)


class _SimpleBar:
    def __init__(self, iterable, desc, leave):
        self.it, self.desc, self.leave = iterable, desc or "", leave
        try:
            self.total = len(iterable)
        except TypeError:
            self.total = None

    def __iter__(self):
        if not self.total or not sys.stdout.isatty():
            yield from self.it
            return
        step = max(1, self.total // 50)
        for i, x in enumerate(self.it, 1):
            yield x
            if i % step == 0 or i == self.total:
                done = int(30 * i / self.total)
                sys.stdout.write(f"\r{self.desc} [{'#' * done}{'.' * (30 - done)}] {i}/{self.total}")
                sys.stdout.flush()
        sys.stdout.write("\n" if self.leave else "\r" + " " * (len(self.desc) + 50) + "\r")
        sys.stdout.flush()


WINDOW_NAME = re.compile(r"^(?P<rec>.+)_w\d+_rows(?P<start>\d+)-(?P<end>\d+)\.csv$")


def is_input_col(c):
    return c.startswith("derived/tf[") or c in EXTRA_INPUTS


def is_output_col(c):
    return c.startswith("mavros/imu/data.") or c == "bluerov/depth.data"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_csv(path, in_cols=None, out_cols=None):
    """Returns X (65, n_in), Y (65, n_out) with NaN in Y where a reading is missing."""
    df = pd.read_csv(path)
    if in_cols is None:
        in_cols = [c for c in df.columns if is_input_col(c)]
        out_cols = [c for c in df.columns if is_output_col(c)]
        for c in EXTRA_INPUTS:
            if c not in in_cols:
                raise ValueError(f"{os.path.basename(path)}: no '{c}' column")
    if len(df) != SEQ_LEN:
        raise ValueError(f"{os.path.basename(path)}: expected {SEQ_LEN} rows, got {len(df)}")
    missing = [c for c in in_cols + out_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(path)}: missing columns, e.g. {missing[:3]}")
    # Short gaps are interpolated; a column empty for the whole window stays NaN
    # here and is filled with the training mean later (see fill_inputs).
    X = df[in_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.interpolate(limit_direction="both").to_numpy(np.float32)
    Y = df[out_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    Y = Y.to_numpy(np.float32)
    return X, Y, in_cols, out_cols


def list_csvs(folder, label):
    if folder is None:
        return []
    folder = os.path.expanduser(folder)
    if not os.path.isdir(folder):
        raise SystemExit(f"{label} folder not found: {folder}")
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found in {label} folder: {folder}")
    return files


def recording_name(path):
    m = WINDOW_NAME.match(os.path.basename(path))
    return m["rec"] if m else None


def check_overlap(sets):
    """Warn when the same recording has windows in more than one folder."""
    where = defaultdict(set)
    for label, files in sets.items():
        for f in files:
            rec = recording_name(f)
            if rec:
                where[rec].add(label)
    shared = {r: sorted(l) for r, l in where.items() if len(l) > 1}
    if shared:
        print(f"WARNING: {len(shared)} recording(s) have windows in more than one folder. "
              f"Overlapping windows make val/test scores look better than they are:")
        for r, labels in sorted(shared.items())[:10]:
            print(f"  {r}: {', '.join(labels)}")
        if len(shared) > 10:
            print(f"  ... and {len(shared) - 10} more")


def load_files(files, in_cols=None, out_cols=None, label="files"):
    xs, ys = [], []
    for f in progress(files, desc=f"loading {label}", unit="file"):
        try:
            x, y, in_cols, out_cols = load_csv(f, in_cols, out_cols)
        except ValueError as e:
            print(f"  skipping {e}")
            continue
        xs.append(x); ys.append(y)
    if not xs:
        return None, None, in_cols, out_cols
    return np.stack(xs), np.stack(ys), in_cols, out_cols


def report_missing(label, X, Y, in_cols, out_cols):
    """Print how many cells are empty, so data problems are visible up front."""
    x_nan = np.isnan(X)
    if x_nan.any():
        win = x_nan.any(axis=1).any(axis=1).sum()
        per_col = x_nan.all(axis=1).sum(0)  # windows where the column is empty throughout
        worst = np.argsort(per_col)[::-1][:5]
        print(f"  {label}: {x_nan.mean():.2%} of input cells empty, in {win} of {len(X)} windows; "
              f"filled with the training mean. Most often empty for a whole window:")
        for i in worst:
            if per_col[i]:
                print(f"    {in_cols[i]}: {per_col[i]} window(s)")
    y_nan = np.isnan(Y)
    if y_nan.all(axis=(0, 1)).any():
        cols = [out_cols[i] for i in np.where(y_nan.all(axis=(0, 1)))[0]]
        print(f"  {label}: no readings at all for {cols}")


def fill_inputs(Xn):
    """After normalization 0 = training mean, so empty input cells become the mean."""
    return np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class StatelessRNN(nn.Module):
    def __init__(self, n_in, n_out, hidden=128, layers=1, cell="rnn", dropout=0.0):
        super().__init__()
        rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.rnn = rnn_cls(n_in, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x):
        # No initial hidden state passed -> zeros every call -> stateless.
        out, _ = self.rnn(x)       # (B, 65, hidden)
        return self.head(out)      # (B, 65, n_out)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
def run_epoch(model, loader, device, y_std, optimizer=None, clip=1.0, desc=None):
    """Returns (normalized masked MSE, RMS error in original units)."""
    training = optimizer is not None
    model.train(training)
    sum_sq_norm, sum_sq_orig, n_vals = 0.0, 0.0, 0
    with torch.set_grad_enabled(training):
        for x, y, m in progress(loader, desc=desc, leave=False, unit="batch"):
            x, y, m = x.to(device), y.to(device), m.to(device)
            pred = model(x)
            sq = ((pred - y) ** 2) * m          # only real readings count
            loss = sq.sum() / m.sum().clamp(min=1)
            if not torch.isfinite(loss):
                raise SystemExit(
                    "Loss became NaN/inf. Try a lower --lr (e.g. 0.0003) or --cell gru; "
                    "if it happens on the first batch, check the data report printed above.")
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            sum_sq_norm += sq.sum().item()
            sum_sq_orig += (sq.detach() * y_std ** 2).sum().item()
            n_vals += m.sum().item()
    return sum_sq_norm / n_vals, (sum_sq_orig / n_vals) ** 0.5


@torch.no_grad()
def per_output_rmse(model, loader, device, y_std):
    model.eval()
    sq, n = 0, 0
    for x, y, m in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        e = ((model(x) - y) * y_std) ** 2 * m
        sq = sq + e.sum((0, 1))
        n = n + m.sum((0, 1))
    return (sq / n.clamp(min=1)).sqrt().cpu().numpy()


def save_curve(history, path="rmse_curve.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ep = range(1, len(history["train"]) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(ep, history["train"], label="train RMSE")
    if history["val"]:
        plt.plot(ep, history["val"], label="val RMSE")
    plt.xlabel("epoch"); plt.ylabel("RMS error (original units)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()
    print(f"RMSE curve saved to {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", required=True, help="folder of training window CSVs")
    p.add_argument("--val_dir", help="folder of validation window CSVs (picks the best epoch)")
    p.add_argument("--test_dir", help="folder of test window CSVs (scored once at the end)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--cell", choices=["rnn", "gru", "lstm"], default="rnn")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="rnn_checkpoint.pt")
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sets = {"train": list_csvs(args.train_dir, "train"),
            "val": list_csvs(args.val_dir, "val"),
            "test": list_csvs(args.test_dir, "test")}
    check_overlap(sets)
    if not sets["val"]:
        print("No --val_dir given: the epoch with the lowest train RMSE will be saved.")
    if not sets["test"]:
        print("No --test_dir given: no final test score.")

    X_tr, Y_tr, in_cols, out_cols = load_files(sets["train"], label="train")
    if X_tr is None:
        raise SystemExit("No usable training files.")
    X_va, Y_va, _, _ = load_files(sets["val"], in_cols, out_cols, label="val")
    X_te, Y_te, _, _ = load_files(sets["test"], in_cols, out_cols, label="test")
    n_in, n_out = len(in_cols), len(out_cols)

    print("Checking for empty cells:")
    for label, X, Y in (("train", X_tr, Y_tr), ("val", X_va, Y_va), ("test", X_te, Y_te)):
        if X is not None:
            report_missing(label, X, Y, in_cols, out_cols)

    # Per-column normalization from training data only, ignoring empty cells.
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
        mask = ~np.isnan(Y)
        Yn = np.nan_to_num((Y - y_mean) / y_std)
        tensors = [torch.from_numpy(fill_inputs((X - x_mean) / x_std)), torch.from_numpy(Yn).float(),
                   torch.from_numpy(mask.astype(np.float32))]
        return DataLoader(TensorDataset(*tensors), batch_size=args.batch_size, shuffle=shuffle)

    train_dl = make_loader(X_tr, Y_tr, True)
    val_dl = make_loader(X_va, Y_va, False) if X_va is not None else None
    test_dl = make_loader(X_te, Y_te, False) if X_te is not None else None
    print(f"{len(X_tr)} train / {len(X_va) if val_dl else 0} val / {len(X_te) if test_dl else 0} test files, "
          f"each ({SEQ_LEN}, {n_in}) -> ({SEQ_LEN}, {n_out}), cell={args.cell}, device={device}")

    model = StatelessRNN(n_in, n_out, hidden=args.hidden, layers=args.layers, cell=args.cell).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    y_std_t = torch.from_numpy(y_std).float().to(device)

    history = {"train": [], "val": []}
    best = float("inf")
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_rmse = run_epoch(model, train_dl, device, y_std_t, optimizer,
                                     desc=f"epoch {epoch}/{args.epochs} train")
        history["train"].append(tr_rmse)
        msg = f"epoch {epoch:3d}/{args.epochs}  train loss {tr_loss:.4f}  train RMSE {tr_rmse:.4f}"
        score = tr_rmse
        if val_dl:
            va_loss, va_rmse = run_epoch(model, val_dl, device, y_std_t,
                                         desc=f"epoch {epoch}/{args.epochs} val")
            history["val"].append(va_rmse)
            msg += f"  |  val loss {va_loss:.4f}  val RMSE {va_rmse:.4f}"
            score = va_rmse
        if score < best:
            best = score
            torch.save({"model": model.state_dict(), "stats": stats, "args": vars(args)}, args.out)
            msg += "  *saved"
        elapsed = time.time() - t_start
        eta = elapsed / epoch * (args.epochs - epoch)
        msg += f"  [{elapsed:.0f}s, ~{eta:.0f}s left]"
        print(msg)

    print(f"\nBest {'val' if val_dl else 'train'} RMSE {best:.4f}; saved to {args.out}")
    model, _ = load_model(args.out, device)

    # Final scores of the saved (best) model on every set
    print("\nFinal RMSE of the saved model (original units):")
    sets_dl = [("train", train_dl), ("val", val_dl), ("test", test_dl)]
    sets_dl = [(n, dl) for n, dl in sets_dl if dl is not None]
    per_out = {}
    for name, dl in sets_dl:
        _, rmse = run_epoch(model, dl, device, y_std_t)
        per_out[name] = per_output_rmse(model, dl, device, y_std_t)
        print(f"  {name:5s} RMSE {rmse:.4f}")

    print("\nRMSE per output:")
    print(f"  {'output':42s}" + "".join(f"{n:>10s}" for n, _ in sets_dl))
    for i, col in enumerate(out_cols):
        print(f"  {col:42s}" + "".join(f"{per_out[n][i]:10.4f}" for n, _ in sets_dl))
    save_curve(history)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    a, s = ckpt["args"], ckpt["stats"]
    model = StatelessRNN(len(s["in_cols"]), len(s["out_cols"]),
                         hidden=a["hidden"], layers=a["layers"], cell=a["cell"])
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
