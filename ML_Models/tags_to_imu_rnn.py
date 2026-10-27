"""
Stateless RNN for fixed-length tag-deformation sequences (PyTorch).

Data layout assumed:
  - DATA_DIR contains one CSV per training example.
  - Each file has exactly SEQ_LEN rows (65 = 1 second at 65 fps).
  - Each row has N_IN input columns followed by N_OUT target columns
    (155 + 11 = 166 columns by default). Change INPUT_COLS / TARGET_COLS
    below if your files are laid out differently.

"Stateless" = the hidden state is reset to zero for every sequence (every file).
Nothing is carried over between batches or between files.

Usage:
    python train_rnn.py --data_dir path/to/csvs
"""

import argparse
import glob
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEQ_LEN = 65                             # rows per file (1 s at 65 fps)
N_IN = 155
N_OUT = 11
INPUT_COLS = slice(0, N_IN)              # first 155 columns
TARGET_COLS = slice(N_IN, N_IN + N_OUT)  # next 11 columns
HAS_HEADER = True                        # set False if CSVs have no header row

# Output mode:
#   "seq2seq" -> predict 11 values at every frame (targets on every row)
#   "seq2one" -> predict 11 values per file, using the last row's targets
OUTPUT_MODE = "seq2seq"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_csv(path):
    df = pd.read_csv(path, header=0 if HAS_HEADER else None)
    arr = df.to_numpy(dtype=np.float32)
    if arr.shape[1] < N_IN + N_OUT:
        raise ValueError(f"{path}: expected >= {N_IN + N_OUT} columns, got {arr.shape[1]}")
    if arr.shape[0] != SEQ_LEN:
        raise ValueError(f"{path}: expected {SEQ_LEN} rows, got {arr.shape[0]}")
    return arr[:, INPUT_COLS], arr[:, TARGET_COLS]


def load_folder(files):
    """Load files into arrays X (N, 65, 155) and Y (N, 65, 11). Skips bad files with a warning."""
    xs, ys = [], []
    for f in files:
        try:
            x, y = load_csv(f)
        except ValueError as e:
            print(f"  skipping {e}")
            continue
        xs.append(x)
        ys.append(y)
    if not xs:
        raise SystemExit("No usable CSV files.")
    return np.stack(xs), np.stack(ys)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class StatelessRNN(nn.Module):
    def __init__(self, n_in=N_IN, n_out=N_OUT, hidden=128, layers=1,
                 cell="rnn", dropout=0.0, mode=OUTPUT_MODE):
        super().__init__()
        rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.rnn = rnn_cls(n_in, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_out)
        self.mode = mode

    def forward(self, x):
        # No initial hidden state passed -> zeros every call -> stateless.
        out, _ = self.rnn(x)                 # (B, 65, hidden)
        if self.mode == "seq2seq":
            return self.head(out)            # (B, 65, 11)
        return self.head(out[:, -1])         # (B, 11)


def targets_for(y, mode):
    return y if mode == "seq2seq" else y[:, -1]


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
def run_epoch(model, loader, device, y_std, optimizer=None, clip=1.0):
    """Returns (normalized MSE loss, RMS error in original units)."""
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.MSELoss()
    sum_loss, sum_sq_orig, n_vals, n_batches = 0.0, 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            target = targets_for(y, model.mode)
            pred = model(x)
            loss = loss_fn(pred, target)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            # Error in original units: undo the per-column scaling
            err_orig = (pred.detach() - target) * y_std
            sum_sq_orig += (err_orig ** 2).sum().item()
            n_vals += err_orig.numel()
            sum_loss += loss.item()
            n_batches += 1
    return sum_loss / n_batches, (sum_sq_orig / n_vals) ** 0.5


@torch.no_grad()
def per_output_rmse(model, loader, device, y_std):
    """RMS error for each of the 11 outputs separately, in original units."""
    model.eval()
    sq, n = torch.zeros(N_OUT, device=device), 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        err = (model(x) - targets_for(y, model.mode)) * y_std
        sq += (err ** 2).reshape(-1, N_OUT).sum(0)
        n += err.reshape(-1, N_OUT).shape[0]
    return (sq / n).sqrt().cpu().numpy()


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
    p.add_argument("--data_dir", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--cell", choices=["rnn", "gru", "lstm"], default="rnn")
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="rnn_checkpoint.pt")
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found in {args.data_dir}")
    random.shuffle(files)
    n_val = max(1, int(len(files) * args.val_frac)) if len(files) > 1 else 0
    val_files, train_files = files[:n_val], files[n_val:]

    X_tr, Y_tr = load_folder(train_files)
    # Normalization stats from training data only, per column
    x_mean, x_std = X_tr.reshape(-1, N_IN).mean(0), X_tr.reshape(-1, N_IN).std(0) + 1e-8
    y_mean, y_std = Y_tr.reshape(-1, N_OUT).mean(0), Y_tr.reshape(-1, N_OUT).std(0) + 1e-8
    stats = dict(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)

    def make_loader(X, Y, shuffle):
        Xn = torch.from_numpy((X - x_mean) / x_std)
        Yn = torch.from_numpy((Y - y_mean) / y_std)
        return DataLoader(TensorDataset(Xn, Yn), batch_size=args.batch_size, shuffle=shuffle)

    train_dl = make_loader(X_tr, Y_tr, True)
    val_dl = None
    if val_files:
        X_va, Y_va = load_folder(val_files)
        val_dl = make_loader(X_va, Y_va, False)
    print(f"{len(train_dl.dataset)} train files, {len(val_dl.dataset) if val_dl else 0} val files, "
          f"shape per file = ({SEQ_LEN}, {N_IN}) -> {'(65, 11)' if OUTPUT_MODE == 'seq2seq' else '(11,)'}, "
          f"cell={args.cell}, device={device}")

    model = StatelessRNN(hidden=args.hidden, layers=args.layers, cell=args.cell).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    y_std_t = torch.from_numpy(y_std).to(device)

    history = {"train": [], "val": []}
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_rmse = run_epoch(model, train_dl, device, y_std_t, optimizer)
        history["train"].append(tr_rmse)
        msg = f"epoch {epoch:3d}  train loss {tr_loss:.4f}  train RMSE {tr_rmse:.4f}"
        score = tr_rmse
        if val_dl:
            va_loss, va_rmse = run_epoch(model, val_dl, device, y_std_t)
            history["val"].append(va_rmse)
            msg += f"  |  val loss {va_loss:.4f}  val RMSE {va_rmse:.4f}"
            score = va_rmse
        print(msg)
        if score < best:
            best = score
            torch.save({"model": model.state_dict(), "stats": stats, "args": vars(args),
                        "mode": model.mode}, args.out)

    print(f"\nBest {'val' if val_dl else 'train'} RMSE {best:.4f}; saved to {args.out}")
    model, _ = load_model(args.out, device)
    per_out = per_output_rmse(model, val_dl or train_dl, device, y_std_t)
    print("RMSE per output (best model):")
    for i, v in enumerate(per_out):
        print(f"  output {i:2d}: {v:.4f}")
    save_curve(history)


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------
def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = StatelessRNN(hidden=a["hidden"], layers=a["layers"], cell=a["cell"], mode=ckpt["mode"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt["stats"]


@torch.no_grad()
def predict_file(model, stats, csv_path, device="cpu"):
    """Returns predictions in original (un-normalized) units."""
    x, _ = load_csv(csv_path)
    x = (x - stats["x_mean"]) / stats["x_std"]
    xt = torch.from_numpy(x).unsqueeze(0).to(device)
    pred = model(xt).squeeze(0).cpu().numpy()
    return pred * stats["y_std"] + stats["y_mean"]


if __name__ == "__main__":
    main()