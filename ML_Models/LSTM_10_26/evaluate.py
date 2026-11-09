"""
How good is a trained model? Scores each output against a do-nothing baseline.

For every output, on each folder you give it:
  RMSE        - the model's error, in original units
  baseline    - error of always predicting the training average (no model at all)
  skill       - how much of the baseline's squared error the model removes:
                  1 - (RMSE / baseline)^2
                  1.0 = perfect, 0 = no better than the baseline, < 0 = worse than it
  corr        - correlation between predicted and true values (-1 .. 1);
                  shows whether predictions move with the truth, even if offset or scaled
Also saves a predicted-vs-true scatter plot per folder.

Works with checkpoints from train_rnn.py or train_zw.py.

Usage:
    python evaluate.py --checkpoint lstm64.pt \
        --data_dir tags_to_imu_data/window_data/val tags_to_imu_data/window_data/test
"""

import argparse
import glob
import os

import numpy as np
import torch

from train_rnn import fill_inputs, load_csv, load_model


@torch.no_grad()
def predict_folder(model, stats, folder, device):
    files = sorted(glob.glob(os.path.join(os.path.expanduser(folder), "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found in {folder}")
    preds, trues = [], []
    for f in files:
        try:
            x, y, _, _ = load_csv(f, stats["in_cols"], stats["out_cols"])
        except ValueError as e:
            print(f"  skipping {e}")
            continue
        xt = torch.from_numpy(fill_inputs((x - stats["x_mean"]) / stats["x_std"])).unsqueeze(0).to(device)
        p = model(xt).squeeze(0).cpu().numpy() * stats["y_std"] + stats["y_mean"]
        preds.append(p); trues.append(y)
    return np.concatenate(preds), np.concatenate(trues), len(preds)


def score(pred, true, train_mean):
    rows = []
    for i in range(true.shape[1]):
        ok = ~np.isnan(true[:, i])
        p, t = pred[ok, i], true[ok, i]
        rmse = np.sqrt(np.mean((p - t) ** 2))
        base = np.sqrt(np.mean((train_mean[i] - t) ** 2))
        skill = 1 - (rmse / base) ** 2 if base > 0 else np.nan
        corr = np.corrcoef(p, t)[0, 1] if p.std() > 0 and t.std() > 0 else np.nan
        rows.append((rmse, base, skill, corr))
    return rows


def verdict(skill):
    if np.isnan(skill):
        return ""
    if skill >= 0.75:
        return "good"
    if skill >= 0.5:
        return "decent"
    if skill >= 0.2:
        return "weak"
    if skill >= 0:
        return "barely useful"
    return "worse than baseline"


def scatter_plot(pred, true, names, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    n = len(names)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        if i >= n:
            ax.axis("off")
            continue
        ok = ~np.isnan(true[:, i])
        t, p = true[ok, i], pred[ok, i]
        if len(t) > 20000:  # keep the plot light
            idx = np.random.default_rng(0).choice(len(t), 20000, replace=False)
            t, p = t[idx], p[idx]
        ax.scatter(t, p, s=2, alpha=0.25)
        lo, hi = np.nanmin([t.min(), p.min()]), np.nanmax([t.max(), p.max()])
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(names[i].replace("mavros/imu/data.", "").replace("bluerov/", ""), fontsize=9)
        ax.set_xlabel("true", fontsize=8); ax.set_ylabel("predicted", fontsize=8)
    fig.suptitle(f"{title}: predicted vs true (dashed = perfect)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  scatter plot saved to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", nargs="+", required=True, help="one or more folders of window CSVs")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, stats = load_model(args.checkpoint, device)
    names = stats["out_cols"]
    ckpt_base = os.path.splitext(os.path.basename(args.checkpoint))[0]

    for folder in args.data_dir:
        label = os.path.basename(os.path.normpath(folder))
        pred, true, n_files = predict_folder(model, stats, folder, device)
        rows = score(pred, true, stats["y_mean"])
        print(f"\n=== {label} ({n_files} files) ===")
        print(f"  {'output':34s}{'RMSE':>9s}{'baseline':>10s}{'skill':>8s}{'corr':>8s}   verdict")
        for name, (rmse, base, skill, corr) in zip(names, rows):
            short = name.replace("mavros/imu/data.", "")
            print(f"  {short:34s}{rmse:9.4f}{base:10.4f}{skill:8.2f}{corr:8.2f}   {verdict(skill)}")
        scatter_plot(pred, true, names, f"{ckpt_base}_{label}_scatter.png", f"{ckpt_base} on {label}")


if __name__ == "__main__":
    main()
