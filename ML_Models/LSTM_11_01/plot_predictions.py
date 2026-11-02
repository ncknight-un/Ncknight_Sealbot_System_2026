#!/usr/bin/env python3
"""
plot_predictions.py - true vs predicted over the whole of each recording, the way the model runs live
(at every frame it is fed the last `window` frames and only the newest prediction is kept).

For each recording, one PNG with a panel per target:
  regression (e.g. turn_rate_avg): true and predicted lines (turn rate shown in deg/s)
  classification (action, diam):   the probability of each class over time, the true class shaded
                                   behind, and the predicted class as a strip along the bottom
and a summary line printed per recording: mean true vs mean predicted turn rate, and the share of
frames each class head got right.

Usage:
  python plot_predictions.py models/run.pt                      # the run's test recordings
  python plot_predictions.py models/run.pt --split val          # its val recordings
  python plot_predictions.py models/run.pt --files data/a.csv data/b.csv
  options: --out_dir plots/  --smooth 1.0  (also draw class probabilities averaged over the last 1 s)
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_rnn as tr  # noqa: E402

FPS = 65.0


def trailing_mean(x, n):
    """Average of the last n rows at each row (past only, as it could run live)."""
    c = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), axis=0)
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(idx - n, 0)
    return (c[idx] - c[lo]) / (idx - lo)[:, None]


def plot_recording(df, stats, name, png, smooth=0.0, excluded=(0, 0)):
    """One PNG for one recording from predict_file's DataFrame. Returns summary lines.
    excluded = (frames at the start, frames at the end) left out of the scores: greyed in every panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    heads, regs = stats["heads"], list(stats["out_cols"])
    t = df["t_s"].to_numpy() if "t_s" in df else np.arange(len(df)) / FPS
    n_pan = max(len(regs) + len(heads), 1)
    fig, axes = plt.subplots(n_pan, 1, figsize=(12, 2.8 * n_pan), sharex=True, squeeze=False)
    axes = axes[:, 0]
    summary = []
    for i, c in enumerate(regs):
        ax = axes[i]
        pred, true = df[c].to_numpy(float), df[f"true:{c}"].to_numpy(float)
        scale, unit = (np.degrees(1.0), "deg/s") if "turn_rate" in c else (1.0, "")
        ax.plot(t, true * scale, color="black", lw=1.2, label="true")
        ax.plot(t, pred * scale, color="tab:orange", lw=1.0, label="predicted")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_ylabel(f"{tr.short_name(c).split('.')[-1]}\n{unit}")
        ax.legend(loc="upper right", fontsize=8)
        ok = np.isfinite(true)
        if ok.any():
            rmse = np.sqrt(np.mean((pred[ok] - true[ok]) ** 2)) * scale
            summary.append(f"{tr.short_name(c).split('.')[-1]}: true {np.mean(true[ok]) * scale:+.1f}, "
                           f"predicted {np.mean(pred[ok]) * scale:+.1f} {unit} (RMSE {rmse:.1f})")
    n_s = max(1, int(round(smooth * FPS)))
    for j, h in enumerate(heads):
        ax = axes[len(regs) + j]
        names = h["names"]
        P = df[[f"p_{h['name']}_{n.replace(' ', '')}" for n in names]].to_numpy(float)
        colors = plt.cm.tab10(np.arange(len(names)))
        true_lab = df[f"true:{h['name']}"].to_numpy()
        for k, n in enumerate(names):                         # true class shaded behind
            m = true_lab == n
            if m.any():
                ax.fill_between(t, 0, 1, where=m, color=colors[k], alpha=0.12, step="mid", lw=0)
        for k, n in enumerate(names):
            ax.plot(t, P[:, k], color=colors[k], lw=0.8, label=n)
        if smooth > 0:
            Ps = trailing_mean(P, n_s)
            for k in range(len(names)):
                ax.plot(t, Ps[:, k], color=colors[k], lw=2.0, alpha=0.8)
        pred_idx = P.argmax(1)                                # predicted class strip along the bottom
        ax.scatter(t, np.full(len(t), -0.06), c=colors[pred_idx], s=4, marker="s", lw=0)
        ax.set_ylim(-0.12, 1.05)
        ax.set_ylabel(f"{h['name']}\nprobability")
        ax.legend(loc="upper right", fontsize=8, ncol=len(names))
        lab = true_lab != ""
        if lab.any():
            acc = np.mean(np.array(names)[pred_idx][lab] == true_lab[lab])
            main_true = max(set(true_lab[lab]), key=list(true_lab[lab]).count)
            txt = f"{h['name']}: true {main_true}, {acc:.0%} of frames right"
            if smooth > 0:
                acc_s = np.mean(np.array(names)[trailing_mean(P, n_s).argmax(1)][lab] == true_lab[lab])
                txt += f" (smoothed {acc_s:.0%})"
            summary.append(txt)
    a0, a1 = excluded
    for ax in axes:
        if a0 > 0:
            ax.axvspan(t[0], t[min(a0, len(t) - 1)], color="grey", alpha=0.25, lw=0)
        if a1 > 0:
            ax.axvspan(t[max(len(t) - a1, 0)], t[-1], color="grey", alpha=0.25, lw=0)
    axes[-1].set_xlabel("time in recording [s]" + ("   (grey: left out of the scores)" if a0 or a1 else ""))
    fig.suptitle(name + ("   (thin: per frame, thick: averaged over the last %g s)" % smooth if smooth > 0 else ""),
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test",
                    help="which recordings of the run's saved split to plot (default test)")
    ap.add_argument("--files", nargs="+", help="plot these CSVs instead of a split")
    ap.add_argument("--out_dir", help="where to put the PNGs (default <checkpoint>_plots/)")
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="also draw class probabilities averaged over the last this-many seconds (e.g. 1.0)")
    a = ap.parse_args()

    model, stats = tr.load_model(a.checkpoint)
    base = os.path.splitext(a.checkpoint)[0]
    if a.files:
        files = a.files
    else:
        sp = f"{base}_split.json"
        if not os.path.isfile(sp):
            raise SystemExit(f"{sp} not found: give --files, or a checkpoint trained with --data_dir / --split_file")
        info = json.load(open(sp))
        files = [os.path.join(info["data_dir"], r) for r in info["splits"][a.split]]
    out_dir = a.out_dir or f"{base}_plots"
    os.makedirs(out_dir, exist_ok=True)
    print(f"{len(files)} recording(s), window {stats['seq_len']} frames; plots -> {out_dir}/\n")

    for f in files:
        df = tr.predict_file(model, stats, f)
        name = os.path.splitext(os.path.basename(f))[0]
        summary = plot_recording(df, stats, name, os.path.join(out_dir, f"{name}.png"), a.smooth)
        print(f"{name}\n    " + "\n    ".join(summary))


if __name__ == "__main__":
    main()
