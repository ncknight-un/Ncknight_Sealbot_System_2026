#!/usr/bin/env python3
"""
plot_predictions.py - true vs predicted class over the whole of each recording for a train_lstm_flow.py
model. The recording is cut into consecutive, non-overlapping segments of the checkpoint's seg_len
frames and each segment is classified on its own, exactly as in training.

For each recording, one PNG with a panel per head (wall test: wall_side, dist_to_wall; angle test: angle_to_head, dist_to_head): the probability of each class as a
step line (one step per segment), the true class shaded behind, and the predicted class as a strip
along the bottom. A summary line is printed per recording: the share of segments each head got right
and the class the recording gets when its segments vote.

Usage:
  python plot_predictions.py models_flow/run.pt                      # the run's test recordings
  python plot_predictions.py models_flow/run.pt --split val          # its val recordings
  python plot_predictions.py models_flow/run.pt --files data/a.csv data/b.csv
  options: --out_dir plots/  --vote 4  (also draw class probabilities averaged over the last 4 segments)
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_lstm_flow as tr  # noqa: E402


def trailing_mean(x, n):
    """Average of the last n rows at each row (past only, as it could run live)."""
    c = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), axis=0)
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(idx - n, 0)
    return (c[idx] - c[lo]) / (idx - lo)[:, None]


def steps(t0, t1, y):
    """Segment values -> x, y arrays that draw each value flat across its own segment."""
    return np.column_stack([t0, t1]).ravel(), np.repeat(y, 2)


def plot_recording(df, stats, name, png, vote=0, excluded=(0.0, 0.0)):
    """One PNG for one recording from predict_file's DataFrame. Returns summary lines.
    excluded = (seconds at the start, seconds at the end) left out of the scores: greyed in every panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    heads = stats["heads"]
    t0, t1 = df["t_start"].to_numpy(float), df["t_end"].to_numpy(float)
    mid = (t0 + t1) / 2
    fig, axes = plt.subplots(len(heads), 1, figsize=(12, 2.8 * len(heads)), sharex=True, squeeze=False)
    axes = axes[:, 0]
    summary = []
    for ax, h in zip(axes, heads):
        names = h["names"]
        P = df[[tr.prob_col(h["name"], n) for n in names]].to_numpy(float)
        colors = plt.cm.tab10(np.arange(len(names)) % 10)
        true_lab = df[f"true:{h['name']}"].to_numpy()
        for k, n in enumerate(names):                         # true class shaded behind
            for a, b in zip(t0[true_lab == n], t1[true_lab == n]):
                ax.axvspan(a, b, color=colors[k], alpha=0.12, lw=0)
        for k, n in enumerate(names):
            ax.plot(*steps(t0, t1, P[:, k]), color=colors[k], lw=0.9, label=n)
        if vote > 1:
            Pv = trailing_mean(P, vote)
            for k in range(len(names)):
                ax.plot(*steps(t0, t1, Pv[:, k]), color=colors[k], lw=2.2, alpha=0.75)
        pred_idx = P.argmax(1)                                # predicted class strip along the bottom
        ax.scatter(mid, np.full(len(mid), -0.06), c=colors[pred_idx], s=10, marker="s", lw=0)
        ax.set_ylim(-0.12, 1.05)
        ax.set_ylabel(f"{h['name']}\nprobability")
        ax.legend(loc="upper right", fontsize=8, ncol=min(len(names), 8))
        lab = true_lab != ""
        if lab.any():
            acc = np.mean(np.array(names)[pred_idx][lab] == true_lab[lab])
            main_true = max(set(true_lab[lab]), key=list(true_lab[lab]).count)
            txt = (f"{h['name']}: true {main_true}, {acc:.0%} of {int(lab.sum())} segments right, "
                   f"vote -> {names[int(P[lab].mean(0).argmax())]}")
            if vote > 1:
                acc_v = np.mean(np.array(names)[trailing_mean(P, vote).argmax(1)][lab] == true_lab[lab])
                txt += f" (last-{vote} average {acc_v:.0%})"
            summary.append(txt)
    a0, a1 = excluded
    for ax in axes:
        if a0 > 0:
            ax.axvspan(t0[0], t0[0] + a0, color="grey", alpha=0.25, lw=0)
        if a1 > 0:
            ax.axvspan(t1[-1] - a1, t1[-1], color="grey", alpha=0.25, lw=0)
    axes[-1].set_xlabel("time in recording [s]" + ("   (grey: left out of the scores)" if a0 or a1 else ""))
    L = int(stats["seg_len"])
    fig.suptitle(f"{name}   ({L}-frame segments"
                 + (f"; thin: per segment, thick: averaged over the last {vote} segments)" if vote > 1 else ")"),
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
    ap.add_argument("--vote", type=int, default=0, metavar="N",
                    help="also draw class probabilities averaged over the last N segments (e.g. 4)")
    a = ap.parse_args()

    model, stats = tr.load_model(a.checkpoint)
    base = os.path.splitext(a.checkpoint)[0]
    if a.files:
        files = a.files
    else:
        sp = f"{base}_split.json"
        if not os.path.isfile(sp):
            raise SystemExit(f"{sp} not found: give --files")
        info = json.load(open(sp))
        files = [os.path.join(info["data_dir"], r) for r in info["splits"][a.split]]
    out_dir = a.out_dir or f"{base}_plots"
    os.makedirs(out_dir, exist_ok=True)
    print(f"{len(files)} recording(s), segments of {stats['seg_len']} frames; plots -> {out_dir}/\n")

    for f in files:
        try:
            df = tr.predict_file(model, stats, f)
        except ValueError as e:
            print(f"skipping {e}")
            continue
        name = os.path.splitext(os.path.basename(f))[0]
        summary = plot_recording(df, stats, name, os.path.join(out_dir, f"{name}.png"), a.vote)
        print(f"{name}\n    " + "\n    ".join(summary))


if __name__ == "__main__":
    main()
