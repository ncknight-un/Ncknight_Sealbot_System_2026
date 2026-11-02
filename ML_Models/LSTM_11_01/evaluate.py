"""
How good is a trained model? Scores each target against a do-nothing baseline, on whole recordings,
predicted the way the model runs live (at every frame it is fed the last `window` frames and only
the newest prediction is kept: train_rnn.predict_file).

Regression targets (IMU, depth, turn_rate_avg), per set:
  RMSE        the model's error, in original units (turn rate also in deg/s)
  baseline    error of always predicting the training average (no model at all)
  skill       1 - (RMSE / baseline)^2: 1.0 = perfect, 0 = no better than the baseline, < 0 = worse
  corr        correlation between predicted and true values (-1 .. 1)
  + a predicted-vs-true scatter plot per set

Classification targets (action, diam), per set:
  accuracy, balanced accuracy (the mean of each class's recall, so rare classes count equally),
  the baseline of always guessing the most common training class, and a confusion matrix.

Per recording: each target's score, worst first, so problems can be traced to files.
With --plots: a time plot per recording (true vs predicted over the whole recording; see
plot_predictions.py).

The first `warmup` frames of every recording (less history than the model was trained to use) are
left out of the scores, as in training; --all_frames scores them too.

Leaving out the start and end of every recording (e.g. to see whether errors cluster where the ROV
starts and stops):
  --trim N            leave out N windows' worth of frames at both ends (e.g. --trim 1 with a 2 s window
                      = the first and last 2 s)
  --trim_start S      leave out the first S seconds; --trim_end S the last S seconds (override --trim)
The left-out stretches are greyed in the --plots time plots. Run once with and once without to compare.

Which recordings:
  python evaluate.py --checkpoint models/run.pt                       the run's test recordings
  python evaluate.py --checkpoint models/run.pt --split val test      several sets of its split
  python evaluate.py --checkpoint models/run.pt --data_dir some/folder   every CSV in a folder
  python evaluate.py --checkpoint models/run.pt --files a.csv b.csv
Options: --plots  --smooth 1.0  --out_dir DIR
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import torch

import train_rnn as tr

FPS = 65.0


def score_regression(pred, true, train_mean):
    rows = []
    for i in range(true.shape[1]):
        ok = ~np.isnan(true[:, i]) & ~np.isnan(pred[:, i])
        p, t = pred[ok, i], true[ok, i]
        if not len(t):
            rows.append((np.nan,) * 4)
            continue
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


def balanced(cm):
    rows = cm.sum(1)
    return float(np.mean(np.diag(cm)[rows > 0] / rows[rows > 0])) if rows.sum() else np.nan


def scatter_plot(pred, true, names, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    n = len(names)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows), squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        if i >= n:
            ax.axis("off")
            continue
        ok = ~np.isnan(true[:, i]) & ~np.isnan(pred[:, i])
        t, p = true[ok, i], pred[ok, i]
        if "turn_rate" in names[i]:
            t, p = np.degrees(t), np.degrees(p)
        if len(t) > 20000:  # keep the plot light
            idx = np.random.default_rng(0).choice(len(t), 20000, replace=False)
            t, p = t[idx], p[idx]
        if not len(t):
            continue
        ax.scatter(t, p, s=2, alpha=0.25)
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        unit = " [deg/s]" if "turn_rate" in names[i] else ""
        ax.set_title(tr.short_name(names[i]) + unit, fontsize=9)
        ax.set_xlabel("true", fontsize=8)
        ax.set_ylabel("predicted", fontsize=8)
    fig.suptitle(f"{title}: predicted vs true (dashed = perfect)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  scatter plot saved to {path}")


def file_sets(args, ckpt):
    """[(label, [csv paths])] from --files, --data_dir or the checkpoint's saved split."""
    if args.files:
        return [("files", args.files)]
    if args.data_dir:
        out = []
        for folder in args.data_dir:
            fs = sorted(glob.glob(os.path.join(os.path.expanduser(folder), "**", "*.csv"), recursive=True))
            if not fs:
                raise SystemExit(f"No CSV files found in {folder}")
            out.append((os.path.basename(os.path.normpath(folder)), fs))
        return out
    sp = f"{os.path.splitext(ckpt)[0]}_split.json"
    if not os.path.isfile(sp):
        raise SystemExit(f"{sp} not found: give --data_dir or --files, or a checkpoint trained with "
                         "--data_dir / --split_file")
    info = json.load(open(sp))
    return [(s, [os.path.join(info["data_dir"], r) for r in info["splits"][s]]) for s in args.split]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", nargs="+", default=["test"], choices=["train", "val", "test"],
                    help="which sets of the run's saved split to score (default test)")
    ap.add_argument("--data_dir", nargs="+", help="score every CSV in these folders instead")
    ap.add_argument("--files", nargs="+", help="score these CSVs instead")
    ap.add_argument("--plots", action="store_true", help="also save a time plot per recording")
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="also score (and plot) class probabilities averaged over the last this-many seconds")
    ap.add_argument("--all_frames", action="store_true",
                    help="also score each recording's first `warmup` frames (left out by default, as in training)")
    ap.add_argument("--trim", type=float, default=0.0, metavar="N",
                    help="leave out N windows' worth of frames at the start and at the end of every recording")
    ap.add_argument("--trim_start", type=float, metavar="S", help="leave out the first S seconds (overrides --trim)")
    ap.add_argument("--trim_end", type=float, metavar="S", help="leave out the last S seconds (overrides --trim)")
    ap.add_argument("--out_dir", help="where plots go (default <checkpoint>_eval/)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, stats = tr.load_model(args.checkpoint, device)
    regs, heads = list(stats["out_cols"]), stats["heads"]
    L = int(stats["seq_len"])
    cut_start = int(round(args.trim_start * FPS)) if args.trim_start is not None else int(round(args.trim * L))
    cut_end = int(round(args.trim_end * FPS)) if args.trim_end is not None else int(round(args.trim * L))
    skip = max(cut_start, 0 if args.all_frames else int(stats.get("warmup", 0)))
    out_dir = args.out_dir or f"{os.path.splitext(args.checkpoint)[0]}_eval"
    os.makedirs(out_dir, exist_ok=True)
    n_s = max(1, int(round(args.smooth * FPS)))
    print(f"Model: {args.checkpoint}  (window {L} frames = {L / FPS:.2f} s). Scoring each recording from frame "
          f"{skip} ({skip / FPS:.2f} s)" + (f" up to its last {cut_end} frames ({cut_end / FPS:.2f} s) left out"
                                           if cut_end else " to its end"))

    for label, files in file_sets(args, args.checkpoint):
        preds, trues = [], []
        cms = {h["name"]: np.zeros((len(h["classes"]),) * 2, np.int64) for h in heads}
        cms_s = {h["name"]: np.zeros((len(h["classes"]),) * 2, np.int64) for h in heads}
        per_rec = []
        for f in files:
            try:
                df = tr.predict_file(model, stats, f, device)
            except ValueError as e:
                print(f"  skipping {e}")
                continue
            name = os.path.splitext(os.path.basename(f))[0]
            end = len(df) - cut_end
            if end - skip < 1:
                print(f"  {name}: only {len(df)} frames, nothing left after trimming - skipped")
                continue
            d = df.iloc[skip:end]
            parts, worst = [], 1.0
            if regs:
                p = d[regs].to_numpy(float)
                t = d[[f"true:{c}" for c in regs]].to_numpy(float)
                preds.append(p)
                trues.append(t)
                for i, c in enumerate(regs):
                    ok = ~np.isnan(t[:, i])
                    if ok.any() and "turn_rate" in c:
                        parts.append(f"turn rate true {np.degrees(t[ok, i].mean()):+.1f} / predicted "
                                     f"{np.degrees(p[ok, i].mean()):+.1f} deg/s")
            for h in heads:
                names = h["names"]
                tl = d[f"true:{h['name']}"].to_numpy()
                lab = tl != ""
                if not lab.any():
                    parts.append(f"{h['name']}: (no label)")
                    continue
                P = df[[f"p_{h['name']}_{n.replace(' ', '')}" for n in names]].to_numpy(float)
                pi = P.argmax(1)[skip:end]
                ti = np.array([names.index(x) for x in tl[lab]])
                np.add.at(cms[h["name"]], (ti, pi[lab]), 1)
                if args.smooth > 0:
                    pis = _trailing(P, n_s).argmax(1)[skip:end]
                    np.add.at(cms_s[h["name"]], (ti, pis[lab]), 1)
                acc = float(np.mean(pi[lab] == ti))
                worst = min(worst, acc)
                main_true = names[np.bincount(ti).argmax()]
                parts.append(f"{h['name']}: {main_true} {acc:6.1%}")
            per_rec.append((worst, name, parts))
            if args.plots:
                from plot_predictions import plot_recording
                plot_recording(df, stats, name, os.path.join(out_dir, f"{label}_{name}.png"), args.smooth,
                               excluded=(skip, cut_end))

        print(f"\n=== {label} ({len(per_rec)} recordings) ===")
        if regs and preds:
            pred, true = np.concatenate(preds), np.concatenate(trues)
            print(f"  {'output':46s}{'RMSE':>9s}{'baseline':>10s}{'skill':>8s}{'corr':>8s}   verdict")
            for c, (rmse, base, skill, corr) in zip(regs, score_regression(pred, true, stats["y_mean"])):
                extra = f"   ({np.degrees(rmse):.1f} deg/s)" if "turn_rate" in c else ""
                print(f"  {tr.short_name(c):46s}{rmse:9.4f}{base:10.4f}{skill:8.2f}{corr:8.2f}   {verdict(skill)}{extra}")
            scatter_plot(pred, true, regs, os.path.join(out_dir, f"{label}_scatter.png"), f"{label}")
        for h in heads:
            cm = cms[h["name"]]
            names = h["names"]
            n = cm.sum()
            if not n:
                continue
            counts = np.array(h.get("train_counts") or cm.sum(1), float)
            maj = int(counts.argmax())
            base_acc = cm[maj].sum() / n
            print(f"\n  {h['name']}: accuracy {np.trace(cm) / n:.1%}, balanced {balanced(cm):.1%}   "
                  f"(baseline: always '{names[maj]}' = {base_acc:.1%}, balanced "
                  f"{(1.0 if cm[maj].sum() else 0.0) / max((cm.sum(1) > 0).sum(), 1):.1%})"
                  + (f"   smoothed over {args.smooth:g} s: {np.trace(cms_s[h['name']]) / n:.1%}, "
                     f"balanced {balanced(cms_s[h['name']]):.1%}" if args.smooth > 0 else ""))
            print("  " + " " * 12 + "".join(f"{a:>10s}" for a in names) + "    recall   (rows = true, columns = predicted)")
            for k, a in enumerate(names):
                rec = f"{cm[k, k] / cm[k].sum():6.1%}" if cm[k].sum() else "     -"
                print(f"  {a:12s}" + "".join(f"{v:10d}" for v in cm[k]) + f"    {rec}")
        print(f"\n  Per recording (worst first):")
        for worst, name, parts in sorted(per_rec, key=lambda x: x[0]):
            print(f"    {name}\n        " + "   |   ".join(parts))
        if args.plots:
            print(f"\n  time plots saved to {out_dir}/{label}_<recording>.png")


def _trailing(P, n):
    """Average of the last n rows at each row (past only)."""
    c = np.cumsum(np.vstack([np.zeros((1, P.shape[1])), P]), axis=0)
    idx = np.arange(1, len(P) + 1)
    lo = np.maximum(idx - n, 0)
    return (c[idx] - c[lo]) / (idx - lo)[:, None]


if __name__ == "__main__":
    main()
