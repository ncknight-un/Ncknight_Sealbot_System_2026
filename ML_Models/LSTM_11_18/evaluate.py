"""
How good is a trained train_lstm_flow.py model? Scores each classification head (wall test: wall_side, dist_to_wall; angle test: angle_to_head, dist_to_head) on whole
recordings, predicted the way the model was trained: every recording is cut into consecutive,
non-overlapping segments of the checkpoint's seg_len frames and each segment is classified on its own
(train_lstm_flow.predict_file).

Per set, for each head:
  accuracy, balanced accuracy (the mean of each class's recall, so rare classes count equally),
  the baseline of always guessing the most common training class, and a confusion matrix
  (counted per segment).
Per recording: the share of its segments predicted right and the class the recording gets when its
segments vote (mean probability), worst first, so problems can be traced to files.
With --plots: a time plot per recording (see plot_predictions.py).

  --vote N            also score each segment from the probabilities averaged over the last N segments
                      of its recording (past only, as it could run live), e.g. --vote 4 with 16-frame
                      segments = roughly the last second
  --trim_start S      leave out segments that start in the first S seconds of every recording
  --trim_end S        leave out segments that end in the last S seconds
                      (e.g. to see whether errors cluster where the ROV starts and stops; the left-out
                      stretches are greyed in the --plots time plots)

Which recordings:
  python evaluate.py --checkpoint models_flow/run.pt                       the run's test recordings
  python evaluate.py --checkpoint models_flow/run.pt --split val test      several sets of its split
  python evaluate.py --checkpoint models_flow/run.pt --data_dir some/folder   every CSV in a folder
  python evaluate.py --checkpoint models_flow/run.pt --files a.csv b.csv
Options: --plots  --vote 4  --out_dir DIR
"""

import argparse
import glob
import json
import os

import numpy as np
import torch

import train_lstm_flow as tr


def trailing_mean(P, n):
    """Average of the last n rows at each row (past only)."""
    c = np.cumsum(np.vstack([np.zeros((1, P.shape[1])), P]), axis=0)
    idx = np.arange(1, len(P) + 1)
    lo = np.maximum(idx - n, 0)
    return (c[idx] - c[lo]) / (idx - lo)[:, None]


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
        raise SystemExit(f"{sp} not found: give --data_dir or --files")
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
    ap.add_argument("--vote", type=int, default=0, metavar="N",
                    help="also score class probabilities averaged over the last N segments")
    ap.add_argument("--trim_start", type=float, default=0.0, metavar="S", help="leave out the first S seconds")
    ap.add_argument("--trim_end", type=float, default=0.0, metavar="S", help="leave out the last S seconds")
    ap.add_argument("--out_dir", help="where plots go (default <checkpoint>_eval/)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, stats = tr.load_model(args.checkpoint, device)
    heads, L = stats["heads"], int(stats["seg_len"])
    out_dir = args.out_dir or f"{os.path.splitext(args.checkpoint)[0]}_eval"
    if args.plots:
        os.makedirs(out_dir, exist_ok=True)
    print(f"Model: {args.checkpoint}  (segments of {L} frames = {L / tr.FPS:.2f} s, no overlap; inputs: "
          f"{' + '.join(stats.get('inputs', ['?']))}; {stats.get('test', '?')} test)"
          + (f". Leaving out the first {args.trim_start:g} s / last {args.trim_end:g} s of every recording"
             if args.trim_start or args.trim_end else ""))

    for label, files in file_sets(args, args.checkpoint):
        cms = {h["name"]: np.zeros((len(h["classes"]),) * 2, np.int64) for h in heads}
        cms_v = {h["name"]: np.zeros((len(h["classes"]),) * 2, np.int64) for h in heads}
        per_rec = []
        for f in files:
            try:
                df = tr.predict_file(model, stats, f, device)
            except ValueError as e:
                print(f"  skipping {e}")
                continue
            name = os.path.splitext(os.path.basename(f))[0]
            t0, t1 = df["t_start"].iloc[0], df["t_end"].iloc[-1]
            keep = ((df["t_start"] >= t0 + args.trim_start - 1e-9)
                    & (df["t_end"] <= t1 - args.trim_end + 1e-9)).to_numpy()
            if not keep.any():
                print(f"  {name}: nothing left after trimming - skipped")
                continue
            parts, worst = [], 1.0
            for h in heads:
                names = h["names"]
                tl = df[f"true:{h['name']}"].to_numpy()
                lab = keep & (tl != "")
                if not lab.any():
                    parts.append(f"{h['name']}: (no label)")
                    continue
                P = df[[tr.prob_col(h["name"], n) for n in names]].to_numpy(float)
                ti = np.array([names.index(x) for x in tl[lab]])
                pi = P.argmax(1)[lab]
                np.add.at(cms[h["name"]], (ti, pi), 1)
                if args.vote > 1:
                    np.add.at(cms_v[h["name"]], (ti, trailing_mean(P, args.vote).argmax(1)[lab]), 1)
                acc = float(np.mean(pi == ti))
                worst = min(worst, acc)
                parts.append(f"{h['name']}: {names[np.bincount(ti).argmax()]} {acc:6.1%} of {len(ti)} segments, "
                             f"vote -> {names[int(P[lab].mean(0).argmax())]}")
            per_rec.append((worst, name, parts))
            if args.plots:
                from plot_predictions import plot_recording
                plot_recording(df, stats, name, os.path.join(out_dir, f"{label}_{name}.png"), args.vote,
                               excluded=(args.trim_start, args.trim_end))

        print(f"\n=== {label} ({len(per_rec)} recordings) ===")
        for h in heads:
            cm, names = cms[h["name"]], h["names"]
            n = cm.sum()
            if not n:
                print(f"\n  {h['name']}: no labelled segments")
                continue
            maj = int(np.argmax(h.get("train_counts") or cm.sum(1)))
            print(f"\n  {h['name']}: accuracy {np.trace(cm) / n:.1%}, balanced {tr.balanced_acc(cm):.1%}   "
                  f"(baseline: always '{names[maj]}' = {cm[maj].sum() / n:.1%}, balanced "
                  f"{(1.0 if cm[maj].sum() else 0.0) / max((cm.sum(1) > 0).sum(), 1):.1%})"
                  + (f"   voting over the last {args.vote} segments: {np.trace(cms_v[h['name']]) / n:.1%}, "
                     f"balanced {tr.balanced_acc(cms_v[h['name']]):.1%}" if args.vote > 1 else ""))
            w = max([len(a) for a in names] + [8]) + 2
            print("  " + " " * w + "".join(f"{a:>{w}s}" for a in names)
                  + "    recall   (rows = true, columns = predicted, per segment)")
            for k, a in enumerate(names):
                rec = f"{cm[k, k] / cm[k].sum():6.1%}" if cm[k].sum() else "     -"
                print(f"  {a:{w}s}" + "".join(f"{v:{w}d}" for v in cm[k]) + f"    {rec}")
        print("\n  Per recording (worst first):")
        for _, name, parts in sorted(per_rec, key=lambda x: x[0]):
            print(f"    {name}\n        " + "   |   ".join(parts))
        if args.plots:
            print(f"\n  time plots saved to {out_dir}/{label}_<recording>.png")


if __name__ == "__main__":
    main()
