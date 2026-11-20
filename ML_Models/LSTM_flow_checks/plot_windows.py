#!/usr/bin/env python3
"""
plot_windows.py - per-class accuracy (recall) against window length, from run_checks.sh results.

Reads results_<DATA>_<SEG>/<model>_eval.txt for every window length given, takes the confusion matrix of
the chosen split, and plots each class's recall (and the balanced accuracy) against the window length,
one panel per model. Also writes the same numbers as a CSV table and prints them.

    python plot_windows.py data_11_17 --windows 4 8 16 32            # test recordings (default)
    python plot_windows.py data_11_17 --windows 4 8 16 32 --split val test   # val + test pooled

Figures (next to the results folders; models without results are left out):
  windows_<DATA>_<split>.png          each class's recall vs window length, one panel per model
  windows_<DATA>_<split>_summary.png  balanced accuracy vs window length, every model in one plot
  confusion_<DATA>_<split>_<W>.png    with --confusion W: each model's confusion matrix at window W,
                                      rows normalised (each row = 100 % of that true class)
and windows_<DATA>_<split>.csv with the numbers.
Val was used to pick each model's best epoch, so val scores are slightly optimistic; test is the
fair number.
"""
import argparse
import csv
import os
import re
import sys

import numpy as np

MODELS = [("diff_whiskers", "All whisker motion (diff)"),
          ("cmdiff", "Relative to the array (cmdiff)"),
          ("cmdiff_panel", "Relative to own panel (cmdiff_panel)"),
          ("cmonly_panel", "Each panel's motion (cmonly_panel)"),
          ("cmonly", "Whole-array motion (cmonly)")]
FPS = 65.0
# reference palette, light mode (validated: CVD-safe for 6 adjacent slots); markers give a second channel
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def read_confusion(path, split):
    """{class: [counts...]} and class order from the '=== <split>' block of an evaluate.py log."""
    lines = open(path).read().splitlines()
    try:
        i = next(k for k, l in enumerate(lines) if l.strip().startswith(f"=== {split} "))
    except StopIteration:
        return None, None
    names, rows = None, {}
    for l in lines[i + 1:]:
        if l.strip().startswith("===") or "Per recording" in l:
            break
        if names is None and "recall" in l and "rows = true" in l:
            names = l.split("recall")[0].split()
            continue
        if names:
            t = l.split()
            if len(t) >= len(names) + 1 and t[0] in names and all(re.fullmatch(r"\d+", x) for x in t[1:len(names) + 1]):
                rows[t[0]] = [int(x) for x in t[1:len(names) + 1]]
    return (names, rows) if names and rows else (None, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", help="data folder name used with run_checks.sh, e.g. data_11_17")
    ap.add_argument("--windows", nargs="+", type=int, default=[4, 8, 16, 32])
    ap.add_argument("--split", nargs="+", default=["test"], choices=["val", "test"])
    ap.add_argument("--confusion", type=int, metavar="W", help="also plot every model's confusion matrix at window W")
    a = ap.parse_args()
    data = os.path.basename(a.data.rstrip("/"))

    table, classes = {}, None                      # (model, window) -> {class: recall, 'balanced': ...}
    for m, _ in MODELS:
        for w in a.windows:
            path = f"results_{data}_{w}/{m}_eval.txt"
            if not os.path.isfile(path):
                continue
            total = None
            for s in a.split:
                names, rows = read_confusion(path, s)
                if names is None:
                    print(f"no '{s}' confusion matrix in {path}")
                    continue
                cm = np.array([rows.get(n, [0] * len(names)) for n in names])
                total = cm if total is None else total + cm
                classes = classes or names
            if total is None:
                continue
            rec = {n: (total[k, k] / total[k].sum() if total[k].sum() else np.nan) for k, n in enumerate(classes)}
            rec["balanced"] = float(np.nanmean([rec[n] for n in classes]))
            rec["segments"] = int(total.sum())
            rec["cm"] = total
            table[(m, w)] = rec
    if not table:
        print("nothing to plot")
        return 1

    tag = "+".join(a.split)
    out_csv = f"windows_{data}_{tag}.csv"
    with open(out_csv, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["model", "window_frames", "window_ms", "segments", "balanced"] + classes)
        for (m, w), r in sorted(table.items(), key=lambda x: ([k for k, _ in MODELS].index(x[0][0]), x[0][1])):
            wr.writerow([m, w, round(1000 * w / FPS), r["segments"], f"{100 * r['balanced']:.1f}"]
                        + [f"{100 * r[c]:.1f}" for c in classes])
    print(f"Recall per class (%), {tag} recordings; chance = {100 / len(classes):.1f}%")
    print(f"  {'model':14s} {'window':>11s} {'balanced':>9s} " + " ".join(f"{c:>6s}" for c in classes))
    for (m, w), r in sorted(table.items(), key=lambda x: ([k for k, _ in MODELS].index(x[0][0]), x[0][1])):
        print(f"  {m:14s} {w:3d} ({1000 * w / FPS:4.0f} ms) {100 * r['balanced']:8.1f} "
              + " ".join(f"{100 * r[c]:6.1f}" for c in classes))

    have = [(m, t) for m, t in MODELS if any((m, w) in table for w in a.windows)]
    for m, _ in have:
        miss = [w for w in a.windows if (m, w) not in table]
        if miss:
            print(f"note: {m} has no results for window(s) {miss}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "axes.titlecolor": INK})
    xs = sorted(a.windows)
    chance = 100 / len(classes)

    def style(ax, title):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.axhline(chance, color=INK2, linestyle=":", linewidth=1.2)
        ax.text(xs[-1], chance + 1.5, f"chance {chance:.0f}%", ha="right", va="bottom", fontsize=8, color=INK2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{w}\n{1000 * w / FPS:.0f} ms" for w in xs], fontsize=8.5)
        ax.minorticks_off()
        ax.set_xlim(xs[0] / 1.25, xs[-1] * 1.25)
        ax.set_ylim(0, 102)
        ax.set_title(title, fontsize=10.5, loc="left")

    def line(ax, m, key, **kw):
        pts = [(w, 100 * table[(m, w)][key]) for w in xs if (m, w) in table]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], **kw)

    # 1. per-class recall, one panel per model
    ncol = 3
    nrow = int(np.ceil(len(have) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.9 * nrow + 0.6), sharey=True, facecolor=SURFACE,
                             squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i >= len(have):
            ax.set_visible(False)
            continue
        m, title = have[i]
        style(ax, title)
        for k, c in enumerate(classes):
            line(ax, m, c, color=COLORS[k % len(COLORS)], linewidth=2, marker=MARKERS[k % len(MARKERS)],
                 markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.2, label=f"{c}°" if c != "none" else "none")
        line(ax, m, "balanced", color=INK, linewidth=2.6, linestyle="--", label="balanced accuracy")
        if i % ncol == 0:
            ax.set_ylabel(f"recall per class, % ({tag})")
        if i >= len(have) - ncol:
            ax.set_xlabel("window length (frames)")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{data}: angle_to_head recall per class vs window length", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    out_png = f"windows_{data}_{tag}.png"
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # 2. balanced accuracy of every model in one plot
    fig, ax = plt.subplots(figsize=(9.6, 4.6), facecolor=SURFACE)
    style(ax, f"{data}: balanced accuracy vs window length ({tag})")
    for k, (m, title) in enumerate(have):
        line(ax, m, "balanced", color=COLORS[k % len(COLORS)], linewidth=2, marker=MARKERS[k % len(MARKERS)],
             markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.2, label=title)
    ax.set_xlabel("window length (frames)")
    ax.set_ylabel("balanced accuracy, %")
    ax.legend(frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    out_sum = f"windows_{data}_{tag}_summary.png"
    fig.savefig(out_sum, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    saved = [out_png, out_sum]

    # 3. confusion matrices at one window
    if a.confusion:
        W = a.confusion
        ms = [(m, t) for m, t in have if (m, W) in table]
        if not ms:
            print(f"no results at window {W} for the confusion plot")
        else:
            from matplotlib.colors import LinearSegmentedColormap
            cmap = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
            lab = [f"{c}°" if c != "none" else "none" for c in classes]
            fig, axes = plt.subplots(1, len(ms), figsize=(3.6 * len(ms), 3.9), facecolor=SURFACE, squeeze=False)
            for ax, (m, title) in zip(axes.flat, ms):
                cm = table[(m, W)]["cm"].astype(float)
                pct = 100 * cm / np.maximum(cm.sum(1, keepdims=True), 1)
                ax.imshow(pct, cmap=cmap, vmin=0, vmax=100)
                for i in range(len(classes)):
                    for j in range(len(classes)):
                        if pct[i, j] >= 0.5:
                            ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center", fontsize=8,
                                    color="#ffffff" if pct[i, j] > 55 else INK)
                ax.set_xticks(range(len(classes)))
                ax.set_xticklabels(lab, fontsize=8)
                ax.set_yticks(range(len(classes)))
                ax.set_yticklabels(lab, fontsize=8)
                ax.set_xlabel("predicted")
                ax.set_title(f"{title}\nbalanced {100 * table[(m, W)]['balanced']:.1f}%", fontsize=9, loc="left")
                for side in ax.spines.values():
                    side.set_visible(False)
            axes.flat[0].set_ylabel("true")
            fig.suptitle(f"{data}: confusion matrices at {W} frames ({1000 * W / FPS:.0f} ms), {tag}, "
                         "% of each true class", x=0.01, ha="left", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.93))
            out_cm = f"confusion_{data}_{tag}_{W}.png"
            fig.savefig(out_cm, dpi=150, facecolor=SURFACE)
            plt.close(fig)
            saved.append(out_cm)
    print("\nSaved " + ", ".join(saved + [out_csv]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
