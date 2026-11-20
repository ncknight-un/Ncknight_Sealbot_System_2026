#!/usr/bin/env python3
"""
fft_whiskers.py - frequency content of the whisker (tag) movement on each side, compared between
classes (angle_to_head for the angle test, wall_side for the wall test).

For every tag_imu_to_flow_class recording of the chosen test:
  1. each tag's translation_mm x / y / z is taken, short detection gaps are held at the last value,
  2. its power spectral density is estimated with Welch's method (Hann windows of --nperseg frames,
     half overlapping, each with its mean and linear trend removed, so constant offsets and slow
     drift do NOT show up here: this looks only at how the whiskers move, not where they sit),
  3. the spectra of a side's tags are averaged (power summed over the --axes).
The recordings of one class are then averaged (line = typical recording, band = spread between
recordings), so a difference between classes only counts if it is bigger than the bands.

Output (in --out_dir, default fft_<test>/):
  spectra.png       one panel per side, one line per class; then left / right power ratio (the part
                    that could tell which side a disturbance is on); then the gyro spectrum as a
                    control: a whisker peak that is also in the gyro is the ROV's own motion
  per_tag.png       the same per single tag (to see whether one tag drives a side's result)
  spectra.csv       the averaged spectra behind spectra.png
and a table of power per frequency band, class and side is printed.

Usage:
  python fft_whiskers.py data_11_16_flow --test angle --left 1,2,3 --right 4,5,6
  options: --axes xyz (default; or x, y, z, xy ...)   --nperseg 256 (longer = finer frequency steps,
           fewer averages)   --min_seen 0.9 (a tag seen in fewer rows of a recording is left out of
           that recording)   --fmax 32.5   --logx
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

FPS = 65.0
IMU = "derived/imu/data/gravity_removed."
GYRO = [IMU + f"angular_velocity.{a}" for a in "xyz"]
TESTS = {"wall": "wall_side", "angle": "angle_to_head"}
TEST_VALUES = {"wall": "dist_to_wall", "angle": "angle_to_head"}
BANDS = [(0.5, 2), (2, 5), (5, 10), (10, 20), (20, 32.5)]


def tag_col(n, field):
    return f"derived/tf[tag_{n}_zero->tag_{n}].{field}"


def label_of(path, col):
    """Most common non-blank value of a label column, read as text ('none' stays a label)."""
    if col not in pd.read_csv(path, nrows=0).columns:
        return ""
    raw = pd.read_csv(path, usecols=[col], dtype=str, keep_default_na=False)[col].str.strip().str.lower()
    num = pd.to_numeric(raw, errors="coerce")
    keys = [f"{round(float(v), 3) + 0.0:g}" if np.isfinite(v) else t for v, t in zip(num, raw)]
    keys = pd.Series([k for k in keys if k not in ("", "nan")], dtype=str)
    return keys.mode().iloc[0] if len(keys) else ""


def order(keys):
    def key(k):
        try:
            return (0, float(k), "")
        except ValueError:
            return (1, 0.0, k)
    return sorted(keys, key=key)


def welch(x, fs, nperseg):
    """Power spectral density (units^2 / Hz) of x: Hann windows, 50 % overlap, each window with its
    mean and linear trend removed."""
    step = nperseg // 2
    win = np.hanning(nperseg)
    scale = fs * (win ** 2).sum()
    k = np.arange(nperseg)
    out = []
    for i in range(0, len(x) - nperseg + 1, step):
        s = x[i:i + nperseg]
        s = s - np.polyval(np.polyfit(k, s, 1), k)
        p = np.abs(np.fft.rfft(s * win)) ** 2 / scale
        p[1:-1 if nperseg % 2 == 0 else None] *= 2
        out.append(p)
    return np.mean(out, axis=0)


def hold(v):
    """Fill gaps with the last value (first value before the first detection)."""
    return pd.Series(v).ffill().bfill().to_numpy(float)


def recording_spectra(path, groups, axes, nperseg, min_seen):
    """{group: PSD, 'tag N': PSD, 'gyro': PSD} for one recording, fs, and the tags left out."""
    df = pd.read_csv(path)
    if len(df) < nperseg:
        return None, None, None
    fs = 1.0 / float(np.nanmedian(np.diff(df["t_s"]))) if "t_s" in df and len(df) > 2 else FPS
    out, skipped = {}, []
    for g, tags in groups.items():
        per_tag = []
        for n in tags:
            cols = [tag_col(n, f"translation_mm.{a}") for a in axes]
            if any(c not in df for c in cols):
                skipped.append(n)
                continue
            seen = pd.to_numeric(df[tag_col(n, "seen")], errors="coerce").mean() if tag_col(n, "seen") in df \
                else 1.0 - df[cols[0]].isna().mean()
            if not seen >= min_seen:
                skipped.append(n)
                continue
            p = sum(welch(hold(pd.to_numeric(df[c], errors="coerce")), fs, nperseg) for c in cols)
            out[f"tag {n}"] = p
            per_tag.append(p)
        if per_tag:
            out[g] = np.mean(per_tag, axis=0)
    if all(c in df for c in GYRO):
        g = [hold(np.degrees(pd.to_numeric(df[c], errors="coerce"))) for c in GYRO]
        if not any(np.isnan(v).any() for v in g):
            out["gyro"] = sum(welch(v, fs, nperseg) for v in g)
    return out, fs, skipped


def log_stats(specs):
    """Typical spectrum (geometric mean over recordings) and its spread (x / by one log-std)."""
    L = np.log10(np.maximum(np.array(specs), 1e-30))
    m, s = L.mean(0), (L.std(0) if len(L) > 1 else np.zeros(L.shape[1]))
    return 10 ** m, 10 ** (m - s), 10 ** (m + s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir")
    ap.add_argument("--test", choices=list(TESTS), help="default: the one test the folder holds")
    ap.add_argument("--left", required=True, metavar="N,N", help="tag numbers of the left-side whiskers")
    ap.add_argument("--right", required=True, metavar="N,N", help="tag numbers of the right-side whiskers")
    ap.add_argument("--axes", default="xyz", help="translation axes whose power is added up (default xyz)")
    ap.add_argument("--nperseg", type=int, default=256, help="frames per Welch window (default 256 = ~3.9 s)")
    ap.add_argument("--min_seen", type=float, default=0.9,
                    help="leave a tag out of a recording if it was seen in fewer than this share of rows (0.9)")
    ap.add_argument("--fmax", type=float, default=32.5, help="highest frequency plotted, Hz (default 32.5)")
    ap.add_argument("--logx", action="store_true", help="logarithmic frequency axis")
    ap.add_argument("--out_dir")
    a = ap.parse_args()
    groups = {"left": [int(x) for x in a.left.split(",") if x.strip()],
              "right": [int(x) for x in a.right.split(",") if x.strip()]}
    axes = [c for c in "xyz" if c in a.axes.lower()]
    if not axes:
        ap.error("--axes needs some of x, y, z")

    files = sorted(glob.glob(os.path.join(os.path.expanduser(a.data_dir), "**", "*.csv"), recursive=True))
    kinds = {f: next((t for t, v in TEST_VALUES.items() if v == label_of(f, "test")), None) for f in files}
    found = sorted({k for k in kinds.values() if k})
    if not found:
        raise SystemExit("no tag_imu_to_flow_class recordings found (need a 'test' column)")
    if a.test is None and len(found) > 1:
        raise SystemExit("this folder holds both tests: choose with --test wall or --test angle")
    test = a.test or found[0]
    label_col = TESTS[test]
    files = [f for f in files if kinds[f] == test]

    by_class, fs_all, left_out, short = {}, [], {}, []
    for f in files:
        cls = label_of(f, label_col)
        if not cls:
            continue
        spec, fs, skipped = recording_spectra(f, groups, axes, a.nperseg, a.min_seen)
        if spec is None:
            short.append(os.path.basename(f))
            continue
        fs_all.append(fs)
        for n in skipped:
            left_out.setdefault(n, []).append(cls)
        for k, p in spec.items():
            by_class.setdefault(cls, {}).setdefault(k, []).append(p)
    if not by_class:
        raise SystemExit("no usable recordings (all shorter than --nperseg?)")
    fs = float(np.median(fs_all))
    freqs = np.fft.rfftfreq(a.nperseg, 1.0 / fs)
    classes = order(by_class)
    print(f"Test: {test}, classes by {label_col}: " + ", ".join(f"{c} ({len(by_class[c].get('gyro', by_class[c].get('left', [])))} rec)" for c in classes))
    print(f"{fs:.1f} fps, windows of {a.nperseg} frames = {a.nperseg / fs:.1f} s -> frequency step {freqs[1]:.2f} Hz, "
          f"up to {freqs[-1]:.1f} Hz. Axes: {''.join(axes)}. Left tags {groups['left']}, right tags {groups['right']}.")
    if short:
        print(f"Skipped {len(short)} recording(s) shorter than one window: {', '.join(short[:4])}")
    for n, cl in sorted(left_out.items()):
        print(f"  tag {n}: left out of {len(cl)} recording(s) (missing or seen < {a.min_seen:.0%}): "
              + ", ".join(f"{c} x{cl.count(c)}" for c in order(set(cl))))

    # ratio left / right per recording (only recordings that have both)
    for c in classes:
        d = by_class[c]
        if "left" in d and "right" in d and len(d["left"]) == len(d["right"]):
            d["left / right"] = [l / np.maximum(r, 1e-30) for l, r in zip(d["left"], d["right"])]

    out_dir = a.out_dir or f"fft_{test}"
    os.makedirs(out_dir, exist_ok=True)

    # band power table
    print("\nMean power per band (mm^2; gyro in (deg/s)^2), typical recording, x/ spread between recordings:")
    print(f"  {'':14s}{label_col:>14s}" + "".join(f"{f'{lo:g}-{hi:g} Hz':>18s}" for lo, hi in BANDS))
    for g in ("left", "right", "left / right", "gyro"):
        for c in classes:
            sp = by_class[c].get(g)
            if not sp:
                continue
            cells = ""
            for lo, hi in BANDS:
                m = (freqs >= lo) & (freqs < hi)
                if g == "left / right":
                    v = np.array([np.exp(np.mean(np.log(p[m]))) for p in sp])
                else:
                    v = np.array([p[m].sum() * freqs[1] for p in sp])
                L = np.log10(v)
                cells += f"{10 ** L.mean():11.3g} x/{10 ** L.std():4.2f}"
            print(f"  {g:14s}{c:>14s}{cells}")
        print()

    rows = []
    for g in ("left", "right", "left / right", "gyro"):
        for c in classes:
            if by_class[c].get(g):
                m, lo, hi = log_stats(by_class[c][g])
                rows += [{"signal": g, label_col: c, "freq_hz": f, "typical": x, "low": y, "high": z}
                         for f, x, y, z in zip(freqs, m, lo, hi)]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "spectra.csv"), index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: no plots (spectra.csv was written)")
        return
    colors = {c: plt.cm.tab10(i % 10) for i, c in enumerate(classes)}
    keep = (freqs > 0) & (freqs <= a.fmax)

    def draw(ax, key, ylabel, ratio=False):
        for c in classes:
            sp = by_class[c].get(key)
            if not sp:
                continue
            m, lo, hi = log_stats(sp)
            ax.plot(freqs[keep], m[keep], color=colors[c], lw=1.4, label=f"{c} (n={len(sp)})")
            ax.fill_between(freqs[keep], lo[keep], hi[keep], color=colors[c], alpha=0.15, lw=0)
        if ratio:
            ax.axhline(1, color="grey", lw=0.8)
        ax.set_yscale("log")
        if a.logx:
            ax.set_xscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, which="both")

    panels = [("left", f"left tags {groups['left']}\nmm$^2$/Hz", False),
              ("right", f"right tags {groups['right']}\nmm$^2$/Hz", False),
              ("left / right", "left / right\npower ratio", True),
              ("gyro", "gyro (control)\n(deg/s)$^2$/Hz", False)]
    panels = [p for p in panels if any(by_class[c].get(p[0]) for c in classes)]
    fig, axs = plt.subplots(len(panels), 1, figsize=(11, 2.9 * len(panels)), sharex=True, squeeze=False)
    for ax, (key, yl, ratio) in zip(axs[:, 0], panels):
        draw(ax, key, yl, ratio)
    axs[0, 0].legend(title=label_col, fontsize=8, ncol=len(classes), loc="upper right")
    axs[-1, 0].set_xlabel("frequency [Hz]")
    fig.suptitle(f"Whisker movement spectra by {label_col} (axes {''.join(axes)}; line = typical recording, "
                 "band = spread between recordings)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "spectra.png"), dpi=120)
    plt.close(fig)

    tags = [n for g in groups.values() for n in g if any(by_class[c].get(f"tag {n}") for c in classes)]
    if tags:
        cols = min(4, len(tags))
        nrow = int(np.ceil(len(tags) / cols))
        fig, axs = plt.subplots(nrow, cols, figsize=(4.2 * cols, 3.0 * nrow), sharex=True, squeeze=False)
        for ax, n in zip(axs.ravel(), tags):
            draw(ax, f"tag {n}", "")
            ax.set_title(f"tag {n} ({'left' if n in groups['left'] else 'right'})", fontsize=9)
        for ax in axs.ravel()[len(tags):]:
            ax.axis("off")
        axs[0, 0].legend(title=label_col, fontsize=7)
        fig.supxlabel("frequency [Hz]")
        fig.supylabel("mm$^2$/Hz")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "per_tag.png"), dpi=110)
        plt.close(fig)
    print(f"Saved {out_dir}/spectra.png, {out_dir}/per_tag.png, {out_dir}/spectra.csv")


if __name__ == "__main__":
    main()
