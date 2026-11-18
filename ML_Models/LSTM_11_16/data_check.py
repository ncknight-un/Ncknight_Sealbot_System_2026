#!/usr/bin/env python3
"""
data_check.py - look at a folder of tag_imu_to_flow_class recordings before blaming the model
(for train_lstm_flow.py: tags + IMU/gyro in; wall test: wall_side + dist_to_wall out, angle test:
angle_to_head + dist_to_head out).

Each recording's 'test' column says which test it belongs to; only the chosen test's recordings are
looked at (--test wall / --test angle; not needed when the folder holds one test). Files without a
'test' column are skipped. Below, "main label" = wall_side or angle_to_head, "dist" = dist_to_wall
or dist_to_head.

Prints, per recording: day, main label and dist, length, how many --seg_len segments it gives, the
mean turn rate, the drift of the lateral acceleration, and tags never detected / often missed. Then:

  1. Segments per class. What train_lstm_flow.py will have to learn from, and whether each head has
     at least two classes (a head with one class is left out of training).
  2. ROV motion by class. The IMU/gyro are model inputs, so if the ROV itself moved differently for
     one class (a steady yaw, more vibration, a different speed), or the gravity-removed
     acceleration carries a different offset, the model can tell the classes apart from the IMU
     alone, without sensing the wall / disturbance. The rows should look alike.
     The matching model test:  train_lstm_flow.py --inputs imu  should score near its baseline.
  3. Tag offsets by day. Each tag's mean offset from its zero pose, per day. If one day's tags sit
     noticeably off (zero frames set differently, tags or camera moved) and the classes weren't
     spread over the days, the model can learn the day instead of the class.
  4. Tag trouble: tags missed in more than 2 % of rows, and tags whose pose jumps (rotated more
     than 30 degrees from the zero pose in some rows = a detection glitch). Candidates for --drop_tags.

Usage:  python data_check.py DATA_DIR [--test wall|angle] [--seg_len 16] [--day REGEX]
        (default day pattern: MM_DD_YYYY in the file name, else the first NN_NN_NN, else the subfolder)
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

FPS = 65.0
IMU = "derived/imu/data/gravity_removed."
GYRO = [IMU + f"angular_velocity.{a}" for a in "xyz"]
ACC = [IMU + f"linear_acceleration.{a}" for a in "xyz"]
TAG_RE = re.compile(r"derived/tf\[tag_(\d+)_zero->tag_\d+\]\.translation_mm\.([xyz])$")
TESTS = {"wall": ("wall_side", "dist_to_wall"), "angle": ("angle_to_head", "dist_to_head")}
TEST_VALUES = {"wall": "dist_to_wall", "angle": "angle_to_head"}
ROT_W = re.compile(r"derived/tf\[tag_(\d+)_zero->tag_\d+\]\.rotation\.w$")
SEEN = re.compile(r"derived/tf\[tag_(\d+)_zero->tag_\d+\]\.seen$")

def label_of(path, col):
    """The most common non-blank value of a label column, written like train_lstm_flow does. Read as
    text so that 'none' / 'None' (no wall) stays a label instead of becoming an empty cell."""
    if not col or col not in pd.read_csv(path, nrows=0).columns:
        return "(none found)"
    raw = pd.read_csv(path, usecols=[col], dtype=str, keep_default_na=False)[col].str.strip().str.lower()
    num = pd.to_numeric(raw, errors="coerce")
    keys = [f"{round(float(v), 3) + 0.0:g}" if np.isfinite(v) else t for v, t in zip(num, raw)]
    keys = pd.Series([k for k in keys if k not in ("", "nan", "inf", "-inf")], dtype=str)
    return keys.mode().iloc[0] if len(keys) else "(blank)"


def col_stat(df, cols, fn):
    return [fn(pd.to_numeric(df[c], errors="coerce").to_numpy(float)) if c in df else np.nan for c in cols]


def summarize(path, root, day_re, main_col, dist_col, seg_len):
    rel = os.path.relpath(path, root)
    m = (re.search(day_re, os.path.basename(path)) or re.search(day_re, rel)
         or re.search(r"\d+_\d+_\d+", os.path.basename(path)))
    day = m.group(0) if m else (rel.split(os.sep)[0] if os.sep in rel else "?")
    df = pd.read_csv(path)
    dur = float(df["t_s"].iloc[-1] - df["t_s"].iloc[0]) if "t_s" in df and len(df) > 1 else len(df) / FPS
    tags = defaultdict(dict)
    for c in df.columns:
        t = TAG_RE.search(c)
        if t:
            tags[int(t.group(1))][t.group(2)] = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
    never, mean_xyz = [], {}
    for k, ax in tags.items():
        if all(np.isnan(v).all() for v in ax.values()):
            never.append(k)
        elif all(a in ax for a in "xyz"):
            mean_xyz[k] = np.array([np.nanmean(ax[a]) for a in "xyz"])
    missed, jumps = {}, {}
    for c in df.columns:
        if SEEN.search(c):
            missed[int(SEEN.search(c).group(1))] = 1.0 - float(pd.to_numeric(df[c], errors="coerce").mean())
        elif ROT_W.search(c):
            w = pd.to_numeric(df[c], errors="coerce").abs().clip(upper=1).to_numpy(float)
            jumps[int(ROT_W.search(c).group(1))] = int(np.nansum(np.degrees(2 * np.arccos(w)) > 30))
    acc_y = pd.to_numeric(df[ACC[1]], errors="coerce").to_numpy(float) if ACC[1] in df else np.full(len(df), np.nan)
    n3 = max(len(df) // 3, 1)
    drift = float(np.nanmean(acc_y[-n3:]) - np.nanmean(acc_y[:n3])) if len(df) > 3 else np.nan
    return {"missed": missed, "jumps": jumps, "acc_drift": drift, "file": os.path.basename(path), "day": day, "main": label_of(path, main_col),
            "dist": label_of(path, dist_col), "dur": dur, "rows": len(df), "segs": len(df) // seg_len,
            "gyro_mean": np.degrees(col_stat(df, GYRO, np.nanmean)), "gyro_std": np.degrees(col_stat(df, GYRO, np.nanstd)),
            "acc_mean": np.array(col_stat(df, ACC, np.nanmean)), "acc_std": np.array(col_stat(df, ACC, np.nanstd)),
            "never": sorted(never), "present": set(tags), "mean_xyz": mean_xyz}


def order(keys):
    """Numbers in numeric order, then text labels ('none', '(blank)') alphabetically."""
    def key(k):
        try:
            return (0, float(k), "")
        except ValueError:
            return (1, 0.0, k)
    return sorted(keys, key=key)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir")
    ap.add_argument("--seg_len", type=int, default=16, help="frames per segment, as in train_lstm_flow.py (16)")
    ap.add_argument("--test", choices=list(TESTS), help="default: the one test the folder holds")
    ap.add_argument("--day", default=r"(?<!\d)(?:0?[1-9]|1[0-2])_(?:0?[1-9]|[12]\d|3[01])_(?:\d{4}|\d{2})(?!\d)",
                    help="regex for the day in the file name (default MM_DD_YYYY or MM_DD_YY, else the first NN_NN_NN)")
    a = ap.parse_args()
    root = os.path.expanduser(a.data_dir)
    every = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
    if not every:
        raise SystemExit(f"no CSV files in {root}")
    kinds = {f: next((t for t, v in TEST_VALUES.items() if v == label_of(f, "test")), None) for f in every}
    skipped = [os.path.basename(f) for f, k in kinds.items() if k is None]
    if skipped:
        print(f"Skipping {len(skipped)} file(s) with no usable 'test' column (not tag_imu_to_flow_class "
              f"exports?), e.g. {', '.join(skipped[:3])}")
    found = sorted({k for k in kinds.values() if k})
    if not found:
        raise SystemExit("no tag_imu_to_flow_class recordings found (the 'test' column should say "
                         + " or ".join(TEST_VALUES.values()) + ")")
    if a.test is None and len(found) > 1:
        raise SystemExit("this folder holds both wall-test and angle-test recordings: choose with --test wall or --test angle")
    test = a.test or found[0]
    files = [f for f in every if kinds[f] == test]
    if not files:
        raise SystemExit(f"no {test}-test recordings here")
    main_col, dist_col = TESTS[test]
    main = main_col
    print(f"Test: {test} ({len(files)} recordings). Labels: {main_col}, {dist_col}")
    recs = [summarize(f, root, a.day, main_col, dist_col, a.seg_len) for f in files]
    all_tags = sorted(set().union(*(r["present"] for r in recs)))
    for r in recs:                                            # tags absent from the file entirely
        r["never"] = sorted(set(r["never"]) | (set(all_tags) - r["present"]))

    print(f"\n{len(recs)} recordings. Segments = whole {a.seg_len}-frame segments ({a.seg_len / FPS:.2f} s each); "
          "yaw = mean gyro z, deg/s, + = left;"
          "\nacc y drift = lateral acceleration, last third minus first third of the recording (m/s^2; a level, "
          "steady run stays near 0).\n")
    print(f"{'file':52s} {'day':9s} {main:>14s} {dist_col:>13s} {'len s':>6s} {'segs':>5s} {'yaw':>7s} {'acc y drift':>11s}  tags never seen")
    for r in sorted(recs, key=lambda r: (r["main"], r["dist"], r["day"], r["file"])):
        print(f"{r['file'][:52]:52s} {r['day']:9s} {r['main']:>14s} {r['dist']:>13s} {r['dur']:6.1f} {r['segs']:5d} "
              f"{r['gyro_mean'][2]:7.2f} {r['acc_drift']:11.2f}  {r['never'] if r['never'] else ''}")
    short = [r["file"] for r in recs if r["segs"] == 0]
    if short:
        print(f"\n  {len(short)} recording(s) shorter than one segment (skipped in training): {', '.join(short[:5])}")

    print("\n1. Segments per class (before the train / val / test split):")
    for what, key in ((main, "main"), (dist_col, "dist")):
        g = defaultdict(lambda: [0, 0])
        for r in recs:
            g[r[key]][0] += 1
            g[r[key]][1] += r["segs"]
        print(f"   {what}: " + ",  ".join(f"{k}: {v[0]} rec / {v[1]} segs" for k, v in
                                           sorted(g.items(), key=lambda kv: order(g).index(kv[0]))))
        real = [k for k in g if not k.startswith("(")]
        if len(real) < 2:
            print(f"      only {len(real)} {what} value(s): nothing to classify, train_lstm_flow.py leaves "
                  f"the {what} head out")
        elif min(v[0] for k, v in g.items() if not k.startswith("(")) < 3:
            print(f"      a class with fewer than 3 recordings can't be in train, val and test at once")

    print("\n2. ROV motion by class: mean over recordings of each recording's mean gyro (deg/s) and of its"
          "\n   gyro / acceleration spread (std, all three axes combined). Rows that differ a lot = the classes"
          "\n   can be told apart from the IMU alone (check with: train_lstm_flow.py --inputs imu):")
    for what, key in ((main, "main"), (dist_col, "dist")):
        groups = defaultdict(list)
        for r in recs:
            groups[r[key]].append(r)
        if len(groups) < 2:
            continue
        print(f"   {'by ' + what:17s} {'n':>3s} {'gyro x':>8s} {'gyro y':>8s} {'gyro z':>8s} {'gyro std':>9s} {'acc x':>7s} {'acc y':>7s} {'acc z':>7s} {'acc std':>8s}")
        for k in order(groups):
            v = groups[k]
            gm = np.nanmean([r["gyro_mean"] for r in v], axis=0)
            gs = np.nanmean([np.linalg.norm(r["gyro_std"]) for r in v])
            as_ = np.nanmean([np.linalg.norm(r["acc_std"]) for r in v])
            am = np.nanmean([r["acc_mean"] for r in v], axis=0)
            print(f"   {k:17s} {len(v):3d} {gm[0]:8.2f} {gm[1]:8.2f} {gm[2]:8.2f} {gs:9.2f} {am[0]:7.2f} {am[1]:7.2f} {am[2]:7.2f} {as_:8.3f}")

    days = sorted({r["day"] for r in recs})
    print("\n3. Tag offsets by day: each tag's mean offset from its zero pose (mm, length of the mean x/y/z"
          "\n   offset over that day's recordings). Rows that change a lot between days = that day's tags sit"
          "\n   somewhere else:")
    print(f"   {'tag':>5s}" + "".join(f"{d:>11s}" for d in days) + "   biggest gap between days")
    for k in all_tags:
        per_day = {}
        for d in days:
            vs = [r["mean_xyz"][k] for r in recs if r["day"] == d and k in r["mean_xyz"]]
            if vs:
                per_day[d] = np.mean(vs, axis=0)
        cells = "".join(f"{np.linalg.norm(per_day[d]):11.2f}" if d in per_day else f"{'-':>11s}" for d in days)
        ks = list(per_day)
        gap = max((np.linalg.norm(per_day[x] - per_day[y]) for i, x in enumerate(ks) for y in ks[i + 1:]), default=np.nan)
        print(f"   {k:>5d}{cells}   {gap:8.2f} mm")
    print(f"\n   Recordings per day and {main}: " + "; ".join(
        f"{d}: " + ", ".join(f"{k} x{sum(r['day'] == d and r['main'] == k for r in recs)}"
                             for k in order({r['main'] for r in recs if r['day'] == d})) for d in days))
    nev = defaultdict(set)
    for r in recs:
        for t in r["never"]:
            nev[t].add(r["day"])
    if nev:
        print("   Tags never seen, by day: " + "; ".join(f"tag {t}: {', '.join(sorted(d))}" for t, d in sorted(nev.items())))


    print("\n4. Tag trouble (candidates for --drop_tags):")
    any_ = False
    for k in all_tags:
        miss = [r["missed"].get(k, 1.0) for r in recs]
        jump = sum(r["jumps"].get(k, 0) for r in recs)
        if np.mean(miss) > 0.02 or jump:
            any_ = True
            print(f"   tag {k:2d}: missed in {np.mean(miss):.1%} of rows (worst recording {np.max(miss):.1%})"
                  + (f", pose jumps of more than 30 deg in {jump} row(s)" if jump else ""))
    if not any_:
        print("   none")


if __name__ == "__main__":
    main()
