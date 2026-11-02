#!/usr/bin/env python3
"""
data_check.py - look at a folder of tag_to_imu_format recordings before blaming the model.

Prints, per recording: day, labels, length, the measured turn rate, tags never detected, and how far
the tags sit from their zero pose. Then two summaries:

  1. Turn rate by label. A 5 m circle flown at the same speed as a 7.5 m circle turns 1.5x faster. If
     the files labelled "right 5 m" turn at the rate of the 7.5 m files, they were flown wider than
     labelled, and the model calling them 7.5 m is reading the data correctly.
  2. Tag offsets on STRAIGHT runs, by day. Going straight, the tags should sit near the same pose every
     day. If one day's straight runs sit noticeably off, every recording from that day carries a fixed
     offset (zero frames set differently, tags or camera moved), which a model can mistake for bending,
     e.g. read as a turn.

Usage:  python data_check.py DATA_DIR [--day REGEX]
        (default day pattern: the first NN_NN_NN in the file name, else the subfolder)
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

GYRO_Z = "derived/imu/data/gravity_removed.angular_velocity.z"
TURN_AVG = "derived/turn.turn_rate_avg"
TAG_RE = re.compile(r"derived/tf\[tag_(\d+)_zero->tag_\d+\]\.translation_mm\.([xyz])$")


def summarize(path, root, day_re):
    rel = os.path.relpath(path, root)
    m = re.search(day_re, os.path.basename(path)) or re.search(day_re, rel)
    day = m.group(0) if m else (rel.split(os.sep)[0] if os.sep in rel else "?")
    df = pd.read_csv(path)
    act = df["action"].dropna().astype(str).str.strip().str.lower() if "action" in df else pd.Series([], dtype=str)
    act = act[act != ""]
    action = act.mode().iloc[0] if len(act) else "(none)"
    dm = pd.to_numeric(df["approx_diam"], errors="coerce").dropna() if "approx_diam" in df else pd.Series([], dtype=float)
    size = "straight" if action == "straight" else (f"{dm.mode().iloc[0]:g} m" if len(dm) else "(no size)")
    dur = float(df["t_s"].iloc[-1] - df["t_s"].iloc[0]) if "t_s" in df and len(df) > 1 else len(df) / 65
    rate_col = TURN_AVG if TURN_AVG in df else GYRO_Z if GYRO_Z in df else None
    rate = float(np.degrees(pd.to_numeric(df[rate_col], errors="coerce").mean())) if rate_col else np.nan
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
    present = {int(t.group(1)) for c in df.columns for t in [TAG_RE.search(c)] if t}
    return {"file": os.path.basename(path), "day": day, "action": action, "size": size, "dur": dur,
            "rate": rate, "rate_src": "turn_rate_avg" if rate_col == TURN_AVG else "gyro z", "never": sorted(never),
            "present": present, "mean_xyz": mean_xyz}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir")
    ap.add_argument("--day", default=r"\d+_\d+_\d+", help="regex for the day in the file name (default NN_NN_NN)")
    a = ap.parse_args()
    root = os.path.expanduser(a.data_dir)
    files = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
    if not files:
        raise SystemExit(f"no CSV files in {root}")
    recs = [summarize(f, root, a.day) for f in files]
    all_tags = sorted(set().union(*(r["present"] for r in recs)))
    for r in recs:                                            # tags absent from the file entirely
        r["never"] = sorted(set(r["never"]) | (set(all_tags) - r["present"]))

    print(f"{len(recs)} recordings. Turn rate: mean over the file, deg/s, + = left ({recs[0]['rate_src']}).\n")
    print(f"{'file':58s} {'day':9s} {'label':16s} {'len s':>6s} {'turn deg/s':>10s}  tags never seen")
    for r in sorted(recs, key=lambda r: (r["action"], r["size"], r["day"], r["file"])):
        print(f"{r['file'][:58]:58s} {r['day']:9s} {(r['action'] + ' ' + (r['size'] if r['action'] != 'straight' else '')).strip():16s} "
              f"{r['dur']:6.1f} {r['rate']:10.2f}  {r['never'] if r['never'] else ''}")

    print("\n1. Turn rate by label (deg/s; + = left). Same speed => a 5 m circle turns 1.5x faster than 7.5 m,"
          "\n   2.5 m 3x faster. Compare each size with its neighbours, and left with right:")
    groups = defaultdict(list)
    for r in recs:
        groups[(r["action"], r["size"])].append(r["rate"])
    for (act, size), v in sorted(groups.items()):
        v = np.array(v, float)
        print(f"   {act:9s} {size:10s} n={len(v):2d}  mean {np.nanmean(v):7.2f}   range {np.nanmin(v):7.2f} .. {np.nanmax(v):7.2f}")

    print("\n2. STRAIGHT runs: each tag's mean offset from its zero pose, by day (mm, length of the mean"
          "\n   x/y/z offset). Rows that change a lot between days = that day's tags sit somewhere else:")
    straight = [r for r in recs if r["action"] == "straight"]
    days = sorted({r["day"] for r in straight})
    if not straight:
        print("   no straight recordings")
        return
    print(f"   {'tag':>5s}" + "".join(f"{d:>11s}" for d in days) + "   biggest gap between days")
    for k in all_tags:
        per_day = {}
        for d in days:
            vs = [r["mean_xyz"][k] for r in straight if r["day"] == d and k in r["mean_xyz"]]
            if vs:
                per_day[d] = np.mean(vs, axis=0)
        cells = "".join(f"{np.linalg.norm(per_day[d]):11.2f}" if d in per_day else f"{'-':>11s}" for d in days)
        ks = list(per_day)
        gap = max((np.linalg.norm(per_day[x] - per_day[y]) for i, x in enumerate(ks) for y in ks[i + 1:]), default=np.nan)
        print(f"   {k:>5d}{cells}   {gap:8.2f} mm")
    print("\n   Straight recordings per day: " + ", ".join(f"{d} {sum(r['day'] == d for r in straight)}" for d in days))
    nev = defaultdict(set)
    for r in recs:
        for t in r["never"]:
            nev[t].add(r["day"])
    if nev:
        print("   Tags never seen, by day: " + "; ".join(f"tag {t}: {', '.join(sorted(d))}" for t, d in sorted(nev.items())))


if __name__ == "__main__":
    main()
