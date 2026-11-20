#!/usr/bin/env python3
"""
offset_check.py - which tags' resting position tells the angles apart?

For every tag_imu_to_flow_class CSV it takes each tag's mean deviation from its zero frame over the
recording (translation_mm.x/y/z, rows where the tag was seen). Then, per tag and axis:

  F = spread of the per-angle means / spread between recordings of the same angle

F is about 1 or less when the angle makes no difference to that tag's resting position, and large when
it does. The angle columns show each angle's mean (mm) with the spread across its recordings (sd).

How to read it:
  - Reference tags (fixed to the camera, not on the whiskers) should NOT change with the angle. If they
    have a large F, the camera or rig moved between angle conditions: setup, not flow.
  - If nearly every tag shifts by about the same amount in the same direction, that also points to the
    whole rig moving. Flow bends the whiskers differently from tag to tag.
  - If the large F values sit on the whisker tags while the reference tags stay near 1, the offset
    comes from the whiskers bending, which is what you want.

--stat jitter compares each tag's frame-to-frame jitter instead: the spread (std) of its change from
one frame to the next, over frames where the tag was seen in both, in mm. That is what a model fed frame
differences reads. Reference tags whose jitter changes with the angle point to the camera's view of the
tags changing (distance, angle, lighting), not to flow.

Usage:  python offset_check.py data_11_16_flow/ [--ref-tags 0,20] [--top 15] [--stat mean|jitter]
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

TAG_RE = re.compile(r"^derived/tf\[tag_(\d+)_zero->tag_\d+\]\.translation_mm\.([xyz])$")


def main():
    ap = argparse.ArgumentParser(description="Per-tag resting offsets by angle_to_head")
    ap.add_argument("data_dir")
    ap.add_argument("--ref-tags", default="0,20", help="tags fixed to the camera (default 0,20)")
    ap.add_argument("--top", type=int, default=15, help="whisker tags to list, largest F first (default 15)")
    ap.add_argument("--stat", choices=("mean", "jitter"), default="mean",
                    help="mean = resting position (default); jitter = std of the frame-to-frame change")
    a = ap.parse_args()
    refs = {int(x) for x in a.ref_tags.split(",") if x.strip()}

    rows = []
    for f in sorted(glob.glob(os.path.join(a.data_dir, "**", "*.csv"), recursive=True)):
        head = pd.read_csv(f, nrows=0).columns
        if "angle_to_head" not in head or "test" not in head:
            continue
        lab = pd.read_csv(f, usecols=["test", "angle_to_head"], dtype=str, keep_default_na=False)
        if lab["test"].str.strip().str.lower().mode().iat[0] != "angle_to_head":
            continue
        angle = lab["angle_to_head"].str.strip().str.lower().mode().iat[0]
        if not angle:
            continue
        df = pd.read_csv(f)
        rec = {"file": os.path.basename(f), "angle": angle}
        for c in df.columns:
            m = TAG_RE.match(c)
            if not m:
                continue
            seen = df.get(c.split(".translation_mm")[0] + ".seen")
            v = pd.to_numeric(df[c], errors="coerce")
            ok = (pd.to_numeric(seen, errors="coerce") == 1) if seen is not None else v.notna()
            if a.stat == "jitter":
                d = v.diff()[ok & ok.shift(1, fill_value=False)]
                rec[(int(m.group(1)), m.group(2))] = float(d.std()) if d.notna().sum() > 2 else np.nan
            else:
                v = v[ok]
                rec[(int(m.group(1)), m.group(2))] = float(v.mean()) if v.notna().any() else np.nan
        rows.append(rec)
    if not rows:
        print("No angle_to_head recordings found.")
        return 1

    def akey(s):
        try:
            return (0, float(s))
        except ValueError:
            return (1, 0.0)
    angles = sorted({r["angle"] for r in rows}, key=akey)
    keys = sorted({k for r in rows for k in r if isinstance(k, tuple)})
    print(("Resting position (mean deviation, mm)" if a.stat == "mean" else
           "Frame-to-frame jitter (std of the change between frames, mm)") + "; per angle: mean (sd across recordings)")
    print(f"{len(rows)} recordings; angles: " + ", ".join(f"{g} x{sum(r['angle'] == g for r in rows)}" for g in angles))

    res = []
    for k in keys:
        groups = [np.array([r[k] for r in rows if r["angle"] == g and np.isfinite(r.get(k, np.nan))]) for g in angles]
        ok = [x for x in groups if len(x) >= 2]
        if len(ok) < 2:
            continue
        within = np.mean([x.var(ddof=1) for x in ok])
        between = np.var([x.mean() for x in ok], ddof=1)
        res.append((k, between / max(within, 1e-12), groups))

    def show(items, title):
        print(f"\n{title}")
        print(f"  {'tag.axis':10s} {'F':>8s}   " + "   ".join(f"{g:>15s}" for g in angles))
        for (tag, ax), F, groups in items:
            fmt = "{:+7.2f} ({:4.2f})" if a.stat == "mean" else "{:7.4f} ({:.4f})"
            cells = [fmt.format(x.mean(), x.std(ddof=1) if len(x) > 1 else 0) if len(x) else " " * 15 for x in groups]
            print(f"  {f'{tag}.{ax}':10s} {F:8.1f}   " + "   ".join(f"{c:>15s}" for c in cells))

    ref = [x for x in res if x[0][0] in refs]
    whisk = sorted((x for x in res if x[0][0] not in refs), key=lambda x: -x[1])
    show(ref, f"Reference tags {sorted(refs)} (should not change with the angle):" if ref
         else f"Reference tags {sorted(refs)}: not in these files")
    show(whisk[: a.top], f"Whisker tags, largest F first (top {min(a.top, len(whisk))} of {len(whisk)}):")
    if whisk:
        Fs = np.array([x[1] for x in whisk])
        print(f"\nWhisker tag/axis pairs with F > 10: {int((Fs > 10).sum())} of {len(Fs)}"
              f"   (median F {np.median(Fs):.1f})")
    if ref:
        print(f"Reference tag max F: {max(x[1] for x in ref):.1f}"
              "   (large = the camera or rig moved between angles)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
