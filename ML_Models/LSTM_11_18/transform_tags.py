#!/usr/bin/env python3
"""
transform_tags.py - rewrite the tag columns of tag_imu_to_flow_class CSVs, so train_lstm_flow.py can be
run unchanged on inputs where a constant tag offset (e.g. the camera or rig sitting slightly differently
between conditions) can't be used.

Writes a copy of every CSV under IN_DIR to OUT_DIR (same file names and subfolders). Only the tag pose
columns (derived/tf[tag_N_zero->tag_N].translation_mm.x/y/z and rotation.x/y/z/w) are changed. The seen
flags, IMU, depth, turn rate, t_s, dt_s and label columns are copied as they are.

Missed detections are first held at the last detected value (earlier rows only, as train_lstm_flow.py
does), so every output cell is filled.

Modes (--mode):
  diff       change since the previous frame. Removes any constant offset exactly; usable live. The frame a
             tag is re-detected after a miss would show a false jump, so the change is set to 0 on any
             frame where the tag was not seen in this frame or the one before.
  highpass   value minus its own average over the previous --window seconds (default 5; past rows only, so
             usable live). Removes offsets and slow drift, keeps faster whisker motion.
  ref        every tag relative to a reference tag (--ref, default 20): translation minus the reference
             tag's translation, rotation = the reference's rotation undone, then the tag's. A shift of the
             camera that moves every tag alike cancels, while the whiskers' steady bend is kept. The
             reference tag's own columns are dropped. Approximate: it assumes the zero frames of the tags
             have about the same orientation (tags on one plate facing the camera).

Usage:
    python transform_tags.py data_11_16_flow data_11_16_flow_diff --mode diff
    python transform_tags.py data_11_16_flow data_11_16_flow_hp5  --mode highpass --window 5
    python transform_tags.py data_11_16_flow data_11_16_flow_ref20 --mode ref --ref 20
--no-seen  also set every seen flag to 1, so the model can't tell from the flags which tags dropped out
           (a missed tag then reads "no change" in diff mode, or its held value otherwise).

then train as before on the new folder (make a new split: the old split file points at the old folder).
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

POSE_RE = re.compile(r"^(derived/tf\[tag_(\d+)_zero->tag_\d+\])\.(translation_mm\.[xyz]|rotation\.[xyzw])$")
FIELDS = [f"translation_mm.{a}" for a in "xyz"] + [f"rotation.{a}" for a in "xyzw"]


def hold(v, w_default):
    """Hold the last value over gaps (earlier rows only); before the first value: the zero pose."""
    v = pd.Series(v, dtype=float).ffill().to_numpy(copy=True)
    v[np.isnan(v)] = w_default
    return v


def quat_mul(a, b):
    ax, ay, az, aw = a.T
    bx, by, bz, bw = b.T
    return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], 1)


def trailing_mean(t, x, win):
    """Mean of x over the last win seconds up to each row (rows t - win < t' <= t)."""
    c = np.r_[0.0, np.cumsum(x)]
    lo = np.searchsorted(t, t - win, side="right")
    hi = np.arange(1, len(t) + 1)
    return (c[hi] - c[lo]) / (hi - lo)


def transform(df, mode, window, ref):
    tags = {}
    for c in df.columns:
        m = POSE_RE.match(c)
        if m:
            tags.setdefault(int(m.group(2)), m.group(1))
    out = df.copy()
    poses = {}
    for n, base in tags.items():
        if not all(f"{base}.{f}" in df.columns for f in FIELDS):
            continue
        poses[n] = {f: hold(pd.to_numeric(df[f"{base}.{f}"], errors="coerce"), 1.0 if f == "rotation.w" else 0.0)
                    for f in FIELDS}
    if mode == "ref" and ref not in poses:
        raise ValueError(f"reference tag {ref} not in this file")
    t = pd.to_numeric(df["t_s"], errors="coerce").to_numpy(float) if "t_s" in df.columns \
        else np.arange(len(df)) / 65.0
    for n, base in tags.items():
        if n not in poses:
            continue
        p = poses[n]
        seen_col = f"{base}.seen"
        seen = pd.to_numeric(df[seen_col], errors="coerce").fillna(0).to_numpy() > 0.5 if seen_col in df.columns \
            else np.ones(len(df), bool)
        if mode == "ref":
            if n == ref:
                out = out.drop(columns=[f"{base}.{f}" for f in FIELDS] + ([seen_col] if seen_col in df.columns else []))
                continue
            r = poses[ref]
            for a in "xyz":
                out[f"{base}.translation_mm.{a}"] = p[f"translation_mm.{a}"] - r[f"translation_mm.{a}"]
            q = np.stack([p[f"rotation.{a}"] for a in "xyzw"], 1)
            qr = np.stack([r[f"rotation.{a}"] for a in "xyzw"], 1) * np.array([-1.0, -1.0, -1.0, 1.0])
            qq = quat_mul(qr, q)
            for k, a in enumerate("xyzw"):
                out[f"{base}.rotation.{a}"] = qq[:, k]
            continue
        for f in FIELDS:
            v = p[f]
            if mode == "diff":
                d = np.r_[0.0, np.diff(v)]
                ok = seen & np.r_[False, seen[:-1]]
                out[f"{base}.{f}"] = np.where(ok, d, 0.0)
            else:
                out[f"{base}.{f}"] = v - trailing_mean(t, v, window)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("in_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--mode", choices=("diff", "highpass", "ref"), required=True)
    ap.add_argument("--window", type=float, default=5.0, help="highpass: seconds of past rows averaged (default 5)")
    ap.add_argument("--ref", type=int, default=20, help="ref: the reference tag number (default 20)")
    ap.add_argument("--no-seen", action="store_true", help="set every seen flag to 1 (removes the dropout pattern)")
    a = ap.parse_args()
    if os.path.abspath(a.in_dir) == os.path.abspath(a.out_dir):
        ap.error("out_dir must differ from in_dir (the originals are never overwritten)")
    files = sorted(glob.glob(os.path.join(a.in_dir, "**", "*.csv"), recursive=True))
    if not files:
        print(f"No CSV files in {a.in_dir}")
        return 1
    done = skipped = 0
    for f in files:
        dst = os.path.join(a.out_dir, os.path.relpath(f, a.in_dir))
        try:
            df = pd.read_csv(f, dtype=str, keep_default_na=False)     # text, so labels like 'none' stay as they are
            num = df.apply(lambda s: pd.to_numeric(s.replace("", np.nan), errors="coerce"))
            pose_cols = [c for c in df.columns if POSE_RE.match(c)]
            work = df.copy()
            work[pose_cols] = num[pose_cols]
            if "t_s" in df.columns:
                work["t_s"] = num["t_s"]
            res = transform(work, a.mode, a.window, a.ref)
        except (ValueError, KeyError) as e:
            print(f"  skipping {os.path.basename(f)}: {e}")
            skipped += 1
            continue
        if a.no_seen:
            for c in res.columns:
                if c.startswith("derived/tf[") and c.endswith(".seen"):
                    res[c] = "1"
        for c in res.columns:
            if POSE_RE.match(c):
                res[c] = [f"{x:.10g}" for x in res[c]]
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        res.to_csv(dst, index=False)
        done += 1
    print(f"Wrote {done} file(s) to {a.out_dir} (mode {a.mode})" + (f", skipped {skipped}" if skipped else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
