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
  cmdiff     diff, then the change shared by all whisker tags in that frame removed: each tag's change minus
             the average change of every tag seen in this frame and the one before (per field). Motion that
             moves the whole array alike (camera or rig) cancels, and, unlike refdiff, no single tag's own
             noise is added to every column. The tags given with --exclude (default 0,20: fixed reference
             tags) are left out of the average and dropped.
  cmonly     the opposite of cmdiff: ONLY the change shared by the whisker tags, i.e. the average frame-to-
             frame change across every tag (not in --exclude) seen in this frame and the one before, written
             as one made-up tag, derived/tf[tag_99_zero->tag_99]. Every real tag is dropped. Tells how much of
             the angle is in motion the whole array shares (camera, rig, or the whole head moving).
  --groups "1-9:10-18"  for cmdiff / cmonly: average within each group of tags separately (e.g. the two
             whisker panels) instead of across the whole array. cmdiff then keeps how each whisker moves
             relative to its OWN panel; cmonly gives one made-up tag per group (tag_91 for the first
             group, tag_92 for the second, ...): how each panel moves as a whole.
  refdiff    ref, then diff: each tag's frame-to-frame change relative to the reference tag. Motion that the
             reference tag shares (camera or rig vibration, common measurement noise) cancels; what is left
             is the whiskers moving relative to it. The change is 0 on frames where this tag or the reference
             was not seen in this frame or the one before.

Usage:
    python transform_tags.py data_11_16_flow data_11_16_flow_diff --mode diff
    python transform_tags.py data_11_16_flow data_11_16_flow_hp5  --mode highpass --window 5
    python transform_tags.py data_11_16_flow data_11_16_flow_ref20 --mode ref --ref 20
--keep translation | rotation  keep only the tags' translation_mm columns, or only their rotation
           columns (the others are dropped). AprilTag rotation estimates are much noisier and depend more
           on the viewing angle than translation, so comparing the two shows which one a model relies on.
--shuffle  after the transform, put each recording's rows in random order (a fixed seed per file). Only
           meaningful with diff / cmdiff / cmonly: each row is then still a real frame-to-frame change, but
           consecutive rows no longer follow each other in time, so a model can only use how the changes
           are distributed, not how they unfold over time. If accuracy stays the same, the model reads a
           per-frame "fingerprint" rather than motion dynamics.
--rotvec   for diff / cmdiff / cmonly: replace each tag's four rotation-component changes with its actual
           rotation between consecutive frames, as a rotation vector in degrees (rotvec_deg.x/y/z: axis
           times angle). Plain differences of the quaternion's x/y/z/w depend on which way the tag already
           points, so they leak each tag's resting orientation (a steady offset) into every frame; the
           rotation vector does not. 0 when the tag was not seen in this frame or the one before.
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
ROTVEC = [f"rotvec_deg.{a}" for a in "xyz"]


def rotvec_deg(q_prev, q):
    """Rotation from q_prev to q (both (n,4) xyzw) as a rotation vector, degrees: axis * angle."""
    d = quat_mul(q_prev * np.array([-1.0, -1.0, -1.0, 1.0]), q)
    d = d * np.where(d[:, 3:4] < 0, -1.0, 1.0)                 # shortest way round
    v = d[:, :3]
    nv = np.linalg.norm(v, axis=1)
    ang = 2.0 * np.arctan2(nv, d[:, 3])
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(nv[:, None] > 1e-12, v / nv[:, None] * ang[:, None], 0.0)
    return np.degrees(out)


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


def parse_groups(text):
    """'1-9:10-18' -> [{1..9}, {10..18}]."""
    out = []
    for part in text.split(":"):
        g = set()
        for item in part.split(","):
            item = item.strip()
            if "-" in item:
                lo, hi = item.split("-")
                g |= set(range(int(lo), int(hi) + 1))
            elif item:
                g.add(int(item))
        out.append(g)
    return out


def transform(df, mode, window, ref, exclude=(0, 20), groups=None, rotvec=False):
    if mode in ("cmdiff", "cmonly"):
        out = transform(df, "diff", window, ref, rotvec=rotvec)
        tags = {}
        for c in out.columns:
            m = POSE_RE.match(c)
            if m:
                tags.setdefault(int(m.group(2)), m.group(1))
        drop = [n for n in tags if n in exclude]
        keep = {n: b for n, b in tags.items() if n not in exclude}
        if len(keep) < 2:
            raise ValueError("cmdiff needs at least two whisker tags")
        ok = {}
        for n, b in keep.items():
            sc = f"{b}.seen"
            seen = pd.to_numeric(df[sc], errors="coerce").fillna(0).to_numpy() > 0.5 if sc in df.columns \
                else np.ones(len(df), bool)
            ok[n] = seen & np.r_[False, seen[:-1]]
        shared = {}
        sets = [set(keep)] if not groups else [set(keep) & g for g in groups]
        missing = set(keep) - set().union(*sets)
        if groups and missing:
            raise ValueError(f"tags {sorted(missing)} are in no --groups group (add them, or to --exclude)")
        for gi, members in enumerate(sets):
            if not members:
                continue
            fake = 99 if not groups else 91 + gi
            for f in FIELDS + ROTVEC:
                cols = {n: f"{keep[n]}.{f}" for n in sorted(members) if f"{keep[n]}.{f}" in out.columns}
                if not cols:
                    continue
                V = np.stack([np.asarray(out[c], float) for c in cols.values()], 1)
                M = np.stack([ok[n] for n in cols], 1)
                cnt = M.sum(1)
                mean = np.where(cnt > 0, (V * M).sum(1) / np.maximum(cnt, 1), 0.0)
                if mode == "cmonly":
                    shared[f"derived/tf[tag_{fake}_zero->tag_{fake}].{f}"] = mean
                    continue
                for j, c in enumerate(cols.values()):
                    out[c] = np.where(M[:, j], V[:, j] - mean, 0.0)
        for n in (list(tags) if mode == "cmonly" else drop):
            b = tags[n]
            out = out.drop(columns=[c for c in out.columns if c.startswith(b + ".")])
        if mode == "cmonly":
            for fake in sorted({int(re.search(r"tag_(\d+)_zero", k).group(1)) for k in shared}):
                shared[f"derived/tf[tag_{fake}_zero->tag_{fake}].seen"] = np.ones(len(out), int)
            out = pd.concat([out, pd.DataFrame(shared, index=out.index)], axis=1)
        return out.copy()
    if mode == "refdiff":
        out = transform(df, "ref", window, ref)
        base_ref = next((m.group(1) for c in df.columns for m in [POSE_RE.match(c)] if m and int(m.group(2)) == ref), None)
        rseen = pd.to_numeric(df.get(f"{base_ref}.seen", 1), errors="coerce")
        rseen = (np.ones(len(df), bool) if np.isscalar(rseen) else rseen.fillna(0).to_numpy() > 0.5)
        for c in [c for c in out.columns if POSE_RE.match(c)]:
            base = POSE_RE.match(c).group(1)
            seen = pd.to_numeric(df[f"{base}.seen"], errors="coerce").fillna(0).to_numpy() > 0.5 \
                if f"{base}.seen" in df.columns else np.ones(len(df), bool)
            both = seen & rseen
            ok = both & np.r_[False, both[:-1]]
            v = np.asarray(out[c], float)
            out[c] = np.where(ok, np.r_[0.0, np.diff(v)], 0.0)
        return out
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
    rot_new = {}
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
        if mode == "diff" and rotvec:
            q = np.stack([p[f"rotation.{a}"] for a in "xyzw"], 1)
            q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
            rv = np.zeros((len(q), 3))
            rv[1:] = rotvec_deg(q[:-1], q[1:])
            ok = seen & np.r_[False, seen[:-1]]
            rot_new[base] = np.where(ok[:, None], rv, 0.0)
    if rot_new:
        out = out.drop(columns=[f"{b}.rotation.{a}" for b in rot_new for a in "xyzw"])
        out = pd.concat([out, pd.DataFrame({f"{b}.{f}": rot_new[b][:, k] for b in rot_new
                                            for k, f in enumerate(ROTVEC)}, index=out.index)], axis=1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("in_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--mode", choices=("diff", "highpass", "ref", "refdiff", "cmdiff", "cmonly"), required=True)
    ap.add_argument("--exclude", default="0,20", help="cmdiff: tags left out of the average and dropped (default 0,20)")
    ap.add_argument("--groups", help='cmdiff/cmonly: average within these tag groups separately, e.g. "1-9:10-18"')
    ap.add_argument("--window", type=float, default=5.0, help="highpass: seconds of past rows averaged (default 5)")
    ap.add_argument("--ref", type=int, default=20, help="ref: the reference tag number (default 20)")
    ap.add_argument("--no-seen", action="store_true", help="set every seen flag to 1 (removes the dropout pattern)")
    ap.add_argument("--shuffle", action="store_true", help="shuffle each recording's rows (destroys time order)")
    ap.add_argument("--rotvec", action="store_true",
                    help="diff/cmdiff/cmonly: rotation change as a rotation vector (deg) instead of quaternion differences")
    ap.add_argument("--keep", choices=("all", "translation", "rotation"), default="all",
                    help="keep only the translation_mm or only the rotation columns of every tag (default all)")
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
            res = transform(work, a.mode, a.window, a.ref, {int(x) for x in a.exclude.split(",") if x.strip()},
                            parse_groups(a.groups) if a.groups else None, a.rotvec)
            if a.rotvec and a.mode not in ("diff", "cmdiff", "cmonly"):
                ap.error("--rotvec only works with --mode diff, cmdiff or cmonly")
        except (ValueError, KeyError) as e:
            print(f"  skipping {os.path.basename(f)}: {e}")
            skipped += 1
            continue
        if a.keep != "all":
            gone = ("rotation.", "rotvec_deg.") if a.keep == "translation" else ("translation_mm.",)
            res = res.drop(columns=[c for c in res.columns if c.startswith("derived/tf[") and
                                    c.split("].", 1)[-1].startswith(gone)])
        if a.shuffle:
            if a.mode not in ("diff", "cmdiff", "cmonly"):
                ap.error("--shuffle only makes sense with --mode diff, cmdiff or cmonly")
            import zlib
            rng = np.random.default_rng(zlib.crc32(os.path.basename(f).encode()))
            res = res.iloc[rng.permutation(len(res))].reset_index(drop=True)
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
