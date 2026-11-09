"""
Stateless sequence model: AprilTag (whisker) deformations -> IMU, depth, turn rate and manoeuvre labels
(PyTorch).

Data = FULL tag_to_imu_format CSVs, one recording per file, exactly as sync_export.py / the viewer
write them. No pre-cutting: windows are cut on the fly, AFTER the recordings are split, so overlapping
windows of one recording never end up on both sides of the split.
  - train: used to fit the model
  - val:   checked every epoch; the epoch with the lowest val score is saved
  - test:  never touched during training; scored once at the end
Two ways to give them:
  1. --data_dir FOLDER: every CSV in it (subfolders included) is split automatically, by WHOLE
     RECORDING, balanced by label: each direction / circle-size combination ("left 7.5 m", "straight",
     "unlabelled", ...) is spread over train / val / test in the --val_frac / --test_frac proportions
     (a combination with only 1-2 recordings can't be spread; that is reported).
       --seed             the split is random but repeatable
       --split_file PATH  reuse a saved split instead of making one
     The split used is saved next to the checkpoint as <out>_split.json.
  2. --train_dir / --val_dir / --test_dir: three folders you fill yourself. Put each recording in
     only ONE folder (the script warns if a file name appears in several).
The labels (action, approx_diam) are the same on every row of a file, so export only the manoeuvre
itself (set Start/Stop around it): rest periods before / after would be labelled as the manoeuvre.

Sliding window (hyperparameters, no need to re-cut files to change them):
  --window   rows the model is fed at once: the buffer (default 32 = 0.5 s at 65 fps)
  --warmup   the first N rows of every window are fed to the model as history but not scored: the
             model is stateless (every window starts from zeros), so its first rows are guesses with
             little context. Default: half the window (16), so every scored prediction was made with
             at least 0.25 s of history. Live, the prediction you'd use is the newest row of the
             buffer, which always has the full history (that is what predict_file does).
  --stride   rows between the starts of consecutive TRAINING windows. Default: window - warmup (16),
             so the scored parts of consecutive windows tile the recording: every row is scored exactly
             once, with the previous window's scored rows reused only as history. A smaller stride
             scores rows several times (more, but repeated, training samples). A final window is added
             at the end of each recording so its last rows are used too.
  --eval_stride  window step for val/test (default window - warmup: every row after the first
             `warmup` rows of a recording is scored exactly once)
  Recordings shorter than --window are skipped (and listed).

What is predicted (--target, one or more; one shared network, the losses are added):
  imu            regression (default): derived/imu/data/gravity_removed linear_acceleration.x/y/z,
                 angular_velocity.x/y/z and depth (global_position/local pose z) -> 7 outputs
  turn_rate_avg  regression: derived/turn.turn_rate_avg, the heading rate averaged over the last 5 s
                 (rad/s, + = left; ≈ 0 going straight). Measured by the gyro.
  action         classification of the 'action' label: straight / right / left
  diam           classification of the 'approx_diam' label: the circle sizes found in the training files
                 (e.g. 2.5, 5, 7.5 m), and straight (-1, written by the export; a blank diameter on a
                 straight row is also taken as straight). Sizes that appear only in val/test files can't
                 be predicted: those rows are left out and listed.
Classes are predicted at every row. Rows with a blank label or blank regression value are left out
of that target's loss. Unlabelled recordings (blank action / approx_diam, or no label columns at all)
can be mixed in freely: they still train the measured targets (imu, turn_rate_avg) and are simply
not scored on the classes. e.g. --target imu turn_rate_avg action diam

Inputs = every column in the files EXCEPT the IMU, depth and turn columns (any column of those
topics, for every target: they would leak the answer), the label columns (action, approx_diam),
IGNORE_INPUTS (t_s, dt_s) and --ignore_inputs. That is every "derived/tf[tag_*...]" column (per tag:
translation_mm.x/y/z, rotation.x/y/z/w and the seen flag) plus anything else you add to the export.

Missed tag detections: each tag's <tag>.seen column is 1 when the tag was detected in that row and 0
when it wasn't. A missed tag holds its last detected value in the RECORDING (only earlier rows are
used, so it works live with a running buffer), or its zero pose (0 mm, identity rotation) before
its first detection. The seen flag tells the model which rows are real.

At the end, the train/val score per epoch is saved next to the --out checkpoint as
<out>_curve.png (needs matplotlib; skipped if it isn't installed).

Models (--model): rnn (plain Elman), gru, lstm. "Stateless" = nothing is carried between windows.

Usage:
    python train_rnn.py --data_dir data/all --model gru
    python train_rnn.py --train_dir data/train --val_dir data/val --test_dir data/test --model gru
    python train_rnn.py --data_dir data/all --target action diam            (window 32, warmup 16, stride 16)
    python train_rnn.py --data_dir data/all --window 65 --warmup 32         (1 s buffer)
    python train_rnn.py --train_dir ... --val_dir ... --target action diam turn_rate_avg --balance_classes
"""

import argparse
import glob
import os
import random
import re
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Never inputs: t_s (time since the export started) would let the model learn "15 s into a run
# usually means deep" instead of reading the tags; dt_s is frame timing. Add names with --ignore_inputs.
IGNORE_INPUTS = ["t_s", "dt_s"]
DROP_TAGS: set = set()          # tag numbers left out of the inputs entirely (--drop_tags)

IMU_PREFIX = "derived/imu/data/gravity_removed."
IMU_FIELDS = [f"linear_acceleration.{a}" for a in "xyz"] + [f"angular_velocity.{a}" for a in "xyz"]
DEPTH_COL = "mavros/global_position/local.pose.pose.position.z"      # metres, negative underwater
TURN_COL = "derived/turn.turn_rate_avg"                              # rad/s, + = left, last 5 s
# Topics the measured targets come from: none of their columns are ever inputs, for any target (they
# would leak the answer). The recorded IMU topic is listed too, in case an export ever contains it.
OUTPUT_TOPICS = (IMU_PREFIX, "mavros/imu/data.", "mavros/global_position/local.", "derived/turn.")

ACTION_COL, DIAM_COL = "action", "approx_diam"
LABEL_COLS = (ACTION_COL, DIAM_COL)
ACTIONS = ("straight", "right", "left")          # class index = position in this tuple
STRAIGHT_DIAM = -1.0                             # approx_diam of a straight run
TARGETS = ("imu", "turn_rate_avg", "action", "diam")

# Set by configure() at startup (or from a checkpoint)
OUTPUT_COLS: list = []          # regression outputs
HEADS: list = []                # classification heads: {"name", "col", "classes" (values), "names"}

# Old pre-cut window files: <recording>_w0000_rows0-129.csv. Used only for the overlap warning.
WINDOW_NAME = re.compile(r"^(?P<rec>.+)_w\d+_rows\d+-\d+\.csv$")


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm as _tqdm               # nicer bar if tqdm is installed
except ImportError:
    _tqdm = None


def progress(iterable, desc="", leave=False):
    """Wrap a loop in a progress bar: tqdm if installed, otherwise a simple text bar."""
    if _tqdm is not None:
        return _tqdm(iterable, desc=desc, leave=leave, dynamic_ncols=True)
    return _SimpleBar(iterable, desc, leave)


class _SimpleBar:
    """Minimal fallback bar: [#####.....] 12/40. Silent when output isn't a terminal."""

    def __init__(self, iterable, desc, leave):
        self.it, self.desc, self.leave = iterable, desc, leave
        try:
            self.total = len(iterable)
        except TypeError:
            self.total = None

    def __iter__(self):
        if not self.total or not sys.stdout.isatty():
            yield from self.it
            return
        step = max(1, self.total // 50)
        for i, x in enumerate(self.it, 1):
            yield x
            if i % step == 0 or i == self.total:
                done = int(30 * i / self.total)
                sys.stdout.write(f"\r{self.desc} [{'#' * done}{'.' * (30 - done)}] {i}/{self.total}")
                sys.stdout.flush()
        sys.stdout.write("\n" if self.leave else "\r" + " " * (len(self.desc) + 50) + "\r")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Columns and labels
# ---------------------------------------------------------------------------
def diam_key(v):
    """Diameter value -> a comparable class key (rounded, so '2.5' and '2.50' are the same class)."""
    return round(float(v), 3)


def diam_name(v):
    return "straight" if v == STRAIGHT_DIAM else f"{v:g} m"


def parse_action(series):
    """action column -> int64 class index per row (-1 = blank). Raises on unknown labels."""
    lab = series.fillna("").astype(str).str.strip().str.lower()
    bad = sorted(set(lab) - set(ACTIONS) - {""})
    if bad:
        raise ValueError(f"unknown action label(s) {bad}; expected {', '.join(ACTIONS)}")
    out = np.full(len(lab), -1, np.int64)
    for k, a in enumerate(ACTIONS):
        out[(lab == a).to_numpy()] = k
    return out


def parse_diam(df):
    """approx_diam per row as floats (NaN = blank). A straight row with a blank diameter -> -1."""
    d = pd.to_numeric(df[DIAM_COL], errors="coerce").to_numpy(float) if DIAM_COL in df.columns \
        else np.full(len(df), np.nan)
    if ACTION_COL in df.columns:
        straight = parse_action(df[ACTION_COL]) == ACTIONS.index("straight")
        d = np.where(straight & np.isnan(d), STRAIGHT_DIAM, d)
    return d


def diam_classes(files):
    """Circle sizes found in the training files (straight = -1 included if present), sorted."""
    vals = set()
    for f in files:
        cols = pd.read_csv(f, nrows=0).columns
        use = [c for c in (ACTION_COL, DIAM_COL) if c in cols]
        if not use:
            continue
        d = parse_diam(pd.read_csv(f, usecols=use))
        vals |= {diam_key(v) for v in d[np.isfinite(d)]}
    return sorted(vals)


def configure(header, targets, train_files):
    """Pick the regression outputs and classification heads for these targets. Sets OUTPUT_COLS and
    HEADS (module globals, so load_csv / is_input_col use them) and returns both."""
    global OUTPUT_COLS, HEADS
    OUTPUT_COLS, HEADS = [], []
    if "imu" in targets:
        OUTPUT_COLS += [IMU_PREFIX + f for f in IMU_FIELDS] + [DEPTH_COL]
    if "turn_rate_avg" in targets:
        OUTPUT_COLS.append(TURN_COL)
    missing = [c for c in OUTPUT_COLS if c not in header]
    if missing:
        raise SystemExit(f"Training files lack target columns, e.g. {missing[:2]}. Re-export them in "
                         "tag_to_imu_format with the current bagsync.")
    if "action" in targets:
        HEADS.append({"name": "action", "col": ACTION_COL, "classes": list(ACTIONS), "names": list(ACTIONS)})
    if "diam" in targets:
        cl = diam_classes(train_files)
        if len(cl) < 2:
            raise SystemExit(f"--target diam needs at least two circle sizes (or straight) in the training "
                             f"files; found {[diam_name(v) for v in cl]}")
        HEADS.append({"name": "diam", "col": DIAM_COL, "classes": cl, "names": [diam_name(v) for v in cl]})
    return OUTPUT_COLS, HEADS


def tag_number(c):
    """'derived/tf[tag_3_zero->tag_3].translation_mm.x' -> 3 (None for other columns)."""
    t = tag_of(c)
    m = re.search(r"(\d+)\]$", t) if t else None
    return int(m.group(1)) if m else None


def is_input_col(c):
    """Every column is an input except the outputs, any IMU / depth / turn topic column, the label
    columns, IGNORE_INPUTS and every column of a tag in DROP_TAGS."""
    return (c not in OUTPUT_COLS and not c.startswith(OUTPUT_TOPICS) and c not in LABEL_COLS
            and c not in IGNORE_INPUTS and not c.startswith("Unnamed:")
            and not (DROP_TAGS and tag_number(c) in DROP_TAGS))


def short_name(c):
    """'derived/imu/data/gravity_removed.angular_velocity.z' -> 'angular_velocity.z', etc."""
    return c.replace(IMU_PREFIX, "").replace("mavros/", "")


def is_seen_col(c):
    return c.startswith("derived/tf[") and c.endswith(".seen")


def tag_of(c):
    """'derived/tf[tag_3_zero->tag_3].translation_mm.x' -> 'derived/tf[tag_3_zero->tag_3]'."""
    return c[: c.index("]") + 1] if c.startswith("derived/tf[") else None


def recording_name(path):
    m = WINDOW_NAME.match(os.path.basename(path))
    return m["rec"] if m else os.path.splitext(os.path.basename(path))[0]


def fill_tag_gaps(X, in_cols):
    """Fill missed tag detections in raw units, in place. X: (N, T, n_in) with NaN gaps.

    A missed tag holds its last detected value (only earlier rows are used, so this works live).
    Before its first detection it gets its zero pose: 0 mm and the identity rotation (0, 0, 0, 1).
    """
    N, T, _ = X.shape
    for j, c in enumerate(in_cols):
        if not c.startswith("derived/tf[") or is_seen_col(c):
            continue
        v = X[:, :, j]
        ok = ~np.isnan(v)
        idx = np.where(ok, np.arange(T)[None, :], 0)
        np.maximum.accumulate(idx, axis=1, out=idx)        # row of the last detection so far
        held = np.take_along_axis(v, idx, axis=1)
        held[~np.maximum.accumulate(ok, axis=1)] = 1.0 if c.endswith("rotation.w") else 0.0
        X[:, :, j] = held
    return X


# ---------------------------------------------------------------------------
# Data loading: one recording per file
# ---------------------------------------------------------------------------
def load_csv(path, in_cols, out_cols, heads):
    """Read one recording (any number of rows).

    Returns X (T, n_in) and Y (T, n_reg) with NaN where a value is missing, C (T, n_heads) int64 class
    per head (-1 = no label), and a list of notes (e.g. circle sizes not among the training classes).
    A tag missing from the file entirely (never detected in that recording, so the export left it out)
    counts as never seen. Any other missing column means the file can't be used.
    """
    name = os.path.basename(path)
    df = pd.read_csv(path)
    absent = [c for c in in_cols if c not in df.columns]
    # (label columns may be missing or blank: unlabelled recordings still train the measured targets)
    missing = ([c for c in absent if not c.startswith("derived/tf[")]
               + [c for c in out_cols if c not in df.columns])
    if missing:
        raise ValueError(f"{name}: missing columns, e.g. {missing[:3]}")
    if absent:                                      # whole tags absent from this recording
        df = pd.concat([df, pd.DataFrame({c: (0.0 if is_seen_col(c) else np.nan) for c in absent},
                                         index=df.index)], axis=1)
    X = df[in_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    Y = df[list(out_cols)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    C = np.full((len(df), len(heads)), -1, np.int64)
    notes = []
    for h_i, h in enumerate(heads):
        if h["name"] == "action":
            if ACTION_COL not in df.columns:
                continue                            # no labels in this file: left out of this head's loss
            try:
                C[:, h_i] = parse_action(df[ACTION_COL])
            except ValueError as e:
                raise ValueError(f"{name}: {e}")
        else:
            d = parse_diam(df)
            lookup = {v: k for k, v in enumerate(h["classes"])}
            keys = [diam_key(v) if np.isfinite(v) else None for v in d]
            C[:, h_i] = [lookup.get(k, -1) if k is not None else -1 for k in keys]
            unseen = sorted({k for k in keys if k is not None and k not in lookup})
            if unseen:
                notes.append(f"{name}: circle size(s) {[diam_name(v) for v in unseen]} not in the training "
                             "files - those rows are left out of the diam score")
    return X.to_numpy(np.float32), np.array(Y.to_numpy(np.float32), copy=True), C, notes


def all_input_cols(files):
    """Input columns found in any of the files: tags in tag-number order, then the other inputs in
    the order first seen. Reports tags that are missing from some recordings."""
    per_tag, extra, where = {}, [], {}
    for f in progress(files, desc="reading headers"):
        cols = pd.read_csv(f, nrows=0).columns
        tags_here = set()
        for c in cols:
            if c.startswith("derived/tf["):
                t = tag_of(c)
                tags_here.add(t)
                lst = per_tag.setdefault(t, [])
                if c not in lst and is_input_col(c):
                    lst.append(c)
            elif is_input_col(c) and c not in extra:
                extra.append(c)
        where.setdefault(recording_name(f), set()).update(tags_here)
    num = lambda t: int(re.search(r"(\d+)\]$", t).group(1)) if re.search(r"(\d+)\]$", t) else 10**9  # noqa: E731
    tags = sorted(per_tag, key=lambda t: (num(t), t))
    gaps = {t: sorted(r for r, have in where.items() if t not in have) for t in tags}
    gaps = {t: r for t, r in gaps.items() if r}
    if gaps:
        print("Tags missing from some recordings (treated as never seen there, seen flag = 0):")
        for t, recs in gaps.items():
            print(f"  {t}: {len(recs)} recording(s), e.g. {', '.join(recs[:3])}")
    return [c for t in tags for c in per_tag[t]] + extra


def list_csvs(folder):
    """Sorted list of CSV paths in a folder ([] if no folder was given)."""
    if folder is None:
        return []
    folder = os.path.expanduser(folder)
    if not os.path.isdir(folder):
        raise SystemExit(f"Folder not found: {folder}")
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found in {folder}")
    return files


def check_overlap(sets):
    """Warn when the same recording appears in more than one folder."""
    where = defaultdict(set)
    for label, files in sets.items():
        for f in files:
            where[recording_name(f)].add(label)
    shared = {r: sorted(l) for r, l in where.items() if len(l) > 1}
    if shared:
        print(f"WARNING: {len(shared)} recording(s) appear in more than one folder, which makes "
              f"val/test scores look better than they are:")
        for r, labels in sorted(shared.items())[:10]:
            print(f"  {r}: {', '.join(labels)}")
        if len(shared) > 10:
            print(f"  ... and {len(shared) - 10} more")


SPLITS = ("train", "val", "test")


def recording_stratum(path):
    """The label combination a recording is balanced by: 'left 7.5 m', 'straight', 'left (no size)',
    'unlabelled'. Uses the most common non-blank label in the file."""
    cols = pd.read_csv(path, nrows=0).columns
    use = [c for c in (ACTION_COL, DIAM_COL) if c in cols]
    if ACTION_COL not in use:
        return "unlabelled"
    df = pd.read_csv(path, usecols=use)
    try:
        act = parse_action(df[ACTION_COL])
    except ValueError:
        return "unlabelled"
    act = act[act >= 0]
    if not len(act):
        return "unlabelled"
    a = ACTIONS[np.bincount(act).argmax()]
    if a == "straight":
        return "straight"
    d = parse_diam(df)
    d = d[np.isfinite(d)]
    if not len(d):
        return f"{a} (no size)"
    vals, cnt = np.unique([diam_key(v) for v in d], return_counts=True)
    return f"{a} {diam_name(vals[cnt.argmax()])}"


def split_recordings(data_dir, val_frac, test_frac, seed):
    """Split every CSV under data_dir by whole recording, balanced by label combination."""
    root = os.path.expanduser(data_dir)
    if not os.path.isdir(root):
        raise SystemExit(f"Folder not found: {data_dir}")
    files = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
    if not files:
        raise SystemExit(f"No CSV files found in {data_dir}")
    rel = {f: os.path.relpath(f, root).replace(os.sep, "/") for f in files}
    strata = {f: recording_stratum(f) for f in progress(files, desc="reading labels")}
    rng = random.Random(seed)
    fracs = {"train": 1 - val_frac - test_frac, "val": val_frac, "test": test_frac}
    if min(fracs.values()) < 0:
        raise SystemExit("--val_frac and --test_frac leave nothing for training")

    def balanced(recs, fr):
        """Share recordings out over the splits in proportions fr, label combination by label combination:
        whole shares first (largest remainder); the leftover recordings go to whichever split is furthest
        below its share OVERALL, so a set of 3-recording combinations alternates val / test instead of
        always picking the same one."""
        by_st = defaultdict(list)
        for f in recs:
            by_st[strata[f]].append(f)
        order = sorted(by_st)
        rng.shuffle(order)
        n_all = len(recs)
        glob_cnt = {s: 0 for s in SPLITS}
        out = {}
        for st in order:
            ks = sorted(by_st[st])
            rng.shuffle(ks)
            n = len(ks)
            base = {s: int(np.floor(fr[s] * n + 1e-9)) for s in SPLITS}
            for _ in range(n - sum(base.values())):
                s = max(SPLITS, key=lambda s: (round(fr[s] * n - base[s], 9),
                                               fr[s] * n_all - glob_cnt[s] - base[s], s == "test"))
                base[s] += 1
            i = 0
            for s in ("test", "val", "train"):
                for f in ks[i:i + base[s]]:
                    out[f] = s
                i += base[s]
                glob_cnt[s] += base[s]
        return out

    assign = balanced(files, fracs)
    sets = {s: sorted(f for f in files if assign[f] == s) for s in SPLITS}
    info = {"data_dir": os.path.abspath(root), "seed": seed, "val_frac": val_frac, "test_frac": test_frac,
            "splits": {s: [rel[f] for f in sets[s]] for s in SPLITS},
            "strata": {rel[f]: strata[f] for f in files}}
    return sets, info


def print_split(sets, info):
    """Recordings per split and label combination; flags combinations missing from val or test."""
    strata = info["strata"]
    root = info["data_dir"]
    rows = defaultdict(lambda: {s: 0 for s in SPLITS})
    for s in SPLITS:
        for f in sets[s]:
            rows[strata[os.path.relpath(f, root).replace(os.sep, "/")]][s] += 1
    print(f"Split of {sum(len(v) for v in sets.values())} recordings, seed {info['seed']}:")
    print(f"  {'label':18s}" + "".join(f"{s:>7s}" for s in SPLITS))
    for st in sorted(rows):
        r = rows[st]
        flag = "   <- none in " + " / ".join(s for s in ("val", "test") if not r[s]) \
            if st != "unlabelled" and (not r["val"] or not r["test"]) else ""
        print(f"  {st:18s}" + "".join(f"{r[s]:7d}" for s in SPLITS) + flag)
    print(f"  {'total':18s}" + "".join(f"{len(sets[s]):7d}" for s in SPLITS))


def load_split_file(path):
    """A saved <out>_split.json -> {split: [paths]}, info."""
    import json
    with open(os.path.expanduser(path)) as fh:
        info = json.load(fh)
    root = info["data_dir"]
    sets = {s: [os.path.join(root, r) for r in info["splits"][s]] for s in SPLITS}
    missing = [f for s in SPLITS for f in sets[s] if not os.path.isfile(f)]
    if missing:
        raise SystemExit(f"{len(missing)} file(s) in {path} not found, e.g. {missing[0]}")
    return sets, info


def load_files(files, in_cols, out_cols, heads, window, label="files"):
    """Load recordings. Returns a list of dicts {name, X, Y, C} (only those with >= window rows)."""
    recs, short = [], []
    for f in progress(files, desc=f"loading {label}"):
        try:
            X, Y, C, notes = load_csv(f, in_cols, out_cols, heads)
        except ValueError as e:
            print(f"  skipping {e}")
            continue
        for n in notes:
            print(f"  {label}: {n}")
        if len(X) < window:
            short.append(f"{os.path.basename(f)} ({len(X)} rows)")
            continue
        recs.append({"name": os.path.basename(f), "X": X, "Y": Y, "C": C})
    if short:
        print(f"  {label}: skipped {len(short)} recording(s) shorter than the {window}-row window: "
              f"{', '.join(short[:5])}{' ...' if len(short) > 5 else ''}")
    if files and not recs:
        raise SystemExit(f"No usable files in {os.path.dirname(files[0])}")
    return recs


def report_missing(label, recs, in_cols, out_cols, heads):
    """Print how often tags were missed, how many cells are empty and the label balance.
    Call it before the gaps are filled."""
    X = np.concatenate([r["X"] for r in recs])
    seen = [j for j, c in enumerate(in_cols) if is_seen_col(c)]
    if seen:
        S = X[:, seen]
        missed = 1.0 - np.nanmean(S, axis=0)
        never = np.array([(np.nan_to_num(r["X"][:, seen]) == 0).all(0) for r in recs]).sum(0)
        print(f"  {label}: tags missed in {1 - np.nanmean(S):.2%} of rows overall "
              f"(held at the last detection or zero pose; seen flag = 0).")
        worst = [k for k in np.argsort(missed)[::-1][:5] if missed[k] > 0]
        if worst:
            print("    Most often missed:")
            for k in worst:
                print(f"      {tag_of(in_cols[seen[k]])}: {missed[k]:.1%} of rows, "
                      f"never seen in {never[k]} recording(s)")
    other = [j for j, c in enumerate(in_cols) if not c.startswith("derived/tf[")]
    if other and np.isnan(X[:, other]).any():
        print(f"  {label}: {np.isnan(X[:, other]).mean():.2%} of non-tag input cells empty "
              f"({', '.join(in_cols[j] for j in other)}); set to the training mean.")
    if out_cols:
        Y = np.concatenate([r["Y"] for r in recs])
        if np.isnan(Y).any():
            print(f"  {label}: {np.isnan(Y).mean():.2%} of regression target cells empty (left out of the loss)")
        dead = np.where(np.isnan(Y).all(0))[0]
        if len(dead):
            print(f"  {label}: no values at all for {[out_cols[i] for i in dead]}")
    for h_i, h in enumerate(heads):
        per = defaultdict(int)
        for r in recs:
            c = r["C"][:, h_i]
            k = c[0] if len(c) and (c == c[0]).all() else None
            per["mixed" if k is None else ("(blank)" if k < 0 else h["names"][k])] += 1
        C = np.concatenate([r["C"][:, h_i] for r in recs])
        print(f"  {label}: {h['name']} labels by recording: "
              + ", ".join(f"{k} {v}" for k, v in sorted(per.items()))
              + "  |  rows: " + ", ".join(f"{n} {(C == k).sum()}" for k, n in enumerate(h["names"]))
              + f", blank {(C < 0).sum()}")


def prepare_recording(rec, stats):
    """Raw recording -> tensors: normalized inputs (gaps filled), normalized targets, mask, classes."""
    X = fill_tag_gaps(np.array(rec["X"][None], np.float32, copy=True), stats["in_cols"])[0]
    Xn = np.nan_to_num((X - stats["x_mean"]) / stats["x_std"], nan=0.0, posinf=0.0, neginf=0.0)
    Y = rec["Y"]
    mask = ~np.isnan(Y)
    Yn = np.nan_to_num((Y - stats["y_mean"]) / stats["y_std"])
    return {"name": rec["name"], "X": torch.from_numpy(Xn.astype(np.float32)),
            "Y": torch.from_numpy(Yn.astype(np.float32)), "M": torch.from_numpy(mask.astype(np.float32)),
            "C": torch.from_numpy(rec["C"].astype(np.int64))}


def window_starts(T, window, stride):
    """Starts of the windows over T rows: every `stride` rows, plus one ending at the last row."""
    s = list(range(0, T - window + 1, stride))
    if s and s[-1] != T - window:
        s.append(T - window)
    return s


class WindowDataset(Dataset):
    """Windows of `window` rows cut on the fly from whole recordings.

    Each item: x (window, n_in), y (window, n_reg), m (window, n_reg) 1 = scored, c (window, n_heads)
    class or -1 = not scored. The first `warmup` rows of every window are never scored. With
    dedupe=True (val/test), rows already scored by an earlier window of the same recording aren't
    scored again, so every row counts once."""

    def __init__(self, recs, window, stride, warmup, dedupe=False):
        self.recs, self.window, self.warmup = recs, window, warmup
        self.items = []                        # (recording index, start, first scored row in window)
        for i, r in enumerate(recs):
            covered = 0
            for s in window_starts(len(r["X"]), window, stride):
                # val/test: never score a row twice. Training: the extra window added at the end of a
                # recording only scores the rows no earlier window reached
                extra = s % stride != 0
                first = max(warmup, covered - s) if (dedupe or extra) else warmup
                if first < window:
                    self.items.append((i, s, first))
                covered = s + window

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        i, s, first = self.items[k]
        r, e = self.recs[i], s + self.window
        m = r["M"][s:e].clone()
        c = r["C"][s:e].clone()
        m[:first] = 0
        c[:first] = -1
        return r["X"][s:e], r["Y"][s:e], m, c


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class InputPrep(nn.Module):
    """Optional input preprocessing shared by all models (no learned weights).

    use_diff:   append the change in every input since the previous row (how fast each tag is
                deforming). Only looks backwards, so it works live. The first row gets 0.
    in_dropout: during training, randomly zero this fraction of input values. Off at eval.
    """

    def __init__(self, n_in, use_diff=False, in_dropout=0.0):
        super().__init__()
        self.use_diff = use_diff
        self.drop = nn.Dropout(in_dropout)
        self.n_feat = n_in * 2 if use_diff else n_in

    def forward(self, x):                              # x: (B, T, n_in)
        if self.use_diff:
            d = torch.diff(x, dim=1, prepend=x[:, :1])
            x = torch.cat([x, d], dim=-1)
        return self.drop(x)


class RecurrentModel(nn.Module):
    """RNN / GRU / LSTM followed by a linear layer applied at every time step. The output holds the
    regression outputs first, then one block of logits per classification head."""

    def __init__(self, n_in, n_out, hidden, layers, kind, dropout,
                 use_diff=False, in_dropout=0.0):
        super().__init__()
        self.prep = InputPrep(n_in, use_diff, in_dropout)
        rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[kind]
        self.rnn = rnn_cls(self.prep.n_feat, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x):                # x: (B, T, n_in)
        x = self.prep(x)
        out, _ = self.rnn(x)             # no initial state passed: zeros every call (stateless)
        return self.head(self.drop(out)) # (B, T, n_out)


def build_model(n_in, n_out, a):
    """Create the model chosen by --model, using the other args for its size."""
    # Keep use_diff support for existing checkpoints; it is no longer a training CLI option.
    extra = dict(use_diff=a.get("use_diff", False), in_dropout=a.get("in_dropout", 0.0))
    return RecurrentModel(n_in, n_out, a["hidden"], a["layers"], a["model"], a["dropout"], **extra)


def head_slices(n_reg, heads):
    """(start, stop) of each head's logits in the model output."""
    out, k = [], n_reg
    for h in heads:
        out.append((k, k + len(h["classes"])))
        k += len(h["classes"])
    return out


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------
def run_epoch(model, loader, device, y_std, n_reg, heads, optimizer=None, clip=1.0, desc=None,
              class_w=None, select="bacc"):
    """One pass over a loader. Trains if an optimizer is given, otherwise just scores.

    Regression uses MSE on scaled outputs (each divided by its training std, so every output counts
    equally). Each classification head uses cross-entropy (optionally class-weighted, class_w = one
    weight tensor or None per head). All parts are added. Only scored rows count (see WindowDataset).
    Returns a dict: scaled (scaled RMSE), rmse (original units), heads (per head: ce = mean
    cross-entropy unweighted, acc = row accuracy, bacc = balanced accuracy = the mean of each class's
    recall, conf = confusion matrix), score (what picks the best epoch, lower is better): scaled RMSE
    plus, per head, 1 - bacc (select="bacc") or ce (select="ce").
    """
    training = optimizer is not None
    model.train(training)
    class_w = class_w or [None] * len(heads)
    sl = head_slices(n_reg, heads)
    sum_sq_norm = sum_sq_orig = n_vals = 0.0
    hs = [{"ce": 0.0, "n": 0.0, "right": 0.0, "conf": np.zeros((len(h["classes"]),) * 2, np.int64)} for h in heads]
    batches = progress(loader, desc=desc) if desc else loader
    with torch.set_grad_enabled(training):
        for x, y, m, c in batches:
            x, y, m, c = x.to(device), y.to(device), m.to(device), c.to(device)
            pred = model(x)
            loss = torch.zeros((), device=device)
            if n_reg:
                pr = pred[..., :n_reg]
                sq = ((pr - y) ** 2) * m
                loss = loss + sq.sum() / m.sum().clamp(min=1)
                sum_sq_norm += sq.sum().item()
                sum_sq_orig += (sq.detach() * y_std ** 2).sum().item()
                n_vals += m.sum().item()
            for h_i, (a, b) in enumerate(sl):
                logits = pred[..., a:b].reshape(-1, b - a)
                cf = c[..., h_i].reshape(-1)
                lab = cf >= 0
                n = lab.sum()
                if n == 0:
                    continue
                loss = loss + nn.functional.cross_entropy(logits[lab], cf[lab], weight=class_w[h_i])
                with torch.no_grad():
                    st = hs[h_i]
                    st["ce"] += nn.functional.cross_entropy(logits[lab], cf[lab], reduction="sum").item()
                    guess = logits[lab].argmax(1)
                    st["right"] += (guess == cf[lab]).sum().item()
                    st["n"] += n.item()
                    np.add.at(st["conf"], (cf[lab].cpu().numpy(), guess.cpu().numpy()), 1)
            if not torch.isfinite(loss):
                raise SystemExit("Loss became NaN/inf. Try a lower --lr (e.g. 0.0003).")
            if training and loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
    def bacc(cm):
        rows = cm.sum(1)
        return float(np.mean(np.diag(cm)[rows > 0] / rows[rows > 0])) if rows.sum() else float("nan")

    out = {"scaled": (sum_sq_norm / n_vals) ** 0.5 if n_vals else float("nan"),
           "rmse": (sum_sq_orig / n_vals) ** 0.5 if n_vals else float("nan"),
           "heads": [{"ce": st["ce"] / st["n"] if st["n"] else float("nan"),
                      "acc": st["right"] / st["n"] if st["n"] else float("nan"),
                      "bacc": bacc(st["conf"]), "conf": st["conf"]} for st in hs]}
    part = [(1 - h["bacc"]) if select == "bacc" else h["ce"] for h in out["heads"]]
    out["score"] = (out["scaled"] if n_reg else 0.0) + sum(v for v in part if v == v)
    return out


def fmt_scores(r, n_reg, heads):
    s = []
    if n_reg:
        s.append(f"scaled {r['scaled']:.4f} (RMSE {r['rmse']:.4f})")
    for h, hr in zip(heads, r["heads"]):
        s.append(f"{h['name']} CE {hr['ce']:.4f} acc {hr['acc']:.1%} bal {hr['bacc']:.1%}")
    return ", ".join(s)


@torch.no_grad()
def per_output_rmse(model, loader, device, y_std, n_reg):
    """RMSE for each regression output separately, in original units."""
    model.eval()
    sq, n = 0, 0
    for x, y, m, _ in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        e = ((model(x)[..., :n_reg] - y) * y_std) ** 2 * m
        sq = sq + e.sum((0, 1))
        n = n + m.sum((0, 1))
    return (sq / n.clamp(min=1)).sqrt().cpu().numpy()


@torch.no_grad()
def recording_report(model, recs, stats, args, n_reg, heads, device, y_std_t, label, only_below=None):
    """Per-recording scores, so errors can be traced to the files that cause them: for each recording
    and classification head, the true class, the share of scored frames predicted right and, when
    wrong, what it was mostly called. Also lists tags never detected in that recording.
    only_below: list only recordings with a head scoring below this accuracy."""
    seen_cols = [j for j, c in enumerate(stats["in_cols"]) if is_seen_col(c)]
    rows = []
    for r in recs:
        prepped = prepare_recording(r, stats)
        ds = WindowDataset([prepped], args.window, args.eval_stride, args.warmup, dedupe=True)
        if not len(ds):
            continue
        res = run_epoch(model, DataLoader(ds, batch_size=args.batch_size), device, y_std_t, n_reg, heads)
        parts, worst = [], 1.0
        for h, hr in zip(heads, res["heads"]):
            cm = hr["conf"]
            if cm.sum() == 0:
                parts.append(f"{h['name']}: (no label)")
                continue
            k = int(cm.sum(1).argmax())                     # the file's (main) true class
            acc = cm[k, k] / cm[k].sum()
            worst = min(worst, acc)
            txt = f"{h['name']}: {h['names'][k]} {acc:6.1%}"
            wrong = cm[k].copy()
            wrong[k] = 0
            if wrong.sum():
                j = int(wrong.argmax())
                txt += f" (else mostly {h['names'][j]} {wrong[j] / cm[k].sum():.0%})"
            parts.append(txt)
        never = [tag_number(stats["in_cols"][j]) for j in seen_cols if np.nansum(r["X"][:, j]) == 0]
        rows.append((worst, r["name"], parts, never))
    shown = [x for x in rows if only_below is None or x[0] < only_below]
    head = f"\nPer recording ({label}" + (f", only those below {only_below:.0%} on some target" if only_below else "") \
        + "; worst first):"
    print(head if shown else head + "  none")
    for worst, name, parts, never in sorted(shown, key=lambda x: x[0]):
        print(f"  {name}")
        print("      " + "   |   ".join(parts) + (f"   |   tags never seen: {never}" if never else ""))


def save_curve(history, path, ylabel):
    """Plot train (and val) score per epoch and save it as a PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping the plot")
        return
    ep = range(1, len(history["train"]) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(ep, history["train"], label="train")
    if history["val"]:
        plt.plot(ep, history["val"], label="val")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"\nCurve saved to {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", help="folder of ALL recordings (subfolders included): split automatically "
                                      "by recording, balanced by label")
    p.add_argument("--val_frac", type=float, default=0.15, help="--data_dir: share of recordings for val (0.15)")
    p.add_argument("--test_frac", type=float, default=0.15, help="--data_dir: share of recordings for test (0.15)")
    p.add_argument("--split_file", help="reuse a split saved by an earlier run (<out>_split.json)")
    p.add_argument("--train_dir", help="folder of training recordings (instead of --data_dir)")
    p.add_argument("--val_dir", help="folder of validation recordings (picks the best epoch)")
    p.add_argument("--test_dir", help="folder of test recordings (scored once at the end)")
    p.add_argument("--target", nargs="+", choices=TARGETS, default=["imu"], metavar="TARGET",
                   help="one or more of: imu (IMU 6 + depth regression, default), turn_rate_avg (regression), "
                        "action (straight/right/left classes), diam (circle size classes)")
    p.add_argument("--balance_classes", action="store_true",
                   help="classification targets: weight each class by 1/its share of training rows, so a "
                        "rare class counts as much as a common one")
    p.add_argument("--window", type=int, default=32,
                   help="rows per window fed to the model (default 32 = 0.5 s at 65 fps)")
    p.add_argument("--stride", type=int,
                   help="rows between training window starts (default window - warmup: every row scored once)")
    p.add_argument("--warmup", type=int,
                   help="first rows of every window that are fed as history but not scored (default window // 2)")
    p.add_argument("--eval_stride", type=int,
                   help="window step for val/test (default window - warmup: every row scored once)")
    p.add_argument("--model", choices=["rnn", "gru", "lstm"], default="rnn")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    p.add_argument("--hidden", type=int, default=128, help="hidden size")
    p.add_argument("--layers", type=int, default=1, help="number of stacked layers")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="fraction of hidden units randomly zeroed during training (e.g. 0.2)")
    p.add_argument("--weight_decay", type=float, default=0.0,
                   help="pulls weights toward zero each step (e.g. 1e-4 to 1e-2)")
    p.add_argument("--in_dropout", type=float, default=0.0,
                   help="fraction of input values randomly zeroed during training (e.g. 0.1)")
    p.add_argument("--drop_tags", default="", metavar="N,N",
                   help="tag numbers to leave out of the inputs entirely (all their columns, seen flag "
                        "included), e.g. 0,20 for stationary reference tags")
    p.add_argument("--select", choices=["bacc", "ce"], default="bacc",
                   help="what picks the best epoch for classification targets: bacc (default) = balanced "
                        "accuracy, the mean of each class's recall, so a rare class counts as much as a common "
                        "one; ce = cross-entropy, which also rises when the model grows over-confident even "
                        "while it gets more answers right")
    p.add_argument("--patience", type=int,
                   help="stop when the val score hasn't improved for this many epochs (default: run all epochs)")
    p.add_argument("--ignore_inputs", nargs="+", default=[], metavar="NAME",
                   help="more columns NOT to use as inputs (besides t_s, dt_s)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="checkpoint.pt", help="where to save the best model")
    args = p.parse_args()
    if args.warmup is None:
        args.warmup = args.window // 2
    if args.stride is None:
        args.stride = args.window - args.warmup
    if args.window < 2 or args.stride < 1 or not (0 <= args.warmup < args.window):
        p.error("need --window >= 2, --stride >= 1 and 0 <= --warmup < --window")
    if args.eval_stride is None:
        args.eval_stride = args.window - args.warmup
    args.target = list(dict.fromkeys(args.target))          # drop repeats, keep order

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    modes = [m for m, on in (("--data_dir", args.data_dir), ("--split_file", args.split_file),
                             ("--train_dir", args.train_dir)) if on]
    if len(modes) != 1:
        p.error("give exactly one of --data_dir, --split_file or --train_dir (+ --val_dir / --test_dir)")
    split_info = None
    if args.train_dir:
        sets = {"train": list_csvs(args.train_dir), "val": list_csvs(args.val_dir), "test": list_csvs(args.test_dir)}
        check_overlap(sets)
    else:
        if args.split_file:
            sets, split_info = load_split_file(args.split_file)
        else:
            sets, split_info = split_recordings(args.data_dir, args.val_frac, args.test_frac, args.seed)
        print_split(sets, split_info)
        if not sets["train"]:
            raise SystemExit("The split left no training recordings.")
        import json
        sp = f"{os.path.splitext(args.out)[0]}_split.json"
        os.makedirs(os.path.dirname(os.path.abspath(sp)), exist_ok=True)
        with open(sp, "w") as fh:
            json.dump(split_info, fh, indent=1)
        print(f"Split saved to {sp}")

    IGNORE_INPUTS.extend(args.ignore_inputs)
    DROP_TAGS.update(int(x) for x in args.drop_tags.split(",") if x.strip())
    if DROP_TAGS:
        print(f"Leaving out tags: {', '.join(map(str, sorted(DROP_TAGS)))}")
    if args.ignore_inputs:
        print(f"Also ignoring inputs: {', '.join(args.ignore_inputs)}")
    out_cols, heads = configure(set(pd.read_csv(sets["train"][0], nrows=0).columns), args.target, sets["train"])
    out_cols, heads = list(out_cols), list(heads)
    in_cols = all_input_cols(sets["train"])
    print(f"Targets: {', '.join(args.target)}"
          + (f"\n  regression: {', '.join(map(short_name, out_cols))}" if out_cols else "")
          + "".join(f"\n  {h['name']} classes: {', '.join(h['names'])}" for h in heads))
    print(f"Window {args.window} rows, train stride {args.stride}, eval stride {args.eval_stride}, "
          f"warmup {args.warmup}")
    recs = {k: load_files(f, in_cols, out_cols, heads, args.window, label=k) for k, f in sets.items()}
    n_in, n_reg = len(in_cols), len(out_cols)
    n_cls = sum(len(h["classes"]) for h in heads)

    print("Checking the data:")
    for label, rr in recs.items():
        if rr:
            report_missing(label, rr, in_cols, out_cols, heads)

    # Normalize each column using training rows only (each row once), from real readings
    # (missed tags are still NaN here). Columns that barely change get mean 0 / std 1.
    Xall = np.concatenate([r["X"] for r in recs["train"]])
    Yall = np.concatenate([r["Y"] for r in recs["train"]]) if n_reg else np.zeros((1, 0), np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        x_mean, x_std = np.nanmean(Xall, 0), np.nanstd(Xall, 0)
        y_mean, y_std = np.nanmean(Yall, 0), np.nanstd(Yall, 0)
    x_mean = np.nan_to_num(x_mean).astype(np.float32)
    x_std = np.where(~np.isfinite(x_std) | (x_std < 1e-6), 1.0, x_std).astype(np.float32)
    y_mean = np.nan_to_num(y_mean).astype(np.float32)
    y_std = np.where(~np.isfinite(y_std) | (y_std < 1e-8), 1.0, y_std).astype(np.float32)
    stats = dict(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std, in_cols=in_cols, out_cols=out_cols,
                 seq_len=args.window, warmup=args.warmup, targets=args.target, heads=heads)

    class_w = []
    for h_i, h in enumerate(heads):
        C = np.concatenate([r["C"][:, h_i] for r in recs["train"]])
        counts = np.bincount(C[C >= 0], minlength=len(h["classes"])).astype(float)
        if (counts == 0).any():
            print(f"WARNING: no training rows for {h['name']} class(es) "
                  f"{[n for n, k in zip(h['names'], counts) if k == 0]}; the model can't learn them.")
        h["train_counts"] = counts.tolist()
        if args.balance_classes:
            w = np.where(counts > 0, counts.sum() / (len(counts) * np.maximum(counts, 1)), 0.0)
            class_w.append(torch.tensor(w, dtype=torch.float32, device=device))
            print(f"{h['name']} class weights: " + ", ".join(f"{n} {x:.2f}" for n, x in zip(h["names"], w)))
        else:
            class_w.append(None)

    def make_loader(rr, train):
        if not rr:
            return None
        prepped = [prepare_recording(r, stats) for r in rr]
        ds = WindowDataset(prepped, args.window, args.stride if train else args.eval_stride, args.warmup,
                           dedupe=not train)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=train)

    train_dl, val_dl, test_dl = (make_loader(recs["train"], True), make_loader(recs["val"], False),
                                 make_loader(recs["test"], False))
    print(f"{len(recs['train'])} train / {len(recs['val'])} val / {len(recs['test'])} test recordings -> "
          f"{len(train_dl.dataset)} / {len(val_dl.dataset) if val_dl else 0} / "
          f"{len(test_dl.dataset) if test_dl else 0} windows, {n_in} inputs -> {n_reg} regression + "
          f"{n_cls} class outputs, model={args.model}, dropout={args.dropout}, weight_decay={args.weight_decay}, "
          f"in_dropout={args.in_dropout}, regression_loss=MSE, device={device}")

    model = build_model(n_in, n_reg + n_cls, vars(args)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    y_std_t = torch.from_numpy(y_std).to(device)
    kw = dict(n_reg=n_reg, heads=heads, select=args.select)

    # Keep the epoch with the lowest val score (train if no val): scaled RMSE (each output in units of
    # its training std; 1.0 ~ no better than the mean) plus, per classification head, 1 - balanced
    # accuracy (--select bacc) or its cross-entropy (--select ce).
    history = {"train": [], "val": []}
    best, best_epoch = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_dl, device, y_std_t, optimizer=optimizer, desc=f"epoch {epoch}/{args.epochs} train",
                       class_w=class_w, **kw)
        history["train"].append(tr["score"])
        msg = f"epoch {epoch:3d}/{args.epochs}  train {fmt_scores(tr, n_reg, heads)}"
        score = tr["score"]
        if val_dl:
            va = run_epoch(model, val_dl, device, y_std_t, desc=f"epoch {epoch}/{args.epochs} val", **kw)
            history["val"].append(va["score"])
            msg += f"  val {fmt_scores(va, n_reg, heads)}"
            score = va["score"]
        if score < best:
            best, best_epoch = score, epoch
            torch.save({"model": model.state_dict(), "stats": stats, "args": vars(args)}, args.out)
            msg += "  *saved"
        print(msg)
        if args.patience and val_dl and epoch - best_epoch >= args.patience:
            print(f"Stopping early: val hasn't improved for {args.patience} epochs (best: epoch {best_epoch})")
            break

    model, _ = load_model(args.out, device)
    loaders = [(n, dl) for n, dl in (("train", train_dl), ("val", val_dl), ("test", test_dl)) if dl]
    print("\nFinal scores of the saved model:")
    per_out, results = {}, {}
    for name, dl in loaders:
        r = run_epoch(model, dl, device, y_std_t, **kw)
        results[name] = r
        if n_reg:
            per_out[name] = per_output_rmse(model, dl, device, y_std_t, n_reg)
        print(f"  {name:5s} {fmt_scores(r, n_reg, heads)}")

    if n_reg:
        print("\nRMSE per output (original units; in brackets: scaled, 1.0 ~ no better than the mean):")
        print(f"  {'output':52s}" + "".join(f"{n:>18s}" for n, _ in loaders))
        for i, col in enumerate(out_cols):
            print(f"  {col:52s}" + "".join(f"{per_out[n][i]:10.4f} ({per_out[n][i] / y_std[i]:4.2f})"
                                           for n, _ in loaders))
    for h_i, h in enumerate(heads):
        for name, _ in loaders:
            cm = results[name]["heads"][h_i]["conf"]
            print(f"\n{h['name']} confusion matrix ({name}, rows = true, columns = predicted, per row of data):")
            print("  " + " " * 10 + "".join(f"{a:>10s}" for a in h["names"]) + "    recall")
            for k, a in enumerate(h["names"]):
                rec = cm[k, k] / cm[k].sum() if cm[k].sum() else float("nan")
                print(f"  {a:10s}" + "".join(f"{v:10d}" for v in cm[k]) + f"    {rec:6.1%}")

    if heads:
        recording_report(model, recs["train"], stats, args, n_reg, heads, device, y_std_t, "train", only_below=0.9)
        for name in ("val", "test"):
            if recs[name]:
                recording_report(model, recs[name], stats, args, n_reg, heads, device, y_std_t, name)

    base = os.path.splitext(args.out)[0]
    save_curve(history, f"{base}_curve.png",
               " + ".join((["scaled RMSE"] if n_reg else [])
                         + [f"{h['name']} " + ("(1 - bal. acc)" if args.select == "bacc" else "CE") for h in heads]))


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def load_model(path, device="cpu"):
    """Rebuild a saved model and return it with its stats. Also sets the column globals so load_csv
    and is_input_col behave as they did in training. Older checkpoints still load."""
    global OUTPUT_COLS, HEADS
    ckpt = torch.load(path, map_location=device, weights_only=False)
    s = ckpt["stats"]
    s.setdefault("seq_len", 65)
    s.setdefault("warmup", 0)
    if "heads" not in s:                        # checkpoints from before the multi-target version
        acts = s.get("actions", list(ACTIONS))
        s["heads"] = ([{"name": "action", "col": ACTION_COL, "classes": acts, "names": acts}]
                      if s.get("n_classes") else [])
    OUTPUT_COLS, HEADS = list(s["out_cols"]), s["heads"]
    n_out = len(s["out_cols"]) + sum(len(h["classes"]) for h in s["heads"])
    model = build_model(len(s["in_cols"]), n_out, ckpt["args"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), s


@torch.no_grad()
def predict_file(model, stats, csv_path, device="cpu", batch=512):
    """Predict a whole recording the way it would run live: at every row, the model is fed the last
    seq_len rows (the buffer, e.g. 32 frames = 0.5 s) ending at that row, and only the prediction for
    that newest row is kept. Rows before the first full buffer use the rows available so far (as at
    start-up live). One prediction per CSV row, ready to plot against the truth.

    Returns a DataFrame with one row per CSV row:
      t_s                              the CSV's time column (if present)
      <output>, true:<output>          each regression output (original units): predicted and true
      <head>_pred, true:<head>         each classification head: predicted and true class name
                                       (true is blank where the file has no label)
      p_<head>_<class>                 the predicted probability of each class
    """
    n_reg, heads = len(stats["out_cols"]), stats["heads"]
    out_cols = list(stats["out_cols"])
    raw = pd.read_csv(csv_path)
    have = [c for c in out_cols if c in raw.columns]
    X, Y, C, _ = load_csv(csv_path, stats["in_cols"], have, heads)
    rec = prepare_recording({"name": csv_path, "X": X, "Y": np.full((len(X), n_reg), np.nan, np.float32),
                             "C": C}, stats)
    T, L = len(X), stats["seq_len"]
    n_out = n_reg + sum(len(h["classes"]) for h in heads)
    out = np.full((T, n_out), np.nan, np.float32)
    x = rec["X"]
    # start-up: rows 0 .. L-2 only have the rows before them; one pass over the first rows gives
    # exactly that (the model is causal: row t only sees rows <= t)
    m0 = min(L - 1, T)
    if m0 > 0:
        out[:m0] = model(x[:m0][None].to(device))[0].cpu().numpy()
    # every later row: its own full buffer of the last L rows, keeping only the newest prediction
    ends = np.arange(L - 1, T)
    for i in range(0, len(ends), batch):
        e = ends[i:i + batch]
        xb = torch.stack([x[t - L + 1:t + 1] for t in e]).to(device)
        out[e] = model(xb)[:, -1].cpu().numpy()
    df = pd.DataFrame(index=range(T))
    if "t_s" in raw.columns:
        df["t_s"] = pd.to_numeric(raw["t_s"], errors="coerce").to_numpy()
    pred = out[:, :n_reg] * stats["y_std"] + stats["y_mean"]
    for j, c in enumerate(out_cols):
        df[c] = pred[:, j]
        df[f"true:{c}"] = Y[:, have.index(c)] if c in have else np.nan
    for h_i, (h, (a, b)) in enumerate(zip(heads, head_slices(n_reg, heads))):
        logits = out[:, a:b]
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        df[f"{h['name']}_pred"] = [h["names"][k] for k in p.argmax(1)]
        df[f"true:{h['name']}"] = [h["names"][k] if k >= 0 else "" for k in C[:, h_i]]
        for k, n in enumerate(h["names"]):
            df[f"p_{h['name']}_{n.replace(' ', '')}"] = p[:, k]
    return df


if __name__ == "__main__":
    main()
