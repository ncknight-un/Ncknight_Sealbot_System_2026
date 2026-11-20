"""
Segment classifier: whisker (AprilTag) deformations + IMU/gyro -> where an outside object or flow
disturbance is (PyTorch). For tag_imu_to_flow_class exports only (bagsync), whose last five columns
are the labels:  test, wall_side, dist_to_wall, angle_to_head, dist_to_head.

Two tests, each with its own pair of classifiers, named after the label columns they read:

  --test wall    wall_side      which side the wall is on (left / right / ...)
                 dist_to_wall   distance to the wall
  --test angle   angle_to_head  angle of the flow disturbance to the head
                 dist_to_head   distance of the flow disturbance from the head

The 'test' column says which test a recording belongs to (dist_to_wall = wall test, angle_to_head =
angle test). Only recordings of the chosen test are used, so a folder may hold both. If --test
isn't given and the folder holds one test, that one is used. Files without a 'test' column (e.g.
tag_to_imu_format exports, which belong to train_rnn.py) are skipped and listed.

Data = FULL tag_to_imu_format CSVs, one recording per file. No sliding window:
  1. every CSV under --data_dir (subfolders included) is split by WHOLE RECORDING into
     train / val / test (--val_frac / --test_frac), balanced by label combination
     ("wall_side left | dist_to_wall 0.3", ...), so one recording never ends up on both sides of the split
       --seed             the split is random but repeatable
       --split_file PATH  reuse a saved split instead of making one
     The split used is saved next to the checkpoint as <out>_split.json.
  2. AFTER the split, each recording is cut into consecutive, NON-OVERLAPPING segments of
     --seg_len frames (default 16 = ~0.25 s at 65 fps): rows 0-15, 16-31, 32-47, ...
     Leftover rows at the end of a recording (fewer than --seg_len) are dropped, and recordings
     shorter than --seg_len are skipped (both are reported).
  3. Each segment is one sample: the model reads its --seg_len frames starting from a zero state
     and gives ONE prediction per head for the whole segment (from the last frame's hidden state).
     Nothing is carried from one segment to the next.
  - train: used to fit the model
  - val:   checked every epoch; the epoch with the best val score is saved
  - test:  never touched during training; scored once at the end

What is predicted (--target; default: both heads of the test; one shared network, the losses are
added). A head needs at least two different label values in the TRAINING recordings; with only one
(e.g. every recording at the same distance) that head is left out with a warning and the run
continues on the other. To train one head only:  --target wall_side   /   --target angle_to_head
The classes of a head are the label values found in the training recordings, numbers or text
(left / right / none, -45 / 0 / 45, 0.12 / 0.3). Values that appear only in val/test can't be
predicted: those segments are left out of that head's score and listed.

A blank label cell means "no label": the segment is left out of that head's loss. So a no-wall run
must say so in words (e.g. wall_side = none) to count as a class; the text "none" is not treated
as blank. A segment's label is the most common non-blank label of its rows.

Inputs (--inputs, default: tags imu):
  tags   every "derived/tf[tag_*...]" column (per tag: translation_mm.x/y/z, rotation.x/y/z/w and
         the seen flag)
  imu    derived/imu/data/gravity_removed linear_acceleration.x/y/z and angular_velocity.x/y/z
  --extra_inputs NAME ...  any other columns to feed in as well
The depth (mavros/global_position/local...position.z) and derived/turn.turn_rate_avg columns of the
export are NOT fed in unless you add them with --extra_inputs.
  seen   ONLY the tags' seen flags, no poses (a check: can the classes be told apart just from
         which tags drop out? It should score near the baseline.)
  --center_recordings  subtract each recording's own mean from every tag pose column, so only the
         movement within the recording is left and any constant offset is gone (a check: is the
         model reading a fixed tag offset that differs between conditions / sessions, or the way
         the tags move? Not usable live as is: it needs the whole recording's mean.)
  --day_baseline       subtract, from every tag pose column of every recording, that DAY's no-object
         baseline: the mean pose over the day's TRAINING recordings labelled --baseline_label (default
         none). Each recording then shows how far the whiskers are from "nothing there" on its own
         day, so a constant offset that differs between days (tags re-zeroed, rig moved) cancels, while
         a steady deflection caused by the object stays. Every day in the data needs at least one such
         recording in the training set, or the script stops and says which day lacks one. Days are read
         from the file name (--day regex). Can't be combined with --center_recordings.
Nothing else is ever an input, so label columns, t_s and dt_s can't leak in. Running the same split
with  --inputs tags,  --inputs imu  and  --inputs tags imu  shows what each contributes: if
--inputs imu alone already scores well, the classes differ in how the ROV moved, not (only) in the flow.

Tag dropout (--tag_dropout P, training only): in every training segment each tag is, with probability
P, made to look missed for the whole segment: its pose holds the value of the segment's first frame
(as a real missed tag holds its last detection) and its seen flag reads 0. The model then can't lean
on any one tag, or on the pattern of which tags happened to drop out in a recording, so tags that
miss sometimes can stay in instead of being dropped with --drop_tags. It lowers that shortcut, it does
not remove it: a tag that misses far more in one class than another still differs between them.

Missed tag detections: each tag's <tag>.seen column is 1 when the tag was detected in that row and 0
when it wasn't. A missed tag holds its last detected value in the recording (earlier rows only, so it
works on a live stream), or its zero pose (0 mm, identity rotation) before its first detection. This
is done on the whole recording before it is cut. Empty IMU cells are set to the training mean.

At the end, the train/val score per epoch is saved as <out>_curve.png (needs matplotlib).

Usage:
    python train_lstm_flow.py --test wall  --data_dir data --out models_wall/wall.pt
    python train_lstm_flow.py --test angle --data_dir data --out models_angle/angle.pt
    python train_lstm_flow.py --test angle --data_dir data --seg_len 32 --out models_angle/angle_32.pt
    python train_lstm_flow.py --test wall  --data_dir data --target wall_side --drop_tags 0 --out models_wall/side.pt
    python train_lstm_flow.py --split_file models_wall/wall_split.json --inputs tags --out models_wall/wall_tags.pt
"""

import argparse
import glob
import json
import os
import random
import re
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FPS = 65.0
TAG_PREFIX = "derived/tf["
IMU_PREFIX = "derived/imu/data/gravity_removed."
IMU_FIELDS = [f"linear_acceleration.{a}" for a in "xyz"] + [f"angular_velocity.{a}" for a in "xyz"]
IMU_COLS = [IMU_PREFIX + f for f in IMU_FIELDS]

TEST_COL = "test"                       # which test a recording belongs to
TESTS = {"wall": ("wall_side", "dist_to_wall"), "angle": ("angle_to_head", "dist_to_head")}
TEST_VALUES = {"wall": "dist_to_wall", "angle": "angle_to_head"}       # what the test column says
TARGETS = TESTS["wall"] + TESTS["angle"]                # a head's name = the label column it reads
SPLITS = ("train", "val", "test")
DAY_RE = r"(?<!\d)(?:0?[1-9]|1[0-2])_(?:0?[1-9]|[12]\d|3[01])_(?:\d{4}|\d{2})(?!\d)"      # MM_DD_YYYY or MM_DD_YY in the file name

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None


def progress(iterable, desc=""):
    """tqdm bar if tqdm is installed, otherwise the plain loop."""
    return _tqdm(iterable, desc=desc, leave=False, dynamic_ncols=True) if _tqdm else iterable


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def label_keys(series):
    """A label column (read as text, see read_labels) -> one key per row ('' = blank). Numbers are
    rounded and written the same way ('45', '45.0' and 45.000 are one class); text is lower-cased."""
    num = pd.to_numeric(series, errors="coerce").to_numpy(float)
    txt = series.fillna("").astype(str).str.strip().str.lower().to_numpy()
    out = []
    for v, s in zip(num, txt):
        if np.isfinite(v):
            out.append(f"{round(float(v), 3) + 0.0:g}")
        else:
            out.append("" if s in ("", "nan", "inf", "-inf") else s)        # "none" is a class (no wall)
    return np.array(out, dtype=object)


def sort_classes(keys):
    """Numbers in numeric order, then text labels (e.g. 'none') alphabetically."""
    def key(k):
        try:
            return (0, float(k), "")
        except ValueError:
            return (1, 0.0, k)
    return sorted(keys, key=key)


def main_key(keys):
    """The most common non-blank key ('' if there is none)."""
    k = keys[keys != ""]
    if not len(k):
        return ""
    vals, cnt = np.unique(k.astype(str), return_counts=True)
    return str(vals[cnt.argmax()])


def read_labels(path, cols):
    """Label columns of a file as text. Read separately from the data because pandas would otherwise
    turn the text 'None' into an empty cell, and 'none' is a class here (no wall)."""
    have = [c for c in cols if c in pd.read_csv(path, nrows=0).columns]
    if not have:
        return pd.DataFrame()
    return pd.read_csv(path, usecols=have, dtype=str, keep_default_na=False)


def file_labels(path, cols):
    """{column: per-row keys} for the label columns present in a file."""
    df = read_labels(path, cols)
    return {c: label_keys(df[c]) for c in df.columns}


def file_test(path):
    """'wall' / 'angle' from a recording's test column; None if it has none or says something else."""
    lab = file_labels(path, [TEST_COL])
    value = main_key(lab[TEST_COL]) if TEST_COL in lab else ""
    return next((t for t, v in TEST_VALUES.items() if v == value), None)


def files_of_test(files, given):
    """Keep the recordings of one test. Returns (test, files). Lists what was left out."""
    kinds = {f: file_test(f) for f in progress(files, desc="reading test column")}
    other = [os.path.basename(f) for f, k in kinds.items() if k is None]
    if other:
        print(f"Skipping {len(other)} file(s) with no usable '{TEST_COL}' column (not tag_imu_to_flow_class "
              f"exports?), e.g. {', '.join(other[:3])}")
    found = sorted({k for k in kinds.values() if k})
    if not found:
        raise SystemExit(f"No tag_imu_to_flow_class recordings found: the '{TEST_COL}' column should say "
                         + " or ".join(TEST_VALUES.values()) + ".")
    if given is None:
        if len(found) > 1:
            raise SystemExit("This folder holds both wall-test and angle-test recordings: choose with "
                             "--test wall or --test angle.")
        given = found[0]
    keep = [f for f in files if kinds[f] == given]
    if not keep:
        raise SystemExit(f"No {given}-test recordings here (found only: {', '.join(found)}).")
    if len(found) > 1:
        print(f"Using the {len(keep)} {given}-test recordings; leaving out {sum(k not in (None, given) for k in kinds.values())} of the other test.")
    return given, keep


def prob_col(head, cls):
    """Name of a class-probability column in predict_file's output."""
    return f"p_{head}_{str(cls).replace(' ', '')}"


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------
def tag_of(c):
    """'derived/tf[tag_3_zero->tag_3].translation_mm.x' -> 'derived/tf[tag_3_zero->tag_3]'."""
    return c[: c.index("]") + 1] if c.startswith(TAG_PREFIX) and "]" in c else None


def tag_number(c):
    t = tag_of(c)
    m = re.search(r"(\d+)\]$", t) if t else None
    return int(m.group(1)) if m else None


def is_seen_col(c):
    return c.startswith(TAG_PREFIX) and c.endswith(".seen")


def short_name(c):
    return c.replace(IMU_PREFIX, "imu.").replace("mavros/", "")


def input_cols(files, inputs, extra, drop_tags):
    """The model's input columns: tags (every tag found in any training file, in tag-number order),
    then the six IMU fields, then --extra_inputs."""
    per_tag, where, header = {}, {}, set()
    for f in progress(files, desc="reading headers"):
        cols = list(pd.read_csv(f, nrows=0).columns)
        header |= set(cols)
        here = set()
        for c in cols:
            t = tag_of(c)
            if t is None or tag_number(c) in drop_tags:
                continue
            here.add(t)
            lst = per_tag.setdefault(t, [])
            if c not in lst:
                lst.append(c)
        where[os.path.basename(f)] = here
    out = []
    if "tags" in inputs or "seen" in inputs:
        tags = sorted(per_tag, key=lambda t: (tag_number(t + ".x") if tag_number(t + ".x") is not None else 10**9, t))
        if not tags:
            raise SystemExit("--inputs tags: no derived/tf[tag_...] columns in the training files")
        gaps = {t: sorted(r for r, have in where.items() if t not in have) for t in tags}
        gaps = {t: r for t, r in gaps.items() if r}
        if gaps:
            print("Tags missing from some recordings (treated as never seen there, seen flag = 0):")
            for t, recs in gaps.items():
                print(f"  {t}: {len(recs)} recording(s), e.g. {', '.join(recs[:3])}")
        out += [c for t in tags for c in per_tag[t] if "tags" in inputs or is_seen_col(c)]
    if "imu" in inputs:
        missing = [c for c in IMU_COLS if c not in header]
        if missing:
            raise SystemExit(f"--inputs imu: training files lack IMU columns, e.g. {missing[:2]}. Re-export "
                             "them in tag_to_imu_format (gravity-removed IMU).")
        out += IMU_COLS
    for c in extra:
        if c not in header:
            raise SystemExit(f"--extra_inputs {c}: no such column in the training files")
        if c not in out:
            out.append(c)
    return out


def day_of(path, pattern=DAY_RE):
    """The recording's day: MM_DD_YYYY / MM_DD_YY (or --day regex) in the file name, else the first NN_NN_NN there,
    else in the folder name, else the folder name itself."""
    base, folder = os.path.basename(path), os.path.basename(os.path.dirname(os.path.abspath(path)))
    m = re.search(pattern, base) or re.search(r"\d+_\d+_\d+", base) or re.search(pattern, folder)
    return m.group(0) if m else folder


def pose_cols(in_cols):
    """Indices of the tag pose columns (everything of a tag except its seen flag)."""
    return [j for j, c in enumerate(in_cols) if c.startswith(TAG_PREFIX) and not is_seen_col(c)]


def make_day_baselines(recs_train, in_cols, main_col, label, pattern):
    """{day: vector (n_in)} for --day_baseline: mean tag pose over each day's training recordings whose
    main label (wall_side / angle_to_head) is `label`; 0 for every non-pose column. Also returns how many
    recordings each day's baseline rests on."""
    pose = pose_cols(in_cols)
    rows, n_rec = defaultdict(list), defaultdict(int)
    for r in recs_train:
        lab = file_labels(r["path"], [main_col]).get(main_col)
        if lab is not None and main_key(lab) == label:
            d = day_of(r["path"], pattern)
            rows[d].append(r["X"][:, pose])
            n_rec[d] += 1
    vec = {}
    for d, parts in rows.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmean(np.concatenate(parts), axis=0)
        v = np.zeros(len(in_cols), np.float32)
        v[pose] = np.nan_to_num(m)
        vec[d] = v
    return vec, dict(n_rec)


def baseline_vector(stats, path):
    """The no-object baseline vector of this recording's day (None if the model wasn't trained with one)."""
    db = stats.get("day_baseline")
    if not db:
        return None
    day = day_of(path, db["day_re"])
    if day not in db["vectors"]:
        raise ValueError(f"{os.path.basename(path)}: day '{day}' has no baseline in this model "
                         f"(it has: {', '.join(sorted(db['vectors']))})")
    return db["vectors"][day]


def fill_tag_gaps(X, in_cols, deviation=False):
    """Fill missed tag detections in raw units, in place. X: (T, n_in) with NaN gaps. A missed tag
    holds its last detected value (earlier rows only); before its first detection it gets its zero
    pose: 0 mm and the identity rotation (0, 0, 0, 1)."""
    T = len(X)
    for j, c in enumerate(in_cols):
        if not c.startswith(TAG_PREFIX) or is_seen_col(c):
            continue
        v = X[:, j]
        ok = ~np.isnan(v)
        idx = np.maximum.accumulate(np.where(ok, np.arange(T), 0))
        held = v[idx]
        # before its first detection: the zero pose. If the poses were centred / baseline-subtracted
        # (deviation=True) the zero pose is 0 in every column, rotation.w included
        held[~np.maximum.accumulate(ok)] = 1.0 if (c.endswith("rotation.w") and not deviation) else 0.0
        X[:, j] = held
    return X


# ---------------------------------------------------------------------------
# Splitting by whole recording
# ---------------------------------------------------------------------------
def split_recordings(root, files, val_frac, test_frac, seed, label_cols):
    """Split these recordings (all under root) by whole recording, balanced by label combination."""
    rel = {f: os.path.relpath(f, root).replace(os.sep, "/") for f in files}

    def stratum(f):
        lab = file_labels(f, label_cols)
        parts = [f"{c} {main_key(lab[c])}" for c in label_cols if c in lab and main_key(lab[c])]
        return " | ".join(parts) or "unlabelled"

    strata = {f: stratum(f) for f in progress(files, desc="reading labels")}
    fracs = {"train": 1 - val_frac - test_frac, "val": val_frac, "test": test_frac}
    if min(fracs.values()) < 0:
        raise SystemExit("--val_frac and --test_frac leave nothing for training")
    rng = random.Random(seed)
    by_st = defaultdict(list)
    for f in files:
        by_st[strata[f]].append(f)
    order = sorted(by_st)
    rng.shuffle(order)
    glob_cnt = {s: 0 for s in SPLITS}
    assign = {}
    for st in order:
        # whole shares first; leftover recordings go to the split furthest below its share overall,
        # so small label groups alternate between val and test instead of always feeding the same one
        ks = sorted(by_st[st])
        rng.shuffle(ks)
        n = len(ks)
        base = {s: int(np.floor(fracs[s] * n + 1e-9)) for s in SPLITS}
        for _ in range(n - sum(base.values())):
            s = max(SPLITS, key=lambda s: (round(fracs[s] * n - base[s], 9),
                                           fracs[s] * len(files) - glob_cnt[s] - base[s], s == "test"))
            base[s] += 1
        i = 0
        for s in ("test", "val", "train"):
            for f in ks[i:i + base[s]]:
                assign[f] = s
            i += base[s]
            glob_cnt[s] += base[s]
    sets = {s: sorted(f for f in files if assign[f] == s) for s in SPLITS}
    info = {"data_dir": os.path.abspath(root), "seed": seed, "val_frac": val_frac, "test_frac": test_frac,
            "splits": {s: [rel[f] for f in sets[s]] for s in SPLITS},
            "strata": {rel[f]: strata[f] for f in files}}
    return sets, info


def load_split_file(path):
    """A saved <out>_split.json -> {split: [paths]}, info."""
    with open(os.path.expanduser(path)) as fh:
        info = json.load(fh)
    root = info["data_dir"]
    sets = {s: [os.path.join(root, r) for r in info["splits"][s]] for s in SPLITS}
    missing = [f for s in SPLITS for f in sets[s] if not os.path.isfile(f)]
    if missing:
        raise SystemExit(f"{len(missing)} file(s) in {path} not found, e.g. {missing[0]}")
    return sets, info


def print_split(sets, info):
    """Recordings per split and label combination; flags combinations missing from val or test."""
    strata, root = info.get("strata", {}), info["data_dir"]
    rows = defaultdict(lambda: {s: 0 for s in SPLITS})
    for s in SPLITS:
        for f in sets[s]:
            rows[strata.get(os.path.relpath(f, root).replace(os.sep, "/"), "?")][s] += 1
    w = max([len(k) for k in rows] + [18])
    print(f"Split of {sum(len(v) for v in sets.values())} recordings, seed {info['seed']}:")
    print(f"  {'label':{w}s}" + "".join(f"{s:>7s}" for s in SPLITS))
    for st in sorted(rows):
        r = rows[st]
        flag = "   <- none in " + " / ".join(s for s in ("val", "test") if not r[s]) \
            if st != "unlabelled" and (not r["val"] or not r["test"]) else ""
        print(f"  {st:{w}s}" + "".join(f"{r[s]:7d}" for s in SPLITS) + flag)
    print(f"  {'total':{w}s}" + "".join(f"{len(sets[s]):7d}" for s in SPLITS))


# ---------------------------------------------------------------------------
# Data loading and cutting into segments
# ---------------------------------------------------------------------------
def configure_heads(targets, train_files):
    """One classification head per target (= label column), its classes = the label values in the
    training files. A target with fewer than two classes is left out with a warning."""
    found = defaultdict(set)
    for f in progress(train_files, desc="reading classes"):
        for c, keys in file_labels(f, targets).items():
            found[c] |= set(keys) - {""}
    heads = []
    for t in targets:
        cl = sort_classes(found[t])
        if len(cl) < 2:
            print(f"WARNING: '{t}' has only {len(cl)} value(s) in the training recordings "
                  f"({', '.join(cl) or 'none - column missing or blank'}): nothing to tell apart, "
                  f"leaving out the {t} head.")
            continue
        heads.append({"name": t, "col": t, "classes": cl, "names": cl})
    if not heads:
        raise SystemExit("No target left to train: every label has fewer than two values in the training recordings.")
    return heads


def load_csv(path, in_cols, heads, center=False):
    """Read one recording. Returns X (T, n_in) float32 with NaN where a value is missing, C (T, n_heads)
    int64 class per row (-1 = blank or a class not in training), t (T,) seconds, and notes.
    A tag missing from the file entirely counts as never seen; any other missing input is an error."""
    name = os.path.basename(path)
    df = pd.read_csv(path)
    absent = [c for c in in_cols if c not in df.columns]
    missing = [c for c in absent if not c.startswith(TAG_PREFIX)]
    if missing:
        raise ValueError(f"{name}: missing columns, e.g. {missing[:3]}")
    if absent:
        df = pd.concat([df, pd.DataFrame({c: (0.0 if is_seen_col(c) else np.nan) for c in absent},
                                         index=df.index)], axis=1)
    X = df[in_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
    if center:                                  # remove each tag pose column's mean over this recording
        pose = [j for j, c in enumerate(in_cols) if c.startswith(TAG_PREFIX) and not is_seen_col(c)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            X[:, pose] -= np.nan_to_num(np.nanmean(X[:, pose], axis=0))
    C = np.full((len(df), len(heads)), -1, np.int64)
    labels = read_labels(path, [h["col"] for h in heads])
    notes = []
    for h_i, h in enumerate(heads):
        if h["col"] not in labels.columns:
            continue
        keys = label_keys(labels[h["col"]])
        lookup = {k: i for i, k in enumerate(h["classes"])}
        C[:, h_i] = [lookup.get(k, -1) for k in keys]
        unseen = sorted(set(keys) - set(lookup) - {""})
        if unseen:
            notes.append(f"{name}: {h['name']} value(s) {unseen} not in the training recordings - "
                         f"those segments are left out of the {h['name']} score")
    t = pd.to_numeric(df["t_s"], errors="coerce").to_numpy(float) if "t_s" in df.columns \
        else np.arange(len(df)) / FPS
    return np.array(X, copy=True), C, t, notes


def load_files(files, in_cols, heads, seg_len, label="files", center=False):
    """Load recordings: a list of dicts {name, X, C, t} (only those with at least seg_len rows)."""
    recs, short = [], []
    for f in progress(files, desc=f"loading {label}"):
        try:
            X, C, t, notes = load_csv(f, in_cols, heads, center)
        except ValueError as e:
            print(f"  skipping {e}")
            continue
        for n in notes:
            print(f"  {label}: {n}")
        if len(X) < seg_len:
            short.append(f"{os.path.basename(f)} ({len(X)} rows)")
            continue
        recs.append({"name": os.path.basename(f), "path": f, "X": X, "C": C, "t": t})
    if short:
        print(f"  {label}: skipped {len(short)} recording(s) shorter than one {seg_len}-frame segment: "
              f"{', '.join(short[:5])}{' ...' if len(short) > 5 else ''}")
    if files and not recs:
        raise SystemExit(f"No usable recordings in the {label} set")
    return recs


def segment_bounds(T, seg_len):
    """(start, stop) rows of the consecutive, non-overlapping segments in T rows. The tail that
    doesn't fill a whole segment is dropped."""
    return [(s, s + seg_len) for s in range(0, T - seg_len + 1, seg_len)]


def normalize(X, stats):
    """Raw recording inputs (T, n_in) -> gaps filled, scaled with the training mean / std."""
    dev = stats.get("deviation", stats.get("center", False))
    X = fill_tag_gaps(np.array(X, np.float32, copy=True), stats["in_cols"], dev)
    return np.nan_to_num((X - stats["x_mean"]) / stats["x_std"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def segment_labels(C, bounds):
    """Per segment and head: the most common labelled class of its rows (-1 if none are labelled)."""
    out = np.full((len(bounds), C.shape[1]), -1, np.int64)
    for i, (s, e) in enumerate(bounds):
        for h in range(C.shape[1]):
            c = C[s:e, h]
            c = c[c >= 0]
            if len(c):
                out[i, h] = np.bincount(c).argmax()
    return out


def make_segments(recs, stats):
    """Cut every recording into segments. Returns X (N, seg_len, n_in) and C (N, n_heads) tensors,
    rec (N,) the recording each segment came from, and the number of leftover rows dropped."""
    L = stats["seg_len"]
    xs, cs, rec, dropped = [], [], [], 0
    for i, r in enumerate(recs):
        b = segment_bounds(len(r["X"]), L)
        Xn = normalize(r["X"], stats)
        xs.append(Xn[: len(b) * L].reshape(len(b), L, -1))
        cs.append(segment_labels(r["C"], b))
        rec += [i] * len(b)
        dropped += len(r["X"]) - len(b) * L
    return (torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(cs)),
            np.array(rec), dropped)


def report_data(label, recs, in_cols, heads):
    """How often tags were missed, empty IMU cells and the label balance. Call before gaps are filled."""
    X = np.concatenate([r["X"] for r in recs])
    seen = [j for j, c in enumerate(in_cols) if is_seen_col(c)]
    if seen:
        missed = 1.0 - np.nanmean(X[:, seen], axis=0)
        never = np.array([(np.nan_to_num(r["X"][:, seen]) == 0).all(0) for r in recs]).sum(0)
        print(f"  {label}: tags missed in {missed.mean():.2%} of rows overall "
              f"(held at the last detection or zero pose; seen flag = 0).")
        for k in [k for k in np.argsort(missed)[::-1][:5] if missed[k] > 0]:
            print(f"      {tag_of(in_cols[seen[k]])}: {missed[k]:.1%} of rows, never seen in {never[k]} recording(s)")
    other = [j for j, c in enumerate(in_cols) if not c.startswith(TAG_PREFIX)]
    if other and np.isnan(X[:, other]).any():
        print(f"  {label}: {np.isnan(X[:, other]).mean():.2%} of IMU / extra input cells empty; "
              "set to the training mean.")
    for h_i, h in enumerate(heads):
        per = defaultdict(int)
        for r in recs:
            c = r["C"][:, h_i]
            c = c[c >= 0]
            per[h["names"][np.bincount(c).argmax()] if len(c) else "(blank)"] += 1
        print(f"  {label}: {h['name']} by recording: " + ", ".join(f"{k} x{v}" for k, v in sorted(per.items())))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SegmentClassifier(nn.Module):
    """LSTM (or GRU / RNN) read over one segment from a zero state; the hidden state at the last
    frame goes through a linear layer that gives one block of logits per classification head."""

    def __init__(self, n_in, n_out, hidden, layers, kind="lstm", dropout=0.0, in_dropout=0.0):
        super().__init__()
        self.in_drop = nn.Dropout(in_dropout)
        rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[kind]
        self.rnn = rnn_cls(n_in, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x):                        # x: (B, seg_len, n_in)
        out, _ = self.rnn(self.in_drop(x))       # no state passed in: zeros for every segment
        return self.head(self.drop(out[:, -1]))  # (B, n_out): one prediction per segment


def build_model(n_in, n_out, a):
    return SegmentClassifier(n_in, n_out, a["hidden"], a["layers"], a.get("model", "lstm"),
                             a.get("dropout", 0.0), a.get("in_dropout", 0.0))


def head_slices(heads):
    """(start, stop) of each head's logits in the model output."""
    out, k = [], 0
    for h in heads:
        out.append((k, k + len(h["classes"])))
        k += len(h["classes"])
    return out


def balanced_acc(cm):
    """Mean of each class's recall (classes with no true segments are left out)."""
    rows = cm.sum(1)
    return float(np.mean(np.diag(cm)[rows > 0] / rows[rows > 0])) if rows.sum() else float("nan")


# ---------------------------------------------------------------------------
# Train / score
# ---------------------------------------------------------------------------
def make_tag_dropout(in_cols, x_mean, x_std, p, device):
    """Tensors for apply_tag_dropout: which tag each input column belongs to (-1 = none), which columns are
    seen flags, and what a seen flag of 0 looks like after normalising."""
    tags = sorted({tag_of(c) for c in in_cols if tag_of(c)})
    ids = {t: i for i, t in enumerate(tags)}
    col_tag = torch.tensor([ids.get(tag_of(c), -1) for c in in_cols])
    is_seen = torch.tensor([is_seen_col(c) for c in in_cols])
    fill = torch.from_numpy((0.0 - x_mean) / x_std).float()
    return {"col_tag": col_tag.to(device), "is_seen": is_seen.to(device), "fill": fill.to(device),
            "n_tags": len(tags), "p": p}


def apply_tag_dropout(x, td):
    """x: (B, T, n_in) normalised segments. Per segment and tag, with probability p: the tag's pose columns
    hold the segment's first-frame value and its seen flag is set to 0 (see --tag_dropout)."""
    m = torch.rand(x.shape[0], td["n_tags"], device=x.device) < td["p"]
    colm = m[:, td["col_tag"].clamp(min=0)] & (td["col_tag"] >= 0)               # (B, n_in)
    pose_m, seen_m = colm & ~td["is_seen"], colm & td["is_seen"]
    x = torch.where(pose_m[:, None, :], x[:, :1, :].expand_as(x), x)
    return torch.where(seen_m[:, None, :], td["fill"].expand_as(x), x)


def run_epoch(model, loader, device, heads, optimizer=None, clip=1.0, class_w=None, select="bacc", desc=None,
              tag_drop=None):
    """One pass over a loader of (x, c) segments. Trains if an optimizer is given, otherwise scores.

    Each head uses cross-entropy (optionally class-weighted) over its labelled segments; the heads'
    losses are added. Returns {"heads": [per head: ce, acc, bacc, conf], "score"}: score is what
    picks the best epoch (lower is better): per head, 1 - balanced accuracy (select="bacc") or the
    cross-entropy (select="ce"), added up."""
    training = optimizer is not None
    model.train(training)
    class_w = class_w or [None] * len(heads)
    sl = head_slices(heads)
    hs = [{"ce": 0.0, "n": 0, "right": 0, "conf": np.zeros((len(h["classes"]),) * 2, np.int64)} for h in heads]
    with torch.set_grad_enabled(training):
        for x, c in (progress(loader, desc=desc) if desc else loader):
            x, c = x.to(device), c.to(device)
            if training and tag_drop:
                x = apply_tag_dropout(x, tag_drop)
            pred = model(x)
            loss = torch.zeros((), device=device)
            for h_i, (a, b) in enumerate(sl):
                lab = c[:, h_i] >= 0
                if not lab.any():
                    continue
                logits, true = pred[lab, a:b], c[lab, h_i]
                loss = loss + nn.functional.cross_entropy(logits, true, weight=class_w[h_i])
                with torch.no_grad():
                    st = hs[h_i]
                    st["ce"] += nn.functional.cross_entropy(logits, true, reduction="sum").item()
                    guess = logits.argmax(1)
                    st["right"] += (guess == true).sum().item()
                    st["n"] += int(lab.sum())
                    np.add.at(st["conf"], (true.cpu().numpy(), guess.cpu().numpy()), 1)
            if not torch.isfinite(loss):
                raise SystemExit("Loss became NaN/inf. Try a lower --lr (e.g. 0.0003).")
            if training and loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
    out = {"heads": [{"ce": st["ce"] / st["n"] if st["n"] else float("nan"),
                      "acc": st["right"] / st["n"] if st["n"] else float("nan"),
                      "bacc": balanced_acc(st["conf"]), "conf": st["conf"]} for st in hs]}
    part = [(1 - h["bacc"]) if select == "bacc" else h["ce"] for h in out["heads"]]
    out["score"] = sum(v for v in part if v == v)
    return out


def fmt_scores(r, heads):
    return ", ".join(f"{h['name']} CE {hr['ce']:.4f} acc {hr['acc']:.1%} bal {hr['bacc']:.1%}"
                     for h, hr in zip(heads, r["heads"]))


@torch.no_grad()
def predict_segments(model, X, device, batch=1024):
    """Logits (N, n_out) for a tensor of segments (N, seg_len, n_in)."""
    model.eval()
    return np.concatenate([model(X[i:i + batch].to(device)).cpu().numpy() for i in range(0, len(X), batch)]) \
        if len(X) else np.zeros((0, model.head.out_features), np.float32)


def softmax(z):
    p = np.exp(z - z.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def print_confusion(cm, names, title):
    w = max([len(n) for n in names] + [8]) + 2
    print(f"\n{title} (rows = true, columns = predicted, counted per segment):")
    print("  " + " " * w + "".join(f"{a:>{w}s}" for a in names) + "    recall")
    for k, a in enumerate(names):
        rec = f"{cm[k, k] / cm[k].sum():6.1%}" if cm[k].sum() else "     -"
        print(f"  {a:{w}s}" + "".join(f"{v:{w}d}" for v in cm[k]) + f"    {rec}")


def recording_report(logits, C, rec, names, heads, label, only_below=None):
    """Per recording and head: the true class, the share of its segments predicted right, what the
    wrong ones were mostly called, and the class the whole recording would get if its segments
    voted (mean probability). Worst first, so errors can be traced to files."""
    rows = []
    for i, name in enumerate(names):
        m = rec == i
        parts, worst = [], 1.0
        for h_i, (h, (a, b)) in enumerate(zip(heads, head_slices(heads))):
            c = C[m, h_i]
            lab = c >= 0
            if not lab.any():
                parts.append(f"{h['name']}: (no label)")
                continue
            p = softmax(logits[m, a:b])
            guess = p.argmax(1)
            k = int(np.bincount(c[lab]).argmax())
            acc = float(np.mean(guess[lab] == c[lab]))
            worst = min(worst, acc)
            txt = f"{h['name']}: {h['names'][k]} {acc:6.1%} of {int(lab.sum())} segments"
            wrong = np.bincount(guess[lab][guess[lab] != c[lab]], minlength=len(h["names"]))
            if wrong.sum():
                txt += f" (else mostly {h['names'][int(wrong.argmax())]})"
            txt += f", vote -> {h['names'][int(p.mean(0).argmax())]}"
            parts.append(txt)
        rows.append((worst, name, parts))
    shown = [x for x in rows if only_below is None or x[0] < only_below]
    head = f"\nPer recording ({label}" + (f", only those below {only_below:.0%}" if only_below else "") + "; worst first):"
    print(head if shown else head + "  none")
    for _, name, parts in sorted(shown, key=lambda x: x[0]):
        print(f"  {name}\n      " + "   |   ".join(parts))


def save_curve(history, path, ylabel):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping the curve plot")
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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", help="folder of ALL recordings (subfolders included): split by whole "
                                      "recording into train / val / test, balanced by label")
    p.add_argument("--split_file", help="reuse a split saved by an earlier run (<out>_split.json)")
    p.add_argument("--val_frac", type=float, default=0.15, help="share of recordings for val (0.15)")
    p.add_argument("--test_frac", type=float, default=0.15, help="share of recordings for test (0.15)")
    p.add_argument("--seg_len", type=int, default=16,
                   help="frames per segment: each split's recordings are cut into consecutive, "
                        "non-overlapping segments of this many frames (default 16 = ~0.25 s at 65 fps)")
    p.add_argument("--test", choices=list(TESTS),
                   help="wall = wall_side + dist_to_wall; angle = angle_to_head + dist_to_head "
                        "(default: the one test the folder holds)")
    p.add_argument("--target", nargs="+", choices=TARGETS, metavar="TARGET",
                   help="heads to train (default: both heads of the test): wall_side dist_to_wall, or "
                        "angle_to_head dist_to_head")
    p.add_argument("--inputs", nargs="+", choices=["tags", "imu", "seen"], default=["tags", "imu"],
                   help="what the model is fed (default: tags imu); seen = only the tags' seen flags (a check)")
    p.add_argument("--day_baseline", action="store_true",
                   help="subtract each day's no-object baseline (mean tag pose of that day's training recordings "
                        "labelled --baseline_label) from every recording of that day")
    p.add_argument("--baseline_label", default="none", help="label of the no-object recordings (default none)")
    p.add_argument("--day", default=DAY_RE, help="regex for the day in the file name (default MM_DD_YYYY or MM_DD_YY, "
                                                 "else the first NN_NN_NN)")
    p.add_argument("--center_recordings", action="store_true",
                   help="subtract each recording's own mean from every tag pose column (a check: removes "
                        "constant offsets, leaves the movement)")
    p.add_argument("--extra_inputs", nargs="+", default=[], metavar="NAME", help="more columns to feed in")
    p.add_argument("--drop_tags", default="", metavar="N,N",
                   help="tag numbers to leave out of the inputs entirely, e.g. 0 for a tag that keeps dropping out")
    p.add_argument("--balance_classes", action="store_true",
                   help="weight each class by 1/its share of training segments, so a rare class counts "
                        "as much as a common one")
    p.add_argument("--model", choices=["lstm", "gru", "rnn"], default="lstm")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0, help="dropout on the hidden state (e.g. 0.2)")
    p.add_argument("--in_dropout", type=float, default=0.0, help="dropout on the input values (e.g. 0.1)")
    p.add_argument("--tag_dropout", type=float, default=0.0, metavar="P",
                   help="training only: make each tag look missed (held pose, seen = 0) for a whole segment "
                        "with probability P (e.g. 0.3), so no tag, and no dropout pattern, can be relied on")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--select", choices=["bacc", "ce"], default="bacc",
                   help="what picks the best epoch: bacc (default) = balanced accuracy, ce = cross-entropy")
    p.add_argument("--patience", type=int, help="stop when val hasn't improved for this many epochs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="checkpoint.pt", help="where to save the best model")
    args = p.parse_args()
    if bool(args.data_dir) == bool(args.split_file):
        p.error("give exactly one of --data_dir or --split_file")
    if args.day_baseline and args.center_recordings:
        p.error("--day_baseline and --center_recordings can't be combined (centring removes the baseline's job)")
    if args.seg_len < 1:
        p.error("--seg_len must be at least 1")
    args.inputs = [i for i in ("tags", "imu", "seen") if i in args.inputs]
    drop_tags = {int(x) for x in args.drop_tags.split(",") if x.strip()}

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. which test, then the split by whole recording (only that test's recordings)
    if args.split_file:
        sets, split_info = load_split_file(args.split_file)
        args.test = args.test or split_info.get("test") or files_of_test(sets["train"], None)[0]
    else:
        root = os.path.expanduser(args.data_dir)
        if not os.path.isdir(root):
            raise SystemExit(f"Folder not found: {args.data_dir}")
        every = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
        if not every:
            raise SystemExit(f"No CSV files found in {args.data_dir}")
        args.test, files = files_of_test(every, args.test)
    wrong = [t for t in (args.target or []) if t not in TESTS[args.test]]
    if wrong:
        p.error(f"--target {' '.join(wrong)} doesn't belong to the {args.test} test "
                f"(its heads: {', '.join(TESTS[args.test])})")
    args.target = [t for t in TESTS[args.test] if not args.target or t in args.target]
    print(f"Test: {args.test}. Heads: {', '.join(args.target)}")
    if not args.split_file:
        sets, split_info = split_recordings(root, files, args.val_frac, args.test_frac, args.seed, args.target)
        split_info["test"] = args.test
    print_split(sets, split_info)
    if not sets["train"]:
        raise SystemExit("The split left no training recordings.")
    base = os.path.splitext(args.out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(f"{base}_split.json", "w") as fh:
        json.dump(split_info, fh, indent=1)
    print(f"Split saved to {base}_split.json")

    # 2. heads, inputs, recordings
    heads = configure_heads(args.target, sets["train"])
    in_cols = input_cols(sets["train"], args.inputs, args.extra_inputs, drop_tags)
    n_tag = sum(c.startswith(TAG_PREFIX) for c in in_cols)
    print(f"Inputs: {len(in_cols)} columns ({n_tag} tag, {len(in_cols) - n_tag} IMU / extra)"
          + (f", tags left out: {sorted(drop_tags)}" if drop_tags else ""))
    for h in heads:
        print(f"  {h['name']} classes: {', '.join(h['names'])}")
    recs = {k: load_files(f, in_cols, heads, args.seg_len, label=k, center=args.center_recordings) for k, f in sets.items()}
    print("Checking the data:")
    for label, rr in recs.items():
        if rr:
            report_data(label, rr, in_cols, heads)

    day_base = None
    if args.day_baseline:
        main_col = TESTS[args.test][0]
        vectors, n_rec = make_day_baselines(recs["train"], in_cols, main_col, args.baseline_label, args.day)
        days = defaultdict(lambda: {s: 0 for s in SPLITS})
        for k in SPLITS:
            for r in recs[k]:
                days[day_of(r["path"], args.day)][k] += 1
        print(f"Day baselines (mean tag pose of the training '{args.baseline_label}' recordings of each day):")
        print(f"  {'day':14s}" + "".join(f"{k:>7s}" for k in SPLITS) + "   baseline from")
        for d in sorted(days):
            print(f"  {d:14s}" + "".join(f"{days[d][k]:7d}" for k in SPLITS)
                  + (f"   {n_rec[d]} training recording(s)" if d in vectors else "   NONE"))
        lacking = sorted(d for d in days if d not in vectors)
        if lacking:
            raise SystemExit(
                f"--day_baseline: no training recording labelled '{args.baseline_label}' ({main_col}) on day(s) "
                f"{', '.join(lacking)}. A day can only be normalised against its own no-object recordings: "
                "record some '" + args.baseline_label + "' runs on that day, or move some into the training set "
                "(--split_file), or leave that day out. Normalising a day that has no baseline would remove the "
                "very difference between the classes.")
        for k in SPLITS:
            for r in recs[k]:
                r["X"] = r["X"] - vectors[day_of(r["path"], args.day)]
        day_base = {"label": args.baseline_label, "day_re": args.day, "vectors": vectors}

    # 3. normalize with training rows only (real readings: missed tags are still NaN here)
    Xall = np.concatenate([r["X"] for r in recs["train"]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        x_mean, x_std = np.nanmean(Xall, 0), np.nanstd(Xall, 0)
    x_mean = np.nan_to_num(x_mean).astype(np.float32)
    x_std = np.where(~np.isfinite(x_std) | (x_std < 1e-6), 1.0, x_std).astype(np.float32)
    stats = dict(x_mean=x_mean, x_std=x_std, in_cols=in_cols, seg_len=args.seg_len, fps=FPS,
                 test=args.test, targets=[h["name"] for h in heads], heads=heads,
                 inputs=args.inputs + (["centered"] if args.center_recordings else []) + (["day-baseline"] if day_base else []),
                 center=args.center_recordings, day_baseline=day_base,
                 deviation=bool(args.center_recordings or day_base))

    # 4. cut each split's recordings into non-overlapping segments
    data = {k: make_segments(rr, stats) if rr else None for k, rr in recs.items()}
    print(f"Segments of {args.seg_len} frames ({args.seg_len / FPS:.2f} s), no overlap:")
    for k in SPLITS:
        if data[k]:
            X, C, _, dropped = data[k]
            print(f"  {k:5s} {len(recs[k]):3d} recordings -> {len(X):6d} segments ({dropped} leftover rows dropped)")

    class_w = []
    Ctr = data["train"][1].numpy()
    for h_i, h in enumerate(heads):
        c = Ctr[:, h_i]
        counts = np.bincount(c[c >= 0], minlength=len(h["classes"])).astype(float)
        h["train_counts"] = counts.tolist()
        print(f"  {h['name']} training segments: " + ", ".join(f"{n} {int(k)}" for n, k in zip(h["names"], counts))
              + f", unlabelled {int((c < 0).sum())}")
        if args.balance_classes:
            w = np.where(counts > 0, counts.sum() / (len(counts) * np.maximum(counts, 1)), 0.0)
            class_w.append(torch.tensor(w, dtype=torch.float32, device=device))
            print(f"  {h['name']} class weights: " + ", ".join(f"{n} {x:.2f}" for n, x in zip(h["names"], w)))
        else:
            class_w.append(None)

    def loader(k, train=False):
        return DataLoader(TensorDataset(data[k][0], data[k][1]), batch_size=args.batch_size, shuffle=train) \
            if data[k] else None

    train_dl, val_dl = loader("train", True), loader("val")
    n_out = sum(len(h["classes"]) for h in heads)
    model = build_model(len(in_cols), n_out, vars(args)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tag_drop = make_tag_dropout(in_cols, stats["x_mean"], stats["x_std"], args.tag_dropout, device) \
        if args.tag_dropout > 0 else None
    if tag_drop:
        print(f"Tag dropout: each of the {tag_drop['n_tags']} tags is made to look missed in {args.tag_dropout:.0%} "
              "of the training segments")
    print(f"{len(in_cols)} inputs -> {n_out} class outputs, model={args.model}, hidden={args.hidden}, "
          f"layers={args.layers}, device={device}")

    # 5. train; keep the epoch with the lowest val score (train score if there is no val set)
    history = {"train": [], "val": []}
    best, best_epoch = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_dl, device, heads, optimizer=optimizer, class_w=class_w, select=args.select,
                       tag_drop=tag_drop,
                       desc=f"epoch {epoch}/{args.epochs}")
        history["train"].append(tr["score"])
        msg, score = f"epoch {epoch:3d}/{args.epochs}  train {fmt_scores(tr, heads)}", tr["score"]
        if val_dl:
            va = run_epoch(model, val_dl, device, heads, select=args.select)
            history["val"].append(va["score"])
            msg += f"  val {fmt_scores(va, heads)}"
            score = va["score"]
        if score < best:
            best, best_epoch = score, epoch
            torch.save({"model": model.state_dict(), "stats": stats, "args": vars(args)}, args.out)
            msg += "  *saved"
        print(msg)
        if args.patience and val_dl and epoch - best_epoch >= args.patience:
            print(f"Stopping early: val hasn't improved for {args.patience} epochs (best: epoch {best_epoch})")
            break

    # 6. final scores of the saved model
    model, _ = load_model(args.out, device)
    print(f"\nFinal scores of the saved model (epoch {best_epoch}):")
    logits = {}
    for k in SPLITS:
        if not data[k]:
            continue
        logits[k] = predict_segments(model, data[k][0], device)
        r = run_epoch(model, loader(k), device, heads, select=args.select)
        print(f"  {k:5s} {fmt_scores(r, heads)}")
        for h, hr in zip(heads, r["heads"]):
            maj = int(np.argmax(h["train_counts"]))
            n = hr["conf"].sum()
            if n:
                print(f"        {h['name']} baseline (always '{h['names'][maj]}'): {hr['conf'][maj].sum() / n:.1%}")
            if k != "train":
                print_confusion(hr["conf"], h["names"], f"{h['name']} confusion matrix ({k})")
    for k in logits:
        recording_report(logits[k], data[k][1].numpy(), data[k][2], [r["name"] for r in recs[k]], heads, k,
                         only_below=0.9 if k == "train" else None)
    save_curve(history, f"{base}_curve.png",
               " + ".join(f"{h['name']} " + ("(1 - bal. acc)" if args.select == "bacc" else "CE") for h in heads))


# ---------------------------------------------------------------------------
# Inference helpers (used by evaluate.py and plot_predictions.py)
# ---------------------------------------------------------------------------
def load_model(path, device="cpu"):
    """Rebuild a saved model; returns it (in eval mode) with its stats."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    s = ckpt["stats"]
    model = build_model(len(s["in_cols"]), sum(len(h["classes"]) for h in s["heads"]), ckpt["args"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), s


@torch.no_grad()
def predict_file(model, stats, csv_path, device="cpu"):
    """Predict one recording the way it was trained: cut into consecutive, non-overlapping segments
    of seg_len frames, each fed to the model on its own.

    Returns a DataFrame with one row per segment:
      seg, row_start, row_end (exclusive), t_start, t_end (seconds)
      <head>_pred, true:<head>      predicted and true class name (true is '' with no label)
      p_<head>_<class>              the predicted probability of each class
    Raises ValueError if the recording is shorter than one segment."""
    heads, L = stats["heads"], int(stats["seg_len"])
    X, C, t, _ = load_csv(csv_path, stats["in_cols"], heads, stats.get("center", False))
    base = baseline_vector(stats, csv_path)
    if base is not None:
        X = X - base
    b = segment_bounds(len(X), L)
    if not b:
        raise ValueError(f"{os.path.basename(csv_path)}: {len(X)} rows, shorter than one {L}-frame segment")
    Xs = torch.from_numpy(normalize(X, stats)[: len(b) * L].reshape(len(b), L, -1))
    logits = predict_segments(model, Xs, device)
    Cs = segment_labels(C, b)
    dt = float(np.nanmedian(np.diff(t))) if len(t) > 1 else 1.0 / FPS
    df = pd.DataFrame({"seg": np.arange(len(b)), "row_start": [s for s, _ in b], "row_end": [e for _, e in b],
                       "t_start": [t[s] for s, _ in b], "t_end": [t[e - 1] + dt for _, e in b]})
    for h_i, (h, (a, z)) in enumerate(zip(heads, head_slices(heads))):
        p = softmax(logits[:, a:z])
        df[f"{h['name']}_pred"] = [h["names"][k] for k in p.argmax(1)]
        df[f"true:{h['name']}"] = [h["names"][k] if k >= 0 else "" for k in Cs[:, h_i]]
        for k, n in enumerate(h["names"]):
            df[prob_col(h["name"], n)] = p[:, k]
    return df


if __name__ == "__main__":
    main()
