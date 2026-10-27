#!/usr/bin/env python3
"""
sync_export.py - resample every topic onto the camera frame times and write one CSV.

Each CSV row is one camera frame:
    timestamp   absolute time of the frame (header.stamp), seconds with full ns precision
    t_s         seconds since the first camera frame (same axis as the viewer)
    frame_idx   index of the frame in the bag (0-based), to pair rows with images
    ...         one column per field, named <topic>.<field>

How each field is brought to the frame time ("mode"):
    exact     topic is stamped with the camera's own stamps (apriltag /detections, tag /tf):
              the value from that same frame, empty if the tag was not seen
    linear    linear interpolation between the samples either side (floats)
    slerp     quaternions - spherical interpolation, sign-continuous
    euler     roll/pitch/yaw recomputed from the interpolated quaternion
    angle_deg heading in degrees - interpolated across the 359->0 wrap
    previous  last value at or before the frame (ints, bools, strings, modes)
Values are left empty (NaN) instead of being invented when the frame falls
outside the topic's data, or inside a gap longer than GAP_FACTOR x the topic's
normal sample period.

Usage
    python sync_export.py BAG --info
    python sync_export.py BAG --start 3.5 --stop 21.0 --out run1.csv
    python sync_export.py BAG --topics imu/data rel_alt detections --out run1.csv
    python sync_export.py BAG --start 3.5 --stop 21 --export-images frames/
BAG is the bag folder (with metadata.yaml) or the .mcap inside it.
--start/--stop are seconds since the first camera frame (as shown in the viewer).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bagcache import BagData, Series, bag_summary, load_bag, quat_groups, quat_to_euler_deg

GAP_FACTOR = 3.0

# Per-column overrides: (regex matched against '<topic short>.<column>', mode).
# First match wins.  Modes: exact, linear, slerp, angle_deg, previous, skip.
RULES: list[tuple[str, str]] = [
    (r"compass_hdg\.data$", "angle_deg"),
    (r"\.heading$|\.hdg$|heading\.heading_deg$", "angle_deg"),
]


@dataclass
class SyncResult:
    t: np.ndarray                    # int64 ns target times
    t0: int                          # ns of first camera frame (t_s origin)
    frame_idx: np.ndarray | None
    names: list[str] = field(default_factory=list)
    kinds: list[str] = field(default_factory=list)
    modes: list[str] = field(default_factory=list)
    values: list[np.ndarray] = field(default_factory=list)
    notes: dict = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return len(self.t)


# --------------------------------------------------------------------------- #
# Interpolation primitives (all vectorised)
# --------------------------------------------------------------------------- #
def _max_gap(ts: np.ndarray, gap_factor: float) -> float:
    if len(ts) < 2:
        return np.inf
    return max(gap_factor * float(np.median(np.diff(ts))), 1e6)


def _bracket(ts, tq, max_gap):
    """Left index, fraction and validity for linear interpolation."""
    n = len(ts)
    if n == 0:
        return np.zeros(len(tq), int), np.zeros(len(tq)), np.zeros(len(tq), bool)
    if n == 1:
        hit = tq == ts[0]
        return np.zeros(len(tq), int), np.zeros(len(tq)), hit
    k = np.searchsorted(ts, tq, side="right") - 1
    k0 = np.clip(k, 0, n - 2)
    t0, t1 = ts[k0], ts[k0 + 1]
    a = (tq - t0).astype(float) / (t1 - t0).astype(float)
    ok = (k >= 0) & (k <= n - 2) & ((t1 - t0) <= max_gap)
    ok |= (k >= 0) & (tq == ts[np.clip(k, 0, n - 1)])     # exact hit on a sample
    last = (k == n - 1) & (tq == ts[-1])
    a = np.where(last, 1.0, a)
    return k0, np.clip(a, 0.0, 1.0), ok


def interp_linear(ts, v, tq, max_gap):
    k0, a, ok = _bracket(ts, tq, max_gap)
    if len(ts) == 1:
        out = np.full(len(tq), np.nan)
        out[ok] = v[0]
        return out
    out = v[k0] + a * (v[k0 + 1] - v[k0])
    out[~ok] = np.nan
    return out


def interp_angle_deg(ts, v, tq, max_gap):
    good = ~np.isnan(v)
    if good.sum() == 0:
        return np.full(len(tq), np.nan)
    ts, v = ts[good], v[good]
    signed = np.nanmin(v) < 0
    un = np.degrees(np.unwrap(np.radians(v)))
    out = interp_linear(ts, un, tq, max_gap)
    return ((out + 180) % 360 - 180) if signed else (out % 360)


def interp_previous(ts, v, tq, max_hold):
    k = np.searchsorted(ts, tq, side="right") - 1
    ok = (k >= 0) & ((tq - ts[np.clip(k, 0, None)]) <= max_hold)
    kk = np.clip(k, 0, len(ts) - 1)
    if v.dtype.kind in "US":
        out = np.where(ok, v[kk], "").astype(object)
    else:
        out = np.where(ok, v[kk], np.nan)
    return out


def match_exact(ts, tq, tol):
    """Index into ts of the sample within tol of each target, or -1."""
    if len(ts) == 0:
        return np.full(len(tq), -1)
    i = np.clip(np.searchsorted(ts, tq), 0, len(ts) - 1)
    j = np.clip(i - 1, 0, len(ts) - 1)
    best = np.where(np.abs(ts[i] - tq) <= np.abs(ts[j] - tq), i, j)
    return np.where(np.abs(ts[best] - tq) <= tol, best, -1)


def take_exact(v, idx):
    if v.dtype.kind in "US":
        return np.where(idx >= 0, v[np.clip(idx, 0, None)], "").astype(object)
    return np.where(idx >= 0, v[np.clip(idx, 0, None)], np.nan)


def slerp_series(ts, q, tq, max_gap):
    """q: (n,4) xyzw. Returns (m,4) interpolated, NaN where invalid."""
    q = q.astype(float).copy()
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = q / norm
    # make consecutive quaternions lie in the same hemisphere
    d = np.sum(q[1:] * q[:-1], axis=1)
    sgn = np.where(d < 0, -1.0, 1.0)
    q[1:] *= np.cumprod(sgn)[:, None]

    k0, a, ok = _bracket(ts, tq, max_gap)
    if len(ts) == 1:
        out = np.tile(q[0], (len(tq), 1))
        out[~ok] = np.nan
        return out
    q0, q1 = q[k0], q[k0 + 1]
    dot = np.clip(np.sum(q0 * q1, axis=1), -1.0, 1.0)
    theta = np.arccos(dot)
    s = np.sin(theta)
    small = s < 1e-6
    with np.errstate(invalid="ignore", divide="ignore"):
        w0 = np.where(small, 1 - a, np.sin((1 - a) * theta) / s)
        w1 = np.where(small, a, np.sin(a * theta) / s)
    out = w0[:, None] * q0 + w1[:, None] * q1
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    out[~ok] = np.nan
    return out


# --------------------------------------------------------------------------- #
# Mode selection
# --------------------------------------------------------------------------- #
def is_camera_locked(s: Series, frame_t: np.ndarray, tol_ns: int) -> bool:
    if len(frame_t) == 0 or len(s.t) == 0:
        return False
    idx = match_exact(frame_t, s.t, tol_ns)
    return float(np.mean(idx >= 0)) >= 0.9


def column_mode(s: Series, col: str, locked: bool, qprefixes: set) -> str:
    full = f"{s.short}.{col}"
    for pat, mode in RULES:
        if re.search(pat, full):
            return mode
    if locked:
        return "exact"
    if col in s.derived:
        return "euler"
    if "." in col and col.rsplit(".", 1)[0] in qprefixes and col[-1] in "xyzw":
        return "slerp"
    return "linear" if s.kinds.get(col) == "float" else "previous"


# --------------------------------------------------------------------------- #
# Main sync
# --------------------------------------------------------------------------- #
def target_times(data: BagData, start_s=None, stop_s=None, rate=None):
    """Frame times (or a fixed-rate grid) inside [start_s, stop_s] seconds since first frame."""
    if rate or data.n_frames == 0:
        rate = rate or 50.0
        lo = start_s if start_s is not None else 0.0
        if stop_s is None:
            ends = [s.t[-1] for s in data.series.values() if len(s.t)]
            stop_s = (max(ends) - data.t0) / 1e9 if ends else 0.0
        t = data.t0 + np.round(np.arange(lo, stop_s + 1e-9, 1.0 / rate) * 1e9).astype(np.int64)
        return t, None
    ft = data.frame_t
    m = np.ones(len(ft), bool)
    # start/stop are typed with 1 ms resolution: include a frame within half a millisecond
    if start_s is not None:
        m &= ft >= data.abs_ns(start_s) - 500_000
    if stop_s is not None:
        m &= ft <= data.abs_ns(stop_s) + 500_000
    idx = np.nonzero(m)[0]
    return ft[idx], idx


def sync(data: BagData, start_s=None, stop_s=None, columns=None,
         gap_factor: float = GAP_FACTOR, rate: float | None = None) -> SyncResult:
    """
    columns: None for everything, or an iterable of (series_name, column) pairs.
    """
    tq, fidx = target_times(data, start_s, stop_s, rate)
    res = SyncResult(t=tq, t0=data.t0, frame_idx=fidx)
    if len(tq) == 0:
        return res
    tol = int(max(1e6, 0.25 * data.frame_period_s * 1e9)) if data.n_frames > 1 else int(1e6)

    wanted: dict[str, list[str]] = {}
    if columns is None:
        for name, s in data.series.items():
            wanted[name] = [c for c in s.cols]
    else:
        for name, c in columns:
            if name in data.series and c in data.series[name].cols:
                wanted.setdefault(name, []).append(c)

    for name in sorted(wanted, key=_series_sort_key):
        s = data.series[name]
        locked = (rate is None and s.info.get("kind") not in ("derived", "tf_static")
                  and is_camera_locked(s, data.frame_t, tol))
        qpref = set(quat_groups(s.cols))
        gap = _max_gap(s.t, gap_factor)
        exact_idx = match_exact(s.t, tq, tol) if locked else None
        slerped: dict[str, np.ndarray] = {}
        res.notes[name] = {"time_source": s.time_source, "camera_locked": bool(locked),
                           "rate_hz": round(s.rate_hz, 3), "max_gap_ms": round(gap / 1e6, 2)
                           if np.isfinite(gap) else None}
        if s.info.get("kind") == "derived":
            res.notes[name].update(s.info)

        def quat_at(p):
            if p not in slerped:
                q = np.stack([s.cols[f"{p}.{a}"] for a in "xyzw"], axis=1)
                if locked:
                    out = np.full((len(tq), 4), np.nan)
                    ok = exact_idx >= 0
                    out[ok] = q[exact_idx[ok]]
                    slerped[p] = out
                else:
                    slerped[p] = slerp_series(s.t, q, tq, gap)
            return slerped[p]

        for col in wanted[name]:
            mode = column_mode(s, col, locked, qpref)
            if mode == "skip":
                continue
            v = s.cols[col]
            kind = s.kinds.get(col, "float")
            if col in s.derived:
                d = s.derived[col]
                q = quat_at(d["quat"])
                eul = dict(zip(("roll", "pitch", "yaw"), quat_to_euler_deg(*q.T)))
                out = eul[d["part"]]
                if d["part"] != "pitch":
                    # match the unwrapped native column (no +/-180 jumps)
                    okv = np.isfinite(v)
                    if okv.sum() >= 1:
                        ref = np.interp(tq.astype(float), s.t[okv].astype(float), v[okv])
                        fin = np.isfinite(out)
                        out = out.copy()
                        out[fin] -= 360.0 * np.round((out[fin] - ref[fin]) / 360.0)
                mode = "euler" if not locked else "exact"
            elif mode == "exact":
                out = take_exact(v, exact_idx)
            elif mode == "slerp":
                p, comp = col.rsplit(".", 1)
                out = quat_at(p)[:, "xyzw".index(comp)]
            elif mode == "angle_deg":
                out = interp_angle_deg(s.t, v, tq, gap)
            elif mode == "linear" and kind == "float":
                out = interp_linear(s.t, v, tq, gap)
            else:
                mode = "previous"
                out = interp_previous(s.t, v, tq, gap)
            res.names.append(f"{s.short}.{col}")
            res.kinds.append(kind)
            res.modes.append(mode)
            res.values.append(out)
    return res


# --------------------------------------------------------------------------- #
# IMU dead-reckoning over a window
# --------------------------------------------------------------------------- #
G = 9.80665
DERIVED_IMU = "/derived/imu_velocity"


def _find_series(data: BagData, want_cols, prefer=()):
    cands = [s for s in data.series.values() if all(c in s.cols for c in want_cols)]
    for p in prefer:
        for s in cands:
            if s.name.endswith(p):
                return s
    return cands[0] if cands else None


def _rotate(q, v):
    """Rotate vectors v (n,3) by unit quaternions q (n,4 xyzw)."""
    qv, w = q[:, :3], q[:, 3:4]
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


# --------------------------------------------------------------------------- #
# Forward-motion diagnostic: the four horizontal thrusters (rc/out channels 0-3)
# --------------------------------------------------------------------------- #
HORIZONTAL_CHANNELS = (0, 1, 2, 3)   # rc/out: the horizontal (forward-moving) thrusters
DERIVED_THR = "/derived/forward_motion_diag"
IDLE_US = 25                         # |pwm - 1500| at or below this counts as idle


def thruster_series(data: BagData, channels=HORIZONTAL_CHANNELS) -> Series | None:
    """
    Forward-motion diagnostic, on rc/out's own times: for each horizontal thruster channel
      ch<N>_us   rc/out channels[N] - 1500 (µs). 0 = idle, the sign gives the direction.
    Outputs reading 0 (unused) or outside 800..2200 are treated as idle (1500).
    """
    rc = _find_series(data, ["channels[0]"], prefer=("rc/out",))
    if rc is None or "rc/out" not in rc.name:
        return None
    use = [int(c) for c in channels if f"channels[{int(c)}]" in rc.cols]
    if not use:
        return None
    cols = {}
    for c in use:
        pwm = rc.cols[f"channels[{c}]"].astype(float)
        cols[f"ch{c}_us"] = np.where((pwm >= 800) & (pwm <= 2200), pwm, 1500.0) - 1500.0
    return Series(DERIVED_THR, "derived/forward_motion_diag", "derived", rc.t, cols,
                  {k: "float" for k in cols}, rc.time_source, {},
                  {"kind": "derived", "channels": use, "source": rc.name,
                   "label": "forward-motion diagnostic: horizontal thrusters, PWM - 1500"})


DERIVED_HDG = "/derived/heading"


def heading_series(data: BagData) -> Series | None:
    """
    One standard heading for the whole bag, from the IMU orientation (on the IMU's own times):
      heading_deg            compass convention: 0 = north, 90 = east, clockwise, 0..360
      heading_unwrapped_deg  the same, but continuous: crossing north goes 359 -> 361 instead of
                             359 -> 0, so it can be differenced, averaged or filtered safely
    (mavros' imu orientation is ENU: yaw 0 = east, counter-clockwise; heading = 90 - yaw.)
    """
    s_imu = _find_series(data, ["orientation.w", "orientation.z"], prefer=("/imu/data",))
    if s_imu is None:
        return None
    q = np.stack([s_imu.cols[f"orientation.{a}"] for a in "xyzw"], axis=1)
    good = np.isfinite(q).all(1) & (np.linalg.norm(np.nan_to_num(q), axis=1) > 0.5)
    if good.sum() < 2:
        return None
    t, q = s_imu.t[good], q[good] / np.linalg.norm(q[good], axis=1, keepdims=True)
    hdg = 90.0 - np.degrees(_yaw_enu(q))
    unwrapped = np.degrees(np.unwrap(np.radians(hdg)))
    unwrapped += 360.0 * np.round((hdg[0] % 360.0 - unwrapped[0]) / 360.0)   # start inside 0..360
    return Series(DERIVED_HDG, "derived/heading", "derived", t,
                  {"heading_deg": hdg % 360.0, "heading_unwrapped_deg": unwrapped},
                  {"heading_deg": "float", "heading_unwrapped_deg": "float"}, s_imu.time_source, {},
                  {"kind": "derived", "source": s_imu.name})


IMU_MODES = ("rest", "odom")
REST_CHECK_S = 1.0       # seconds after Start and before Stop used for the at-rest IMU check
REST_GYRO_RMS = 0.02     # rad/s: more gyro motion than this during the Start rest = not still
REST_ACC_STD = 0.15      # m/s²: more accelerometer variation than this = not still
# The Stop tilt correction only needs the ROV to be stopped (not travelling); rocking in place
# averages out over ALIGN_S seconds. Measuring a gyro offset at Start needs true stillness.
ALIGN_S = 3.0
ALIGN_GYRO_RMS = 0.25    # rad/s
ALIGN_G_TOL = 0.03       # mean |f| within 3 % of g: no sustained acceleration


def _yaw_enu(q):
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_to_mat(q):
    """(n,4) unit quaternions xyzw -> (n,3,3) rotation matrices (body -> world)."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], 1)


def _cumtrapz(y, dt):
    """Cumulative trapezoid integral along axis 0, starting at 0. dt: (n-1,)."""
    out = np.zeros_like(y, dtype=float)
    shape = (-1,) + (1,) * (y.ndim - 1)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dt.reshape(shape), axis=0)
    return out


def rest_check(t_s, R, f, n_s=REST_CHECK_S, g=G) -> dict:
    """
    At-rest consistency of the IMU at Start and at Stop, straight from the data (no integration).
    At rest the accelerometer must read gravity only, so f should equal R^T [0, 0, g]:
      residual_body   f - R^T g, averaged (m/s², x fwd, y left, z up): what will leak into the
                      integration as a false acceleration
      tilt_mismatch   angle between the measured gravity direction and the one the quaternion
                      implies (deg); 1° ≈ 0.17 m/s² of false horizontal acceleration
      g_measured      |f| (should be ≈ 9.81)
    The drift correction removes a leak that stays the same from Start to Stop. If the residual
    CHANGES between Start and Stop, the leftover is roughly an arch peaking near
    |change| × T / 8 in the middle of the window.
    """
    out = {}
    T = float(t_s[-1] - t_s[0])
    gw = np.array([0.0, 0.0, g])
    for name, m in (("start", t_s <= t_s[0] + n_s), ("stop", t_s >= t_s[-1] - n_s)):
        if m.sum() < 2:
            continue
        g_b = np.einsum("nji,j->ni", R[m], gw)                 # R^T g, gravity in the body frame
        fm, gm = f[m].mean(0), g_b.mean(0)
        cosang = np.dot(fm, gm) / max(np.linalg.norm(fm) * np.linalg.norm(gm), 1e-9)
        out[name] = {"residual_body": (f[m] - g_b).mean(0).round(4).tolist(),
                     "tilt_mismatch_deg": round(float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))), 3),
                     "g_measured": round(float(np.linalg.norm(f[m], axis=1).mean()), 4),
                     "accel_noise_std": round(float(np.linalg.norm(f[m].std(0))), 4)}
    if "start" in out and "stop" in out:
        d = np.array(out["stop"]["residual_body"]) - np.array(out["start"]["residual_body"])
        out["change_body"] = d.round(4).tolist()
        out["arch_estimate_m_s"] = round(float(np.linalg.norm(d[:2]) * T / 8), 3)
    return out


def _quat_from_rp(roll, pitch, yaw=0.0):
    cr, sr, cp, sp = np.cos(roll / 2), np.sin(roll / 2), np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy])


def _propagate(q0, w, t_s):
    """Integrate body rates w (n,3) from attitude q0 (midpoint rule)."""
    q = np.empty((len(t_s), 4))
    q[0] = q0
    dt = np.diff(t_s)
    for k in range(len(t_s) - 1):
        rot = 0.5 * (w[k] + w[k + 1]) * dt[k]
        ang = float(np.linalg.norm(rot))
        if ang < 1e-12:
            q[k + 1] = q[k]
            continue
        ax = rot / ang
        sa = np.sin(ang / 2)
        x2, y2, z2, w2 = ax[0] * sa, ax[1] * sa, ax[2] * sa, np.cos(ang / 2)
        x1, y1, z1, w1 = q[k]
        qn = np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                       w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])
        q[k + 1] = qn / np.linalg.norm(qn)
    return q


def _still(w, f):
    g_rms = float(np.linalg.norm(w.std(0)))
    a_std = float(np.linalg.norm(f.std(0)))
    return (g_rms <= REST_GYRO_RMS and a_std <= REST_ACC_STD), g_rms, a_std


def imu_attitude(t_s, w, f, rest_s=None):
    """
    Attitude from the IMU's own sensors only (no autopilot estimate). Yaw starts at 0, so world
    x is the hull's heading at Start (no compass is used).

      Start tilt     the accelerometer's gravity direction over the first rest_s seconds, with
                     each sample first rotated back to the Start attitude using the gyro, so a
                     little rocking during that time doesn't bias it. Its magnitude is used as g.
      Gyro offset    measured over the same seconds IF the vehicle is still there; otherwise 0
                     (the gyro's own calibration at boot is trusted).
      Propagation    gyro integration (midpoint rule).
      Stop check     if the last rest_s seconds are still too, the accelerometer's gravity there
                     shows how far the integrated tilt drifted; the constant gyro offset that
                     explains it is removed (roll/pitch only - yaw drift can't be seen this way).
    Returns (q (n,4) xyzw body->world, info).
    """
    rest_s = REST_CHECK_S if rest_s is None else rest_s
    rest = t_s <= t_s[0] + rest_s
    if rest.sum() < 3:
        rest = np.arange(len(t_s)) < 3
    end = t_s >= t_s[-1] - max(rest_s, ALIGN_S)
    still, g_rms, a_std = _still(w[rest], f[rest])
    _, g_rms2, a_std2 = _still(w[end], f[end])
    bias = w[rest].mean(0) if still else np.zeros(3)

    def start_attitude(wc):
        qr = _propagate(np.array([0.0, 0.0, 0.0, 1.0]), wc[rest], t_s[rest])
        fr = np.einsum("nij,nj->ni", quat_to_mat(qr), f[rest]).mean(0)     # in the Start frame
        return fr, _quat_from_rp(float(np.arctan2(fr[1], fr[2])), float(np.arctan2(-fr[0], np.hypot(fr[1], fr[2]))))

    fr, q0 = start_attitude(w - bias)
    g = float(np.linalg.norm(fr))
    q = _propagate(q0, w - bias, t_s)
    g_end = float(np.linalg.norm(f[end].mean(0)))
    # rocking moves gravity between the axes (large accel variation) but averages out in the comparison,
    # so only "no sustained acceleration" (mean |f| ≈ g) and moderate rotation are required here
    stop_still = (g_rms2 <= ALIGN_GYRO_RMS and abs(g_end / g - 1) <= ALIGN_G_TOL)
    info = {"attitude": "IMU only (gyro + accelerometer); autopilot orientation not used",
            "start_still": bool(still), "start_gyro_rms_rad_s": round(g_rms, 4), "start_accel_std": round(a_std, 4),
            "stop_still": bool(stop_still), "stop_gyro_rms_rad_s": round(g_rms2, 4), "stop_accel_std": round(a_std2, 4),
            "gyro_offset_source": "measured at Start (still)" if still else "none - Start not still, gyro boot calibration trusted",
            "rest_used_s": round(float(t_s[rest][-1] - t_s[0]), 3), "gravity_measured": round(g, 4)}

    if stop_still and end.sum() >= 3 and not end[0]:
        dt = np.diff(t_s)
        for _ in range(4):                       # refine: the tilt error is nearly linear in the offset
            R = quat_to_mat(q)
            pred = np.einsum("nji,j->ni", R[end], np.array([0.0, 0.0, 1.0])).mean(0)
            pred /= np.linalg.norm(pred)
            meas = f[end].mean(0) / np.linalg.norm(f[end].mean(0))
            d = R[end][len(R[end]) // 2] @ (pred - meas)          # world-frame tilt error at Stop
            phi = np.array([d[1], -d[0]])                          # rotation error about world x, y
            M = _cumtrapz(R, dt)[-1][:2]                           # how a body offset turns into it
            db = np.linalg.lstsq(M, phi, rcond=None)[0]
            bias = bias + db
            fr, q0 = start_attitude(w - bias)
            q = _propagate(q0, w - bias, t_s)
        R = quat_to_mat(q)
        pred = np.einsum("nji,j->ni", R[end], np.array([0.0, 0.0, 1.0])).mean(0)
        meas = f[end].mean(0)
        ang = np.degrees(np.arccos(np.clip(pred @ meas / np.linalg.norm(pred) / np.linalg.norm(meas), -1, 1)))
        info["stop_alignment"] = (f"applied over the last {ALIGN_S:.0f} s: gyro offset chosen so the tilt matches the "
                                  f"accelerometer at Stop (left over {ang:.2f}°)")
    else:
        info["stop_alignment"] = ("not applied - the ROV is still travelling or accelerating at Stop" if not stop_still
                                  else "not applied")
    info["gyro_bias_rad_s"] = np.round(bias, 5).tolist()
    r0, p0, _ = quat_to_euler_deg(*q[:1].T)
    info["start_roll_deg"], info["start_pitch_deg"] = round(float(r0[0]), 3), round(float(p0[0]), 3)
    return q, info


def integrate_imu(data: BagData, start_s: float, stop_s: float, mode: str = "rest",
                  imu: str | None = None, odom: str | None = None,
                  yaw_offset_deg: float = 0.0) -> Series | None:
    """
    Velocity from the IMU alone, between start_s and stop_s (seconds since the first frame).
    The autopilot's orientation estimate is NOT used; only the IMU's gyro (angular_velocity)
    and accelerometer (linear_acceleration), body FLU:

      Attitude R               from imu_attitude(): roll/pitch from the accelerometer while at
                               rest at Start, gyro offset from the same rest, then gyro
                               integration. Yaw starts at 0 = the heading at Start.
      Gravity g                the magnitude the accelerometer reads at rest (not 9.81)
      World acceleration       a = R f - [0, 0, g]
      World velocity           v = integral of a (trapezoid, on the IMU's own samples)
      Drift correction         "rest": v = 0 at Start; whatever is left at Stop is drift and is
                               removed linearly over the window. "odom": anchored to odometry.
      Body velocity            surge, sway, heave = R^T v      (nose, left, up)
      Heading frame            forward = body x flattened onto the horizontal

    Start MUST be at rest (the first REST_CHECK_S seconds). info['rest_check'] then shows how far
    the integrated attitude has drifted by Stop (if Stop is at rest too). att.roll/pitch/yaw_deg
    is the IMU-only attitude. yaw_offset_deg turns the reported body axes about z.
    """
    if mode not in IMU_MODES:
        raise ValueError(f"mode must be one of {IMU_MODES}")
    s_imu = data.series.get(imu) if imu else _find_series(
        data, ["angular_velocity.x", "linear_acceleration.x"], prefer=("/imu/data",))
    if s_imu is None:
        return None
    t_lo, t_hi = data.abs_ns(start_s), data.abs_ns(stop_s)
    m = (s_imu.t >= t_lo) & (s_imu.t <= t_hi)
    if m.sum() < 3:
        return None
    t = s_imu.t[m]
    f = np.stack([s_imu.cols[f"linear_acceleration.{a}"][m] for a in "xyz"], axis=1).astype(float)
    w = np.stack([s_imu.cols[f"angular_velocity.{a}"][m] for a in "xyz"], axis=1).astype(float)
    good = np.isfinite(f).all(1) & np.isfinite(w).all(1)
    t, f, w = t[good], f[good], w[good]
    if len(t) < 3:
        return None
    tsec = (t - t[0]) / 1e9
    dt = np.diff(tsec)
    if tsec[-1] <= 0:
        return None
    q, att_info = imu_attitude(tsec, w, f)               # IMU only, no autopilot estimate
    R = quat_to_mat(q)                                   # body -> world (yaw 0 = heading at Start)
    g = att_info["gravity_measured"]

    a_w = np.einsum("nij,nj->ni", R, f) - np.array([0.0, 0.0, g])
    vel_raw = _cumtrapz(a_w, dt)

    info = {"kind": "derived", "mode": mode, "imu": s_imu.name, "gravity": g, **att_info,
            "window_t_s": [start_s, stop_s], "duration_s": round(float(tsec[-1]), 3),
            "yaw_offset_deg": float(yaw_offset_deg), "imu_rate_hz": round((len(t) - 1) / max(tsec[-1], 1e-9), 2),
            "imu_samples": int(len(t)), "imu_max_gap_ms": round(float(dt.max() * 1e3), 1),
            "imu_median_dt_ms": round(float(np.median(dt) * 1e3), 2),
            "rest_check": rest_check(tsec, R, f, g=g)}
    v_start = np.zeros(3)
    v_end = np.zeros(3)
    if mode == "odom":
        s_odom = data.series.get(odom) if odom else _find_series(
            data, ["twist.twist.linear.x"], prefer=("global_position/local",))
        if s_odom is None:
            return None
        gap = _max_gap(s_odom.t, GAP_FACTOR)
        v_od = np.stack([interp_linear(s_odom.t, s_odom.cols[f"twist.twist.linear.{a}"],
                                       t[[0, -1]], gap) for a in "xyz"], axis=1)
        if not np.isfinite(v_od).all():
            return None
        v_start, v_end = v_od[0], v_od[1]
        info["v0_from"] = s_odom.name
    else:
        info["v0_from"] = "rest (0) at Start and Stop"
    vel_raw += v_start
    err = vel_raw[-1] - v_end
    vel = vel_raw - err[None, :] * (tsec / tsec[-1])[:, None]
    info["end_error_before_correction"] = err.round(4).tolist()
    info["implied_accel_bias"] = (err / tsec[-1]).round(5).tolist()

    # body axes from the quaternion, optionally turned by the mounting yaw offset
    th = np.radians(yaw_offset_deg)
    Rz = np.array([[np.cos(th), -np.sin(th), 0.0], [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])
    Rb = R @ Rz
    to_body = lambda v: np.einsum("nji,nj->ni", Rb, v)  # noqa: E731  R^T v
    body, body_raw, acc_body = to_body(vel), to_body(vel_raw), to_body(a_w)

    nose = Rb[:, :, 0]
    fh = nose[:, :2] / np.maximum(np.linalg.norm(nose[:, :2], axis=1, keepdims=True), 1e-9)
    left = np.stack([-fh[:, 1], fh[:, 0]], 1)

    def heading_frame(v):
        return (v[:, :2] * fh).sum(1), (v[:, :2] * left).sum(1), v[:, 2]

    fwd, side, vert = heading_frame(vel)
    fwd_raw, _, _ = heading_frame(vel_raw)
    afwd, aside, avert = heading_frame(a_w)
    disp = _cumtrapz(vel, dt)
    speed = np.hypot(vel[:, 0], vel[:, 1])
    dist = _cumtrapz(speed, dt)
    fdist = _cumtrapz(fwd, dt)
    surge_d, sway_d = _cumtrapz(body[:, 0], dt), _cumtrapz(body[:, 1], dt)
    hdg = np.degrees(np.unwrap(np.arctan2(fh[:, 1], fh[:, 0])))

    info["distance_m"] = round(float(dist[-1]), 3)
    info["forward_distance_m"] = round(float(fdist[-1]), 3)
    info["surge_distance_m"] = round(float(surge_d[-1]), 3)
    info["sway_distance_m"] = round(float(sway_d[-1]), 3)
    info["travel_angle_deg"] = round(float(np.degrees(np.arctan2(sway_d[-1], surge_d[-1]))), 2) \
        if abs(surge_d[-1]) + abs(sway_d[-1]) > 1e-3 else 0.0
    info["peak_forward_m_s"] = round(float(fwd[np.argmax(np.abs(fwd))]), 3)
    info["mean_forward_m_s"] = round(float(np.mean(fwd)), 3)
    info["peak_sway_m_s"] = round(float(body[np.argmax(np.abs(body[:, 1])), 1]), 3)
    info["min_surge_m_s"] = round(float(body[:, 0].min()), 3)
    info["heading_change_deg"] = round(float(hdg.max() - hdg.min()), 1)

    cols = {"forward_speed": fwd, "sideways_speed": side, "vertical_speed": vert, "speed": speed,
            "distance": dist, "forward_distance": fdist, "forward_speed_raw": fwd_raw,
            "acc_forward": afwd, "acc_sideways": aside, "acc_vertical": avert}
    for i, a in enumerate(("surge", "sway", "heave")):
        cols[a] = body[:, i]
    cols["surge_distance"], cols["sway_distance"] = surge_d, sway_d
    for i, a in enumerate(("surge", "sway", "heave")):
        cols[f"{a}_raw"] = body_raw[:, i]
    for i, a in enumerate(("surge", "sway", "heave")):
        cols[f"acc_{a}"] = acc_body[:, i]
    for i, a in enumerate("enu"):
        cols[f"vel.{a}"] = vel[:, i]
    for i, a in enumerate("enu"):
        cols[f"disp.{a}"] = disp[:, i]
    for i, a in enumerate("enu"):
        cols[f"acc_world.{a}"] = a_w[:, i]
    r_, p_, y_ = quat_to_euler_deg(*q.T)                 # the IMU-only attitude itself
    cols["att.roll_deg"], cols["att.pitch_deg"] = r_, p_
    cols["att.yaw_deg"] = np.degrees(np.unwrap(np.radians(y_)))
    return Series(DERIVED_IMU, "derived/imu_vel", "derived", t, cols,
                  {k: "float" for k in cols}, s_imu.time_source, {}, info)


def rest_check_text(info: dict) -> str:
    """Plain-language reading of info['rest_check']."""
    rc = info.get("rest_check") or {}
    parts = []
    for k in ("start", "stop"):
        if k in rc:
            r = rc[k]
            parts.append(f"{k}: tilt mismatch {r['tilt_mismatch_deg']:.2f}°, residual {r['residual_body']} m/s², "
                         f"|g| {r['g_measured']:.3f}")
    if "change_body" in rc:
        parts.append(f"change start→stop {rc['change_body']} m/s² → expected arch ≈ {rc['arch_estimate_m_s']:.2f} m/s")
    return "; ".join(parts)


def imu_summary(info: dict) -> str:
    """One-line description of an integration result."""
    err = np.round(info.get("end_error_before_correction", []), 3).tolist()
    return (f"IMU velocity {info['window_t_s'][0]:.2f}–{info['window_t_s'][1]:.2f} s ({info['mode']} mode): "
            f"forward distance {info.get('forward_distance_m', 0):.2f} m, surge {info.get('surge_distance_m', 0):.2f} m, "
            f"sway {info.get('sway_distance_m', 0):.2f} m, travel angle {info.get('travel_angle_deg', 0):+.1f}°, "
            f"drift removed {err} m/s  |  at rest: {rest_check_text(info)}")


REFERENCE_TAGS = (0, 20)   # tags fixed relative to the camera (left out of the pixel-centre dialog)


# --------------------------------------------------------------------------- #
# TF tree: any frame relative to any other, in both directions, measured over time
# --------------------------------------------------------------------------- #
DERIVED_TF_PREFIX = "/derived/tf"
TF_EDGE_RE = re.compile(r"\[(.+)->(.+)\]$")


def quat_mul(a, b):
    """Hamilton product of (n,4) xyzw quaternions."""
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], 1)


def quat_conj(q):
    return q * np.array([-1.0, -1.0, -1.0, 1.0])


def tf_compose(ta, qa, tb, qb):
    """T_a ∘ T_b: pose b expressed through a. (n,3),(n,4) each."""
    return ta + _rotate(qa, tb), quat_mul(qa, qb)


def tf_inverse(t, q):
    qi = quat_conj(q)
    return -_rotate(qi, t), qi


def tf_edges(data: BagData) -> list[dict]:
    """Every edge of the tf tree: {'parent','child','kind':'dynamic'|'static', 'series' | 't','q'}."""
    from bagcache import static_tf_edges
    edges = []
    for s in data.series.values():
        if s.info.get("kind") != "tf":
            continue
        m = TF_EDGE_RE.search(s.name)
        if m and all(f"translation.{a}" in s.cols for a in "xyz") and all(f"rotation.{a}" in s.cols for a in "xyzw"):
            edges.append({"parent": m.group(1).strip("/"), "child": m.group(2).strip("/"),
                          "kind": "dynamic", "series": s})
    for e in static_tf_edges(data.meta):
        edges.append({"parent": e["parent"], "child": e["child"], "kind": "static",
                      "t": np.array(e["translation"], float), "q": np.array(e["rotation_xyzw"], float)})
    return edges


def tf_frames(data: BagData) -> list[str]:
    out = set()
    for e in tf_edges(data):
        out.update((e["parent"], e["child"]))
    return sorted(out)


def _edge_at(data: BagData, e: dict, tq: np.ndarray, tol: int):
    """Edge transform parent->child at times tq: (n,3), (n,4), NaN where not available."""
    n = len(tq)
    if e["kind"] == "static":
        q = e["q"] / np.linalg.norm(e["q"])
        return np.tile(e["t"], (n, 1)), np.tile(q, (n, 1))
    s = e["series"]
    tr = np.stack([s.cols[f"translation.{a}"] for a in "xyz"], 1).astype(float)
    qq = np.stack([s.cols[f"rotation.{a}"] for a in "xyzw"], 1).astype(float)
    if is_camera_locked(s, data.frame_t, tol):
        idx = match_exact(s.t, tq, tol)
        ok = idx >= 0
        t_out, q_out = np.full((n, 3), np.nan), np.full((n, 4), np.nan)
        t_out[ok], q_out[ok] = tr[idx[ok]], qq[idx[ok]]
    else:
        gap = _max_gap(s.t, GAP_FACTOR)
        t_out = np.stack([interp_linear(s.t, tr[:, k], tq, gap) for k in range(3)], 1)
        q_out = slerp_series(s.t, qq, tq, gap)
    with np.errstate(invalid="ignore", divide="ignore"):
        q_out = q_out / np.linalg.norm(q_out, axis=1, keepdims=True)
    return t_out, q_out


def tf_path(edges: list[dict], a: str, b: str, exclude=None, dynamic_only=False):
    """Shortest chain of edges from frame a to frame b: [(edge, forward?)] or None."""
    from collections import deque
    adj: dict = {}
    for e in edges:
        if e is exclude or (dynamic_only and e["kind"] != "dynamic"):
            continue
        adj.setdefault(e["parent"], []).append((e["child"], e, True))
        adj.setdefault(e["child"], []).append((e["parent"], e, False))
    prev = {a: None}
    dq = deque([a])
    while dq:
        u = dq.popleft()
        if u == b:
            break
        for v, e, fwd in adj.get(u, []):
            if v not in prev:
                prev[v] = (u, e, fwd)
                dq.append(v)
    if b not in prev or a == b:
        return None
    path, u = [], b
    while prev[u] is not None:
        pu, e, fwd = prev[u]
        path.append((e, fwd))
        u = pu
    return path[::-1]


def tf_lookup(data: BagData, a: str, b: str, path):
    """Pose of frame b in frame a on every camera frame time: t (n,3), q (n,4), valid mask."""
    tq = data.frame_t
    tol = int(max(1e6, 0.25 * data.frame_period_s * 1e9)) if data.n_frames > 1 else int(1e6)
    n = len(tq)
    t_acc, q_acc = np.zeros((n, 3)), np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
    for e, fwd in path:
        te, qe = _edge_at(data, e, tq, tol)
        if not fwd:
            te, qe = tf_inverse(te, qe)
        t_acc, q_acc = tf_compose(t_acc, q_acc, te, qe)
    ok = np.isfinite(t_acc).all(1) & np.isfinite(q_acc).all(1)
    return t_acc, q_acc, ok


def _tf_series(name, short, msgtype, t, tr, q, info, mm=False):
    """Series with translation (m, or translation_mm in mm when mm=True) and rotation
    (quaternion + roll/pitch/yaw in degrees)."""
    from bagcache import add_euler_columns, make_quats_continuous
    if mm:
        cols = {f"translation_mm.{a}": tr[:, k] * 1000.0 for k, a in enumerate("xyz")}
        ok = np.isfinite(tr).all(1)
        if ok.any():
            info["translation_mm_mean"] = np.round(tr[ok].mean(0) * 1000.0, 4).tolist()
            info["translation_mm_std"] = np.round(tr[ok].std(0) * 1000.0, 4).tolist()
    else:
        cols = {f"translation.{a}": tr[:, k] for k, a in enumerate("xyz")}
    cols.update({f"rotation.{a}": q[:, k] for k, a in enumerate("xyzw")})
    s = Series(name, short, msgtype, t, cols, {c: "float" for c in cols}, "header", {}, info)
    make_quats_continuous(s)
    add_euler_columns(s)
    return s


def tf_pair_series(data: BagData, a: str, b: str, path=None) -> Series | None:
    """Measured pose of frame b in frame a over time (one direction), translation in mm."""
    edges = tf_edges(data)
    path = path or tf_path(edges, a, b)
    if path is None or not any(e["kind"] == "dynamic" for e, _ in path):
        return None
    tr, q, ok = tf_lookup(data, a, b, path)
    if ok.sum() == 0:
        return None
    chain = " → ".join([a] + [(e["child"] if f else e["parent"]) for e, f in path])
    info = {"kind": "tf_measured", "from": a, "to": b, "chain": chain, "frames_seen": int(ok.sum())}
    return _tf_series(f"{DERIVED_TF_PREFIX}[{a}->{b}]", f"derived/tf[{a}->{b}]", "derived",
                      data.frame_t[ok], tr[ok], q[ok], info, mm=True)


# Words that mark a reference ("zero") frame for a live tag: tag_0_zero, static_tag_0, tag_0_static ...
REF_WORDS = ("zero", "static")
REF_WORD = re.compile(r"(^|[_\-/.:])(?:" + "|".join(REF_WORDS) + r")(?=$|[_\-/.:])", re.I)


def _strip_static(name: str) -> str:
    """'tag_0_zero' / 'static_tag_0' / 'tag_0_static' / 'zero/tag_0' -> 'tag_0' (unchanged if none)."""
    return REF_WORD.sub(lambda m: m.group(1) if m.start() == 0 else "", name).strip("_-/.:")


def static_live_pairs(data: BagData, edges=None) -> list[tuple[str, str]]:
    """
    (reference frame, live frame) pairs, generated automatically for every tag:
    a frame whose name carries a reference word (tag_#_zero, static_tag_#, tag_#_static) is
    paired with the live /tf frame of the same name once that word is removed:
    tag_0_zero <-> tag_0. If the names don't match exactly, it falls back to the live frame
    ending in the same tag number, when exactly one does. The reference frame may come from
    /tf_static or /tf.
    """
    edges = edges if edges is not None else tf_edges(data)
    frames = {f for e in edges for f in (e["parent"], e["child"])}
    refs = sorted(f for f in frames if _strip_static(f) != f)
    live = {f for e in edges if e["kind"] == "dynamic" for f in (e["parent"], e["child"])} - set(refs)
    by_name = {f.lower(): f for f in live}
    pairs = []
    for rf in refs:
        base = _strip_static(rf)
        lf = by_name.get(base.lower())
        if lf is None:
            num = re.search(r"(\d+)$", base)
            cands = [f for f in live if num and re.search(rf"(?<!\d){num.group(1)}$", f)]
            lf = cands[0] if len(cands) == 1 else None
        if lf is not None:
            pairs.append((rf, lf))
    return pairs


def tf_tree_series(data: BagData) -> tuple[list[Series], list[str]]:
    """
    Every live tag against its reference ("zero") frame, generated for every pair found by
    static_live_pairs (tag_N_zero -> tag_N, static_tag_N -> tag_N): the pose of the live tag in
    the reference frame on every camera frame,
        T(tag_N_zero -> tag_N) = inverse(T(root -> tag_N_zero)) * T(root -> tag_N)
    (root = whatever frame both hang from, usually the camera; longer chains are followed).
    It is identity when the tag sits exactly on its zero pose, so translation_mm.x/y/z and
    rotation.*_deg ARE the deviation, along the zero frame's own axes.
    Static edges that are not part of such a pair are listed with their constant value.
    Returns (series, notes).
    """
    edges = tf_edges(data)
    out, notes = [], []
    if data.n_frames == 0:
        return out, notes
    used = set()
    pairs = static_live_pairs(data, edges)
    paired = {a for a, _ in pairs}
    for sf, lf in pairs:
        path = tf_path(edges, sf, lf)
        s = tf_pair_series(data, sf, lf, path=path)
        if s is None:
            notes.append(f"{lf} in {sf}: not measured (no chain through /tf between them)")
            continue
        s.info["deviation_of"] = lf
        out.append(s)
        used.update(id(e) for e, _ in path if e["kind"] == "static")
    unpaired = sorted({f for e in edges if e["kind"] == "static" for f in (e["parent"], e["child"])
                       if _strip_static(f) != f and f not in paired})
    for f in unpaired:
        notes.append(f"{f}: reference frame with no live frame of the same name or tag number in /tf")
    t2 = data.frame_t[[0, -1]]
    for e in edges:
        if e["kind"] != "static" or id(e) in used:
            continue
        q0 = e["q"] / np.linalg.norm(e["q"])
        out.append(_tf_series(f"/tf_static[{e['parent']}->{e['child']}]", f"tf_static[{e['parent']}->{e['child']}]",
                              "geometry_msgs/msg/TransformStamped", t2,
                              np.tile(e["t"], (2, 1)), np.tile(q0, (2, 1)),
                              {"kind": "tf_static", "from": e["parent"], "to": e["child"],
                               "note": "static value (constant)"}, mm=True))
    return out, notes


def tf_tree_text(data: BagData) -> str:
    """The whole tf tree for Bag info: dynamic edges with rate, static edges with values."""
    lines = ["TF tree (all edges):"]
    for e in tf_edges(data):
        if e["kind"] == "dynamic":
            s = e["series"]
            lines.append(f"  /tf         {e['parent']} -> {e['child']}   {len(s.t)} msgs @ {s.rate_hz:.2f} Hz")
        else:
            r, p, y = (float(v[0]) for v in quat_to_euler_deg(*[np.array([c]) for c in e["q"]]))
            lines.append(f"  /tf_static  {e['parent']} -> {e['child']}   t = [{e['t'][0]:+.5f}, {e['t'][1]:+.5f}, "
                         f"{e['t'][2]:+.5f}] m   rpy = [{r:+.2f}, {p:+.2f}, {y:+.2f}] deg")
    dev = [s for s in data.series.values() if s.info.get("deviation_of") and "translation_mm_mean" in s.info]
    if dev:
        lines.append("\nLive tags in their zero frames (inverse(zero) × live; 0 = on the zero pose):")
    for s in sorted(dev, key=lambda s: s.name):
        lines.append(f"  {s.info['to']} in {s.info['from']}: mean {s.info['translation_mm_mean']} mm, "
                     f"std {s.info['translation_mm_std']} mm, {s.info['frames_seen']} frames  ({s.info['chain']})")
    return "\n".join(lines)


def tag_pixel_position(data: BagData, ser: Series, rectified: bool = True) -> dict | None:
    """
    Lateral tag position from the detection's pixel centre instead of the 3-D pose:
        x = (u - cx) * z / fx,   y = (v - cy) * z / fy      (metres, camera optical frame)
    The pixel centre is much less noisy than the pose's x/y (on surfaceLine1, tag 18: ~6 um vs
    ~41 um in x). Nothing is smoothed: z is the median pose depth (a constant scale). fx, fy, cx, cy come from camera_info: P (rectified image) or K (raw image).
    Adds translation.x_px / translation.y_px to the tf series.
    """
    ci = data.meta.get("camera_info")
    m = re.search(r"(\d+)\]?$", ser.name)
    if not ci or not m:
        return None
    tag_id = int(m.group(1))
    det = next((s for s in data.series.values()
                if s.info.get("kind") == "tag" and s.info.get("tag_id") == tag_id), None)
    if det is None or "centre.x" not in det.cols:
        return None
    P = np.array(ci["p"], float).reshape(3, 4)
    K = np.array(ci["k"], float).reshape(3, 3)
    use_p = rectified and P[0, 0] > 0
    fx, fy, cx, cy = (P[0, 0], P[1, 1], P[0, 2], P[1, 2]) if use_p else (K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    idx = match_exact(det.t, ser.t, int(2e6))
    ok = idx >= 0
    u = np.full(len(ser.t), np.nan)
    v = np.full(len(ser.t), np.nan)
    u[ok] = det.cols["centre.x"][idx[ok]]
    v[ok] = det.cols["centre.y"][idx[ok]]
    # Fixed depth scale: median of the pose z. No filtering anywhere, so every frequency the
    # camera can see is kept. (The pose z is ~57 um noisy; a real 150 um z change would change
    # x/y by < 0.1 %, so a constant z costs nothing and adds no z noise to x/y.)
    zr = ser.cols["translation.z"].astype(float)
    z = float(np.nanmedian(zr)) if np.isfinite(zr).any() else np.nan
    calc = ser.info.setdefault("calc_cols", [])
    for name, arr in (("translation.x_px", (u - cx) * z / fx), ("translation.y_px", (v - cy) * z / fy)):
        ser.cols[name] = arr
        ser.kinds[name] = "float"
        if name not in calc:
            calc.append(name)
    info = {"matched": int(ok.sum()), "of": int(len(ok)), "intrinsics": "P (rectified)" if use_p else "K (raw)",
            "fx": fx, "fy": fy, "cx": cx, "cy": cy, "z_m": z}
    ser.info["pixel_position"] = info
    return info


def _series_sort_key(name: str):
    # camera-related first, then mavros, then the rest, alphabetical inside
    pri = 0 if "detections" in name or name.startswith("/tf") else 1
    return (pri, name)


# --------------------------------------------------------------------------- #
# CSV writing
# --------------------------------------------------------------------------- #
def _fmt_ns(t: int) -> str:
    return f"{t // 1_000_000_000}.{t % 1_000_000_000:09d}"


def _fmt_col(v: np.ndarray, kind: str) -> list[str]:
    if v.dtype == object or v.dtype.kind in "US":
        return [str(x) for x in v]
    if kind in ("int", "bool"):
        return ["" if x != x else str(int(round(x))) for x in v.tolist()]
    return ["" if x != x else f"{x:.10g}" for x in v.tolist()]


def write_csv(res: SyncResult, path: str | Path, meta: dict | None = None, sidecar: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [[_fmt_ns(int(t)) for t in res.t],
            [f"{x:.6f}" for x in (res.t - res.t0) / 1e9],
            [str(int(i)) for i in res.frame_idx] if res.frame_idx is not None else [""] * res.n_rows]
    cols += [_fmt_col(v, k) for v, k in zip(res.values, res.kinds)]
    bad = [n for n, c in zip(["timestamp", "t_s", "frame_idx"] + res.names, cols) if len(c) != res.n_rows]
    if bad:
        raise RuntimeError(f"internal error: column length mismatch in {bad[:5]}")
    header = ["timestamp", "t_s", "frame_idx"] + res.names
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(zip(*cols))
    side = {
        "csv": path.name,
        "rows": res.n_rows,
        "window_t_s": [float((res.t[0] - res.t0) / 1e9), float((res.t[-1] - res.t0) / 1e9)] if res.n_rows else None,
        "window_timestamp": [_fmt_ns(int(res.t[0])), _fmt_ns(int(res.t[-1]))] if res.n_rows else None,
        "frame_idx_range": [int(res.frame_idx[0]), int(res.frame_idx[-1])]
        if res.frame_idx is not None and res.n_rows else None,
        "columns": {n: m for n, m in zip(res.names, res.modes)},
        "series": res.notes,
    }
    if meta:
        side.update(meta)
    if sidecar:
        path.with_suffix(".sync.json").write_text(json.dumps(side, indent=1))
    return path


# --------------------------------------------------------------------------- #
# Even re-timing: arrival-stamped topics (MAVROS with timesync off) -> evenly spaced stamps
# --------------------------------------------------------------------------- #
RETIME_DEFAULT = ("imu/data", "bluerov/imu", "bluerov/depth")   # series whose name ends with one of these
RETIME_ENVELOPE_PCT = 1.0      # the new stamps follow the earliest 1 % of arrivals (arrivals are only ever late)
RETIME_CHUNK_S = 2.0           # length of the pieces checked for dropped messages
RETIME_MAX_STEP = 0.5          # refuse if the earliest arrivals wander by more than this x the period


def retime_check(t_ns: np.ndarray) -> dict:
    """
    Fit evenly spaced stamps to a topic stamped on ARRIVAL (bunched by the link):
        new_k = t_first + offset + k * period         k = 0, 1, 2 ... in arrival order
    period and offset come from a straight-line fit through all stamps, then the line is moved
    down to the earliest arrivals (RETIME_ENVELOPE_PCT), since a message can arrive late but
    never early. That only holds if no message was dropped: one missing message shifts every
    later index by one period. So the earliest-arrival level is compared piece by piece
    (RETIME_CHUNK_S); if it wanders by more than RETIME_MAX_STEP x period, 'ok' is False.
    Returns {'ok', 'reason', 'period_ms', 'rate_hz', 't_new' (if ok), 'shift_ms' stats ...}.
    """
    t = np.asarray(t_ns, np.int64)
    n = len(t)
    if n < 20:
        return {"ok": False, "reason": f"only {n} messages"}
    k = np.arange(n, dtype=float)
    ts = (t - t[0]) / 1e9
    dt_s = np.diff(ts)
    drops = []
    # A dropped message makes every later message look one period later than its index says, so
    # the earliest-arrival level (the "floor": the least-delayed messages) steps up by one period.
    # Windows of RETIME_CHUNK_S / 2 always hold some least-delayed messages, so their floors are
    # reliable. Where the floor steps up, the slot goes right after the last message that is still
    # on the lower floor: a message can arrive late, but never before it was measured.
    period0 = float(np.median(dt_s[dt_s > 0])) if np.any(dt_s > 0) else 0.0
    b = (ts[-1] / (n - 1)) if n > 1 else period0
    W = max(20, int(round(RETIME_CHUNK_S / 2 / b))) if b > 0 else 20
    for _ in range(100):
        b, a = np.polyfit(k, ts, 1)                # ts ≈ a + b k
        if b <= 0:
            return {"ok": False, "reason": "stamps do not increase"}
        r = ts - (a + b * k)
        nw = n // W
        if nw < 2:
            break
        bounds = [(i * W, (i + 1) * W if i < nw - 1 else n) for i in range(nw)]
        floors = np.array([r[s0:s1].min() for s0, s1 in bounds])
        step = np.diff(floors)
        j = int(np.argmax(step))
        m = int(round(step[j] / b))
        if step[j] < 0.6 * b or m < 1:
            break
        lo, hi = bounds[j][0], bounds[j + 1][1]
        low = np.flatnonzero(r[lo:hi] < floors[j] + 0.5 * b)
        split = lo + int(low.max()) + 1 if len(low) else bounds[j + 1][0]
        if split >= n:
            break
        k[split:] += m
        drops.append({"before_message": int(split), "t_s": round(float(ts[split]), 3), "missing": m})
    b, a = np.polyfit(k, ts, 1)
    r = ts - (a + b * k)
    base = float(np.percentile(r, RETIME_ENVELOPE_PCT))
    span = ts[-1]
    nch = max(1, int(span // RETIME_CHUNK_S))
    floors = [float(np.percentile(c, RETIME_ENVELOPE_PCT * 5)) for c in np.array_split(r, nch) if len(c) >= 10]
    wander = (max(floors) - min(floors)) if floors else 0.0
    t_new = t[0] + np.round((a + base + b * k) * 1e9).astype(np.int64)
    shift = (t_new - t) / 1e6                       # ms, new - old (mostly negative: arrivals were late)
    dt_old = np.diff(t) / 1e6
    out = {"period_ms": round(b * 1e3, 4), "rate_hz": round(1.0 / b, 3), "messages": n,
           "dropped": int(sum(d["missing"] for d in drops)), "drops": drops[:50],
           "earliest_arrival_wander_ms": round(wander * 1e3, 2),
           "old_spacing_ms": {"median": round(float(np.median(dt_old)), 2),
                              "p10": round(float(np.percentile(dt_old, 10)), 2),
                              "p90": round(float(np.percentile(dt_old, 90)), 2),
                              "max": round(float(dt_old.max()), 2)},
           "shift_ms": {"median": round(float(np.median(shift)), 2),
                        "p5": round(float(np.percentile(shift, 5)), 2),
                        "min": round(float(shift.min()), 2), "max": round(float(shift.max()), 2)}}
    if wander > RETIME_MAX_STEP * b:
        out.update(ok=False, reason=(f"the earliest arrivals wander by {wander * 1e3:.1f} ms "
                                     f"(> {RETIME_MAX_STEP:g} x the {b * 1e3:.2f} ms period): messages were "
                                     "dropped in a way that couldn't be located, so evenly spaced stamps "
                                     "would slide out of place"))
        return out
    out.update(ok=True, reason="", t_new=t_new)
    return out


def retime_series(data: BagData, patterns=RETIME_DEFAULT) -> list[dict]:
    """Give every recorded series whose name ends with one of `patterns` evenly spaced stamps
    (see retime_check), in place. A series that fails the check is left unchanged. Run it right
    after load_bag, before anything is derived or synced. Returns one report per series."""
    reports = []
    for s in list(data.series.values()):
        if s.msgtype == "derived" or not any(s.name.rstrip("/").endswith(p) for p in patterns):
            continue
        chk = retime_check(s.t)
        rep = {"series": s.short, "time_source": s.time_source,
               **{k: v for k, v in chk.items() if k != "t_new"}}
        if chk["ok"]:
            s.t = chk["t_new"]
            s.info["retimed"] = {k: v for k, v in rep.items() if k not in ("series",)}
            rep["applied"] = True
        else:
            rep["applied"] = False
        reports.append(rep)
    return reports


def retime_text(reports: list[dict]) -> str:
    lines = []
    for r in reports:
        if r.get("applied"):
            sm = r["shift_ms"]
            dr = r.get("dropped", 0)
            lines.append(f"{r['series']}: re-timed to an even {r['period_ms']:.3f} ms ({r['rate_hz']:.2f} Hz), "
                         f"{r['messages']} msgs, "
                         + (f"{dr} dropped (left as empty slots, at t = "
                            + ", ".join(f"{x['t_s']:.2f}" for x in r.get("drops", [])[:8])
                            + (" ..." if len(r.get("drops", [])) > 8 else "") + " s after its first message)"
                            if dr else "none dropped") + f". Old spacing median {r['old_spacing_ms']['median']} ms, "
                         f"10-90 % {r['old_spacing_ms']['p10']}-{r['old_spacing_ms']['p90']} ms, max "
                         f"{r['old_spacing_ms']['max']} ms. Stamps moved by median {sm['median']:+.1f} ms "
                         f"(range {sm['min']:+.1f} to {sm['max']:+.1f} ms).")
        else:
            lines.append(f"{r['series']}: NOT re-timed - {r['reason']}")
    return "\n".join(lines) if lines else "no IMU or depth topic to re-time"


def clock_report(data: BagData, patterns=RETIME_DEFAULT) -> str:
    """Which clock the camera, IMU and depth stamps are on, and any clock warnings."""
    fi = data.meta.get("frame_info", {})
    ck = data.meta.get("clock") or {}
    lines = [f"camera {data.camera_topic}: stamps from {fi.get('time_source', '?')}"]
    if ck.get("playback_recording"):
        lines.append(f"bag recorded from a playback at ~{1 / ck['receive_per_header_s']:.2f}x; "
                     f"all topics use their original header stamps (reference {short_name_(ck.get('reference', ''))})")
    names = []
    for s in data.series.values():
        if s.msgtype != "derived" and any(s.name.rstrip("/").endswith(p) for p in patterns):
            names.append(s.name)
            src = {"header": "header stamps, same clock as the camera",
                   "header-converted": "header stamps on a DIFFERENT clock from the camera's - mapped onto it "
                                       "through the recorder's receive times",
                   "receive": "no header - recorder receive time",
                   "receive-mapped": "no header - recorder receive time mapped onto the camera's clock",
                   "receive-corrected": "no header - receive time shifted back by a sibling topic's delay"
                   }.get(s.time_source, s.time_source)
            lines.append(f"{s.short}: {src}")
    for w in data.meta.get("warnings", []):
        if any(n in w for n in names) or "playback" in w:
            lines.append("  warning: " + w)
    return "\n".join(lines)


def short_name_(name: str) -> str:
    from bagcache import short_name
    return short_name(name) if name else ""


# --------------------------------------------------------------------------- #
# tag_to_imu_format: a compact, fixed-order CSV for tag -> IMU work
# --------------------------------------------------------------------------- #
TAG_IMU_FORMAT = "tag_to_imu_format"
TAG_IMU_IMU_FIELDS = ([f"linear_acceleration.{a}" for a in "xyz"] + [f"angular_velocity.{a}" for a in "xyz"]
                      + [f"orientation.{a}" for a in "xyzw"])
TAG_IMU_TF_FIELDS = [f"translation_mm.{a}" for a in "xyz"] + [f"rotation.{a}" for a in "xyzw"]
DEPTH_PREFER = ("bluerov/depth", "bluerov2/depth", "/depth")


def _tag_sort_key(child: str):
    m = re.search(r"(\d+)$", child)
    return (0, int(m.group(1)), child) if m else (1, 0, child)


def tag_to_imu_layout(data: BagData) -> tuple[list[tuple[str, str]], list[str]]:
    """
    Column order of the tag_to_imu_format CSV, after the dt column:
      1. every live tag's deviation from its zero frame, derived/tf[tag_N_zero->tag_N], sorted by
         tag number: translation_mm.x/y/z (mm, along the zero frame's axes) and rotation.x/y/z/w
         (quaternion; identity = on the zero pose). Tags without a zero frame are left out.
      2. the IMU (imu/data): linear_acceleration.x/y/z, angular_velocity.x/y/z, orientation.x/y/z/w
      3. the depth topic (bluerov/depth; falls back to global_position/rel_alt if there is none)
    Returns ([(series_name, column)], notes).
    """
    notes: list[str] = []
    cols: list[tuple[str, str]] = []

    # each live tag's deviation from its zero frame: derived/tf[tag_N_zero->tag_N]
    # (built by tf_tree_series; added here if the caller hasn't done it yet)
    devs = {s.info["deviation_of"]: s for s in data.series.values()
            if s.info.get("deviation_of") and all(c in s.cols for c in TAG_IMU_TF_FIELDS)}
    if not devs and data.n_frames:
        for s in tf_tree_series(data)[0]:
            data.series.setdefault(s.name, s)
            if s.info.get("deviation_of") and all(c in s.cols for c in TAG_IMU_TF_FIELDS):
                devs[s.info["deviation_of"]] = s
    live = set()
    for s in data.series.values():
        m = TF_EDGE_RE.search(s.name) if s.info.get("kind") == "tf" else None
        if m and _strip_static(m.group(2).strip("/")) == m.group(2).strip("/"):
            live.add(m.group(2).strip("/"))
    no_zero = sorted((live - set(devs)) & {f for f in live if "tag" in f.lower()}, key=_tag_sort_key)
    if not devs:
        notes.append("no tag_N_zero reference frames found, so no tag deviations")
    if no_zero:
        notes.append(f"left out (no zero frame): {', '.join(no_zero)}")
    for child in sorted(devs, key=_tag_sort_key):
        cols += [(devs[child].name, c) for c in TAG_IMU_TF_FIELDS]

    imu = _find_series(data, TAG_IMU_IMU_FIELDS, prefer=("/imu/data",))
    if imu is None:
        notes.append("no IMU topic with linear_acceleration, angular_velocity and orientation found")
    else:
        cols += [(imu.name, c) for c in TAG_IMU_IMU_FIELDS]
        notes.append(f"IMU: {imu.short}")

    recorded = [s for s in data.series.values() if s.msgtype != "derived" and s.info.get("kind") == "topic"]
    depth = None
    for p in DEPTH_PREFER:
        depth = next((s for s in recorded if p in s.name.lower()), None)
        if depth:
            break
    if depth is None:
        depth = next((s for s in recorded if "depth" in s.name.lower()), None)
    if depth is None:
        depth = next((s for s in recorded if s.name.endswith("rel_alt")), None)
        if depth is not None:
            notes.append(f"no bluerov/depth topic in this bag - used {depth.short} instead")
    if depth is None:
        notes.append("no depth topic found")
    else:
        dcols = [c for c in depth.cols if depth.kinds.get(c) == "float"]
        dcols = [c for c in dcols if c in ("data", "depth")] or dcols[:1]
        cols += [(depth.name, c) for c in dcols]
        notes.append(f"depth: {depth.short}.{', '.join(dcols)}")
    return cols, notes


def tag_to_imu_sync(data: BagData, start_s=None, stop_s=None, gap_factor: float = GAP_FACTOR,
                    rate: float | None = None) -> tuple[SyncResult, list[str]]:
    """Sync just the tag_to_imu_format columns and put them in that order."""
    wanted, notes = tag_to_imu_layout(data)
    if not wanted:
        raise RuntimeError("Nothing to export in tag_to_imu_format: " + "; ".join(notes))
    res = sync(data, start_s, stop_s, wanted, gap_factor, rate)
    by_name = {n: i for i, n in enumerate(res.names)}
    order = [by_name[f"{data.series[n].short}.{c}"] for n, c in wanted if f"{data.series[n].short}.{c}" in by_name]
    out = SyncResult(t=res.t, t0=res.t0, frame_idx=res.frame_idx, notes=res.notes)
    for i in order:
        out.names.append(res.names[i])
        out.kinds.append(res.kinds[i])
        out.modes.append(res.modes[i])
        out.values.append(res.values[i])
    return out, notes


def frame_dt(data: BagData, res: SyncResult, rate: float | None = None) -> np.ndarray:
    """Seconds between each row's camera frame and the frame before it in the bag (NaN for the
    bag's first frame). With fixed-rate rows: the row spacing."""
    if res.frame_idx is not None and data.n_frames:
        idx = np.asarray(res.frame_idx, int)
        dt = np.full(len(idx), np.nan)
        ok = idx > 0
        dt[ok] = (data.frame_t[idx[ok]] - data.frame_t[idx[ok] - 1]) / 1e9
        return dt
    dt = np.full(res.n_rows, np.nan)
    if res.n_rows:
        dt[1:] = np.diff(res.t) / 1e9
        dt[0] = 1.0 / rate if rate else np.nan
    return dt


def write_tag_to_imu_csv(data: BagData, res: SyncResult, path: str | Path, rate: float | None = None,
                         meta: dict | None = None, sidecar: bool = False) -> Path:
    """tag_to_imu_format CSV: dt_s first, then the tag tf columns, the IMU columns, the depth column."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dt = frame_dt(data, res, rate)
    cols = [["" if x != x else f"{x:.9f}" for x in dt.tolist()]]
    cols += [_fmt_col(v, k) for v, k in zip(res.values, res.kinds)]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dt_s"] + res.names)
        w.writerows(zip(*cols))
    if sidecar:
        side = {"csv": path.name, "format": TAG_IMU_FORMAT, "rows": res.n_rows,
                "first_timestamp": _fmt_ns(int(res.t[0])) if res.n_rows else None,
                "window_t_s": [float((res.t[0] - res.t0) / 1e9), float((res.t[-1] - res.t0) / 1e9)]
                if res.n_rows else None,
                "frame_idx_range": [int(res.frame_idx[0]), int(res.frame_idx[-1])]
                if res.frame_idx is not None and res.n_rows else None,
                "dt_s": "seconds since the previous camera frame in the bag (blank for the bag's first frame)",
                "columns": {n: m for n, m in zip(res.names, res.modes)}, "series": res.notes}
        if meta:
            side.update(meta)
        path.with_suffix(".sync.json").write_text(json.dumps(side, indent=1))
    return path


def export_images(data: BagData, res: SyncResult, out_dir: str | Path, progress=None) -> int:
    """Write the full-resolution frames of a sync window as PNGs named <frame_idx>_<timestamp>.png."""
    import cv2
    if res.frame_idx is None or res.n_rows == 0:
        return 0
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    i0, i1 = int(res.frame_idx[0]), int(res.frame_idx[-1])
    n = 0
    for k, t, img in data.iter_full_frames(i0, i1):
        cv2.imwrite(str(out / f"{k:06d}_{_fmt_ns(t)}.png"), img)
        n += 1
        if progress and n % 20 == 0:
            progress(n / (i1 - i0 + 1), f"Writing frame {n}/{i1 - i0 + 1}")
    return n


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cli_progress(frac, text):
    sys.stderr.write(f"\r{text} ({frac * 100:5.1f}%)   ")
    sys.stderr.flush()
    if frac >= 1.0:
        sys.stderr.write("\n")


# --------------------------------------------------------------------------- #
# IMU self-test: simulated runs where the true surge / sway are known
# --------------------------------------------------------------------------- #
def _euler_to_quat(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = (np.cos(roll / 2), np.sin(roll / 2), np.cos(pitch / 2), np.sin(pitch / 2),
                              np.cos(yaw / 2), np.sin(yaw / 2))
    return np.stack([sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy], 1)


def simulate_imu_run(turn_deg=0.0, waves_deg=0.0, bias=(0.0, 0.0, 0.0), noise=0.0, mount_yaw_deg=0.0,
                     roll_err_deg=0.0, pitch_err_per_ms2=0.0, heading0_deg=30.0, cruise=0.5, seed=0,
                     yaw_err_thrust_deg=0.0, thrust_err=(0.0, 0.0, 0.0), return_thrust=False,
                     roll_drift_deg=0.0, gyro_bias=(0.0, 0.0, 0.0), gyro_noise=0.0, return_gyro=False,
                     pitch_osc_deg=0.0, waves_from_start=False, rock_start_deg=0.0):
    """
    A 30 s run: still 3 s, speed up to `cruise` m/s straight along the hull's nose, cruise, slow
    down, still 3 s. True sway is 0 at every moment. Options: a turn of turn_deg during the
    cruise, roll/pitch wave motion, a body-frame accelerometer bias, white noise (m/s²), an IMU
    mounted turned mount_yaw_deg to the left of the hull's nose, a constant roll error in the
    reported attitude, and a pitch error proportional to forward acceleration (an AHRS being
    pulled by acceleration), and a compass error: the reported yaw is off by yaw_err_thrust_deg
    while the thrusters run (magnetic disturbance), although the true heading never changes, and
    an accelerometer error thrust_err (m/s², body) that scales with the forward thruster
    command (thrust ~ 150 µs at cruise). Returns (t_ns, q_imu, f_imu, true_surge, true_sway) at 100 Hz.
    """
    rng = np.random.default_rng(seed)
    fs_hi, T = 1000, 30.0
    t = np.arange(0, T, 1 / fs_hi)
    def smooth(x0, x1, a, b):                       # 0 -> 1 smoothstep between a and b
        z = np.clip((t - a) / (b - a), 0, 1)
        return x0 + (x1 - x0) * z * z * (3 - 2 * z)
    u = smooth(0, cruise, 3, 7) - smooth(0, cruise, 23, 27)
    yaw = np.radians(heading0_deg) + np.radians(turn_deg) * (smooth(0, 1, 9, 21))
    wv = 1.0 if waves_from_start else smooth(0, 1, 1.5, 2.5)       # still during the Start rest
    rock = np.radians(rock_start_deg) * np.sin(2 * np.pi * 0.7 * t) * (1 - smooth(0, 1, 1.2, 2.4))  # Start only
    roll = np.radians(waves_deg) * np.sin(2 * np.pi * 0.3 * t) * wv + rock
    pitch = np.radians(waves_deg) * 0.7 * np.sin(2 * np.pi * 0.21 * t + 1.0) * wv \
        + np.radians(pitch_osc_deg) * np.sin(2 * np.pi * 0.5 * t) * smooth(0, 1, 4, 6)
    q_h = _euler_to_quat(roll, pitch, yaw)
    R_h = quat_to_mat(q_h)
    v_w = R_h[:, :, 0] * u[:, None]                 # moving along the hull nose: zero sway
    a_w = np.gradient(v_w, t, axis=0)
    d = np.radians(mount_yaw_deg)
    Rz = np.array([[np.cos(d), -np.sin(d), 0], [np.sin(d), np.cos(d), 0], [0, 0, 1.0]])
    R_i = R_h @ Rz                                  # IMU frame turned by d about z
    f = np.einsum("nji,nj->ni", R_i, a_w + np.array([0, 0, G])) + np.array(bias)
    # attitude the IMU reports (with errors)
    acc_fwd = np.gradient(u, t)
    # forward thruster command: drag (~ speed²) + acceleration, ~150 µs at cruise
    thr = 150.0 * (u / cruise) ** 2 + 400.0 * acc_fwd if cruise else 0 * u
    f = f + np.outer(thr / 150.0, np.array(thrust_err, float))
    yaw_err = np.radians(yaw_err_thrust_deg) * np.clip(np.abs(thr) / 150.0, 0, 1.5)
    Rze = quat_to_mat(_euler_to_quat(np.zeros(len(t)), np.zeros(len(t)), yaw_err))
    roll_e = np.radians(roll_err_deg) + np.radians(roll_drift_deg) * t / T      # constant + drifting
    rep = Rze @ R_i @ quat_to_mat(_euler_to_quat(roll_e,
                                                 np.radians(pitch_err_per_ms2) * acc_fwd, np.zeros(len(t))))
    # back to quaternions (from rotation matrices)
    def mat_to_quat(M):
        w = np.sqrt(np.maximum(0, 1 + M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2])) / 2
        x = np.copysign(np.sqrt(np.maximum(0, 1 + M[:, 0, 0] - M[:, 1, 1] - M[:, 2, 2])) / 2, M[:, 2, 1] - M[:, 1, 2])
        y = np.copysign(np.sqrt(np.maximum(0, 1 - M[:, 0, 0] + M[:, 1, 1] - M[:, 2, 2])) / 2, M[:, 0, 2] - M[:, 2, 0])
        z = np.copysign(np.sqrt(np.maximum(0, 1 - M[:, 0, 0] - M[:, 1, 1] + M[:, 2, 2])) / 2, M[:, 1, 0] - M[:, 0, 1])
        return np.stack([x, y, z, w], 1)
    q_rep = mat_to_quat(rep)
    # gyro: the IMU frame's true body rates, w = vee(R^T dR/dt), plus bias and noise
    dR = np.gradient(R_i, t, axis=0)
    W = np.einsum("nji,njk->nik", R_i, dR)
    gyro = np.stack([W[:, 2, 1], W[:, 0, 2], W[:, 1, 0]], 1) + np.array(gyro_bias, float)
    k = slice(None, None, 10)                       # 100 Hz
    f = f[k] + rng.normal(0, noise, f[k].shape) if noise else f[k]
    gyro = gyro[k] + (rng.normal(0, gyro_noise, gyro[k].shape) if gyro_noise else 0.0)
    t_ns = (1_700_000_000.0 * 1e9 + t[k] * 1e9).astype(np.int64)
    out = (t_ns, q_rep[k], f, u[k], np.zeros(len(t_ns)))
    if return_thrust:
        out = out + (thr[k],)
    return out + (gyro,) if return_gyro else out


class _SimBag:
    """Just enough of BagData for integrate_imu."""
    def __init__(self, t_ns, q, f, thr=None, w=None):
        cols = {f"orientation.{a}": q[:, i] for i, a in enumerate("xyzw")}
        cols.update({f"linear_acceleration.{a}": f[:, i] for i, a in enumerate("xyz")})
        if w is not None:
            cols.update({f"angular_velocity.{a}": w[:, i] for i, a in enumerate("xyz")})
        self.series = {"/mavros/imu/data": Series("/mavros/imu/data", "mavros/imu/data", "sensor_msgs/msg/Imu",
                                                  t_ns, cols, {c: "float" for c in cols})}
        if thr is not None:
            self.series[DERIVED_THR] = Series(DERIVED_THR, "derived/thrusters", "derived", t_ns,
                                              {"ch0_us": thr}, {"ch0_us": "float"},
                                              "header", {}, {"kind": "derived", "channels": [0]})
        self.t0 = int(t_ns[0])
        self.meta = {}

    def abs_ns(self, rel_s):
        return int(round(self.t0 + rel_s * 1e9))


IMU_TEST_CASES = [
    ("straight, level, no errors", {}),
    ("pitching ±25° (0.5 Hz) while cruising", dict(pitch_osc_deg=25)),
    ("waves ±5° roll/pitch (still at Start)", dict(waves_deg=5)),
    ("rocking ±5° DURING the Start rest", dict(waves_deg=5, waves_from_start=True)),
    ("90° turn", dict(turn_deg=90)),
    ("autopilot roll off by 3.5° (not used now)", dict(roll_err_deg=3.5)),
    ("autopilot compass swings 30° (not used now)", dict(yaw_err_thrust_deg=30)),
    ("gyro offset 0.01 rad/s, still Start (measured)", dict(gyro_bias=(0.01, -0.008, 0.005))),
    ("gyro offset 0.002 rad/s, Start rocking ±3°, still Stop", dict(gyro_bias=(0.002, -0.0015, 0.001), rock_start_deg=3)),
    ("accel bias 0.04 m/s² + noise", dict(bias=(0.04, -0.03, 0.05), noise=0.02)),
    ("accel bias + noise + pitching ±25° + gyro noise", dict(bias=(0.04, -0.03, 0.05), noise=0.02,
                                                            pitch_osc_deg=25, gyro_noise=0.002)),
    ("IMU mounted 8° left of the nose", dict(mount_yaw_deg=8.0)),
    ("-0.2 m/s² surge error while thrusting", dict(thrust_err=(-0.2, 0.0, 0))),
    ("+0.25 m/s² SWAY error while thrusting", dict(thrust_err=(0.0, 0.25, 0))),
    ("yaw gyro offset 0.003 rad/s, Start rocking", dict(gyro_bias=(0, 0, 0.003), rock_start_deg=3)),
]


def imu_selftest(fn=None, cases=IMU_TEST_CASES, verbose=True):
    """Run integrate_imu (or fn with the same signature) on each simulated case; true sway = 0.
    A case is (name, simulation options[, integrate_imu options])."""
    fn = fn or integrate_imu
    rows = []
    for case in cases:
        name, kw = case[0], case[1]
        fit = case[2] if len(case) > 2 else {}
        t_ns, q, f, u, _, thr, gy = simulate_imu_run(**kw, return_thrust=True, return_gyro=True)
        bag = _SimBag(t_ns, q, f, thr, gy)
        ser = fn(bag, 0.0, (t_ns[-1] - t_ns[0]) / 1e9, "rest", **fit)
        surge, sway = ser.cols["surge"], ser.cols["sway"]
        true_dist = float(np.sum(0.5 * (u[1:] + u[:-1]) * np.diff(t_ns) / 1e9))
        dist = float(np.sum(0.5 * (surge[1:] + surge[:-1]) * np.diff(t_ns) / 1e9))
        row = {"case": name, "surge_rms_err": float(np.sqrt(np.mean((surge - u) ** 2))),
               "sway_peak": float(np.max(np.abs(sway))), "dist_err_pct": 100 * (dist - true_dist) / true_dist,
               "surge_min": float(np.min(surge)), "travel_angle": ser.info.get("travel_angle_deg"),
               "arch": (ser.info.get("rest_check") or {}).get("arch_estimate_m_s"),
               "still": ser.info.get("start_still")}
        if "mounted" in name and ser.info.get("travel_angle_deg") is not None:
            fixed = fn(bag, 0.0, (t_ns[-1] - t_ns[0]) / 1e9, "rest", yaw_offset_deg=ser.info["travel_angle_deg"])
            row["sway_peak_after_offset"] = float(np.max(np.abs(fixed.cols["sway"])))
        rows.append(row)
    if verbose:
        print(f"{'simulated case (true sway = 0, surge >= 0)':44s} {'surge RMS err':>13s} {'min surge':>10s} "
              f"{'peak |sway|':>11s} {'distance err':>12s} {'travel ang':>10s} {'rest-check arch':>15s}")
        for r in rows:
            ta = "" if r["travel_angle"] is None else f"{r['travel_angle']:+.1f}°"
            sw = "" if r["arch"] is None else f"{r['arch']:.3f} m/s"
            print(f"{r['case']:44s} {r['surge_rms_err']:9.3f} m/s {r['surge_min']:+7.3f} {r['sway_peak']:8.3f} m/s "
                  f"{r['dist_err_pct']:+10.1f} % {ta:>10s} {sw:>15s}")
            if r.get("still") is False:
                print("   ^ flagged: Start was not still, so the gyro offset wasn't measured there")
            if "sway_peak_after_offset" in r:
                print(f"{'   ... with yaw offset = travel angle':44s} {'':13s} {'':10s} {r['sway_peak_after_offset']:8.3f} m/s")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", nargs="?", help="bag folder (with metadata.yaml) or the .mcap/.db3 inside it")
    ap.add_argument("--info", action="store_true", help="print topics, rates, time sources, the tf tree and exit")
    ap.add_argument("--start", type=float, help="window start, seconds since first camera frame")
    ap.add_argument("--stop", type=float, help="window stop, seconds since first camera frame")
    ap.add_argument("--out", "-o", help="output CSV (default: <bag>_synced.csv next to the bag)")
    ap.add_argument("--topics", nargs="*", help="only series whose name contains any of these substrings")
    ap.add_argument("--rate", type=float, help="use a fixed output rate (Hz) instead of camera frames")
    ap.add_argument("--gap-factor", type=float, default=GAP_FACTOR,
                    help="blank values inside gaps longer than this x the topic's sample period (default 3)")
    ap.add_argument("--camera", help="image topic to sync to (default: first */image_raw)")
    ap.add_argument("--msg-dir", help="folder of .msg files for custom types not embedded in the bag")
    ap.add_argument("--include-covariance", action="store_true", help="also export covariance arrays")
    ap.add_argument("--export-images", metavar="DIR", help="also write the window's full-res frames as PNG")
    ap.add_argument("--integrate-imu", nargs="?", const="rest", choices=IMU_MODES,
                    help="add IMU-integrated velocity over the window (columns derived/imu_vel.*). "
                         "'rest' (default): ROV still at --start and --stop. "
                         "'odom': anchor to global_position/local velocity (needs GPS/DVL)")
    ap.add_argument("--imu-yaw-offset", type=float, default=0.0, metavar="DEG",
                    help="IMU mounted this many degrees left of the hull's nose (turns surge/sway axes)")
    ap.add_argument("--thrusters", default=",".join(map(str, HORIZONTAL_CHANNELS)),
                    help="rc/out channels of the horizontal thrusters for derived/forward_motion_diag (default 0,1,2,3)")
    ap.add_argument("--tf", nargs="*", metavar="A:B", default=[],
                    help="also add the measured pose of frame B in frame A (and B in A's reverse), e.g. tag_3:tag_7")
    ap.add_argument("--tag-px", dest="tag_px", nargs="*", metavar="TF",
                    help="add unfiltered pixel-centre x/y (translation.x_px / y_px, full bandwidth) to tf "
                         "series containing these words (e.g. tag_18). No smoothing is applied.")
    ap.add_argument("--format", choices=("standard", TAG_IMU_FORMAT), default="standard",
                    help=f"'{TAG_IMU_FORMAT}': dt between camera frames, then each tag's tf (translation + "
                         "quaternion), the IMU (linear_acceleration, angular_velocity, orientation), and the "
                         "depth topic last. Ignores --topics and --integrate-imu.")
    ap.add_argument("--retime", action="store_true",
                    help="give the IMU and depth topics evenly spaced stamps before syncing (for topics stamped "
                         "on arrival, e.g. MAVROS with timesync off). Dropped messages are located and left as "
                         "empty slots; if the timing can't be lined up, that topic is left as recorded.")
    ap.add_argument("--sidecar", action="store_true", help="also write <csv>.sync.json with the settings used")
    ap.add_argument("--rebuild", action="store_true", help="ignore the cache and re-read the bag")
    ap.add_argument("--selftest-imu", action="store_true",
                    help="check the IMU velocity maths on simulated runs (true sway = 0) and exit")
    a = ap.parse_args(argv)

    if a.selftest_imu:
        imu_selftest()
        return 0
    if not a.bag:
        ap.error("the bag folder is required")

    t_load = time.time()
    data = load_bag(a.bag, camera_topic=a.camera, progress=_cli_progress, rebuild=a.rebuild,
                    msg_dir=a.msg_dir, include_covariance=a.include_covariance)
    print(f"Loaded in {time.time() - t_load:.1f} s")
    print("Clocks:\n  " + clock_report(data).replace("\n", "\n  "))
    retime_reports = []
    if a.retime:
        retime_reports = retime_series(data)
        print("Even re-timing:\n  " + retime_text(retime_reports).replace("\n", "\n  "))
    tf_ser, tf_notes = tf_tree_series(data)
    for s_ in tf_ser:
        data.series[s_.name] = s_
    for pair in a.tf:
        if ":" not in pair:
            print(f"--tf {pair}: use FRAME_A:FRAME_B")
            continue
        fa, fb = pair.split(":", 1)
        for x, y in ((fa, fb), (fb, fa)):
            s_ = tf_pair_series(data, x, y)
            if s_ is None:
                print(f"--tf {x} -> {y}: no measured path through /tf. Frames: {', '.join(tf_frames(data))}")
            else:
                data.series[s_.name] = s_
    if a.info:
        print(bag_summary(data))
        print("\n" + tf_tree_text(data))
        for n in tf_notes:
            print("  " + n)
        return 0

    for sub in a.tag_px or []:
        for ser in data.series.values():
            if ser.info.get("kind") == "tf" and sub in ser.name:
                info = tag_pixel_position(data, ser)
                print(f"Pixel-centre x/y for {ser.short}: "
                      + (f"{info['matched']}/{info['of']} frames, {info['intrinsics']}" if info
                         else "no matching detections or camera_info; skipped"))
    hdg = heading_series(data)
    if hdg is not None:
        data.series[hdg.name] = hdg
    thr = thruster_series(data, [int(x) for x in a.thrusters.split(",") if x.strip()])
    if thr is not None:
        data.series[thr.name] = thr
    if a.integrate_imu:
        lo = a.start if a.start is not None else 0.0
        hi = a.stop if a.stop is not None else float(data.rel(data.frame_t[-1])) if data.n_frames else 1e9
        s_int = integrate_imu(data, lo, hi, a.integrate_imu, yaw_offset_deg=a.imu_yaw_offset)
        if s_int is None:
            print("Could not integrate the IMU (no IMU with orientation + linear_acceleration in the window,"
                  " or no odometry for 'odom' mode); skipping.")
        else:
            data.series[s_int.name] = s_int
            print(imu_summary(s_int.info))

    if a.format == TAG_IMU_FORMAT:
        res, notes = tag_to_imu_sync(data, a.start, a.stop, a.gap_factor, a.rate)
        if res.n_rows == 0:
            print("No camera frames inside that window.")
            return 1
        out = Path(a.out) if a.out else data.bag_dir.parent / f"{data.bag_dir.name}_{TAG_IMU_FORMAT}.csv"
        write_tag_to_imu_csv(data, res, out, a.rate, {"bag": str(data.bag_dir), "camera_topic": data.camera_topic,
                                                      "gap_factor": a.gap_factor, "notes": notes,
                                                      "retimed": retime_reports},
                             sidecar=a.sidecar)
        print(f"Wrote {out}  ({res.n_rows} rows x {len(res.names) + 1} columns, {TAG_IMU_FORMAT})")
        for n in notes:
            print("  " + n)
        if a.export_images:
            n = export_images(data, res, a.export_images, _cli_progress)
            print(f"Wrote {n} frames to {a.export_images}")
        return 0

    columns = None
    if a.topics:
        a.topics = list(a.topics) + (["derived/imu_vel"] if a.integrate_imu else []) + \
            ["derived/forward_motion_diag", "derived/heading"]
        columns = [(n, c) for n, s in data.series.items()
                   if any(sub in n or sub in s.short for sub in a.topics) for c in s.cols]
        if not columns:
            print(f"No series match {a.topics}. Use --info to list them.")
            return 1

    res = sync(data, a.start, a.stop, columns, a.gap_factor, a.rate)
    if res.n_rows == 0:
        print("No camera frames inside that window.")
        return 1
    out = Path(a.out) if a.out else data.bag_dir.parent / f"{data.bag_dir.name}_synced.csv"
    write_csv(res, out, {"bag": str(data.bag_dir), "camera_topic": data.camera_topic,
                         "gap_factor": a.gap_factor, "fixed_rate_hz": a.rate,
                         "warnings": data.meta.get("warnings", []), "retimed": retime_reports},
              sidecar=a.sidecar)
    blanks = sum(int(np.sum(v != v)) if v.dtype != object else 0 for v in res.values)
    print(f"Wrote {out}  ({res.n_rows} rows x {len(res.names) + 3} columns, "
          f"t_s {(res.t[0] - res.t0) / 1e9:.3f} -> {(res.t[-1] - res.t0) / 1e9:.3f}, "
          f"{blanks} blank cells)")
    if a.export_images:
        n = export_images(data, res, a.export_images, _cli_progress)
        print(f"Wrote {n} frames to {a.export_images}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
