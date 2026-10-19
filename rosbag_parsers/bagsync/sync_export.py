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
    if start_s is not None:
        m &= ft >= data.abs_ns(start_s) - 500
    if stop_s is not None:
        m &= ft <= data.abs_ns(stop_s) + 500
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
        locked = rate is None and s.info.get("kind") != "derived" and is_camera_locked(s, data.frame_t, tol)
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
# Thruster activity (rc/out) - shown for reference when choosing Start and Stop
# --------------------------------------------------------------------------- #
# BlueROV2 (standard and Heavy): outputs 1-4 are the horizontal thrusters, 5+ vertical.
THRUSTER_CHANNELS = (0, 1, 2, 3)
DERIVED_THR = "/derived/thrusters"


def thruster_series(data: BagData, channels=THRUSTER_CHANNELS) -> Series | None:
    """max |pwm - 1500| over the chosen rc/out channels (0 = thrusters off), on rc/out's own times."""
    rc = _find_series(data, ["channels[0]"], prefer=("rc/out",))
    if rc is None or "rc/out" not in rc.name:
        return None
    use = [c for c in channels if f"channels[{c}]" in rc.cols]
    if not use:
        return None
    pwm = np.stack([rc.cols[f"channels[{c}]"] for c in use], axis=1)
    pwm = np.where((pwm >= 800) & (pwm <= 2200), pwm, 1500.0)    # 0 = unused output
    dev = np.max(np.abs(pwm - 1500.0), axis=1)
    return Series(DERIVED_THR, "derived/thrusters", "derived", rc.t, {"max_dev_us": dev},
                  {"max_dev_us": "float"}, rc.time_source, {},
                  {"kind": "derived", "channels": use, "source": rc.name})


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


def _yaw_enu(q):
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def integrate_imu(data: BagData, start_s: float, stop_s: float, mode: str = "rest",
                  imu: str | None = None, odom: str | None = None) -> Series | None:
    """
    Velocity from the IMU, integrated from start_s to stop_s (seconds since first frame).

    linear_acceleration is the specific force in the body frame (it reads +9.81 on z at rest).
    It is rotated into the ENU world frame with the IMU's orientation, gravity is removed, and
    the result is integrated (trapezoid) on the IMU's own samples.

    mode "rest" (default, no GPS/DVL): you place Start and Stop where the vehicle is still.
        Velocity starts at 0, and whatever velocity is left at Stop is treated as accelerometer
        bias and removed linearly, so it ends at 0 too.
    mode "odom": starts from, and is corrected to, the odometry velocity (needs GPS/DVL).

    Columns:
      forward_speed            speed along the ROV's heading, horizontal (m/s, + = ahead)
      sideways_speed           horizontal, across the heading (m/s, + = to the left)
      vertical_speed           up/down (m/s, + = up)
      speed                    total horizontal speed (m/s, always >= 0)
      distance                 horizontal path length since Start (m)
      forward_distance         distance along the heading since Start (m, backwards subtracts)
      forward_speed_raw        forward_speed before the drift correction
      acc_forward / acc_sideways / acc_vertical   gravity-free acceleration, same axes (m/s^2)
      surge, sway, heave       velocity along the ROV's own body axes, which tilt with it
                               (forward along the nose, + left, + up through the top) (m/s);
                               the frame the water and anything mounted on the ROV "feel"
      surge_raw, ..., acc_surge, ...   before drift correction / gravity-free acceleration, body axes
      vel.e/n/u, disp.e/n/u, acc_world.e/n/u      the same in the world frame (east, north, up)
    """
    if mode not in IMU_MODES:
        raise ValueError(f"mode must be one of {IMU_MODES}")
    s_imu = data.series.get(imu) if imu else _find_series(
        data, ["orientation.w", "linear_acceleration.x"], prefer=("/imu/data",))
    if s_imu is None:
        return None
    t_lo, t_hi = data.abs_ns(start_s), data.abs_ns(stop_s)
    m = (s_imu.t >= t_lo) & (s_imu.t <= t_hi)
    if m.sum() < 3:
        return None
    t = s_imu.t[m]
    q = np.stack([s_imu.cols[f"orientation.{a}"][m] for a in "xyzw"], axis=1)
    acc = np.stack([s_imu.cols[f"linear_acceleration.{a}"][m] for a in "xyz"], axis=1)
    good = np.isfinite(q).all(1) & np.isfinite(acc).all(1) & (np.linalg.norm(q, axis=1) > 0.5)
    t, q, acc = t[good], q[good], acc[good]
    if len(t) < 3:
        return None
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    a_w = _rotate(q, acc) - np.array([0.0, 0.0, G])

    tsec = (t - t[0]) / 1e9
    dt = np.diff(tsec)[:, None]
    vel_raw = np.zeros_like(a_w)
    vel_raw[1:] = np.cumsum(0.5 * (a_w[1:] + a_w[:-1]) * dt, axis=0)

    info = {"kind": "derived", "mode": mode, "imu": s_imu.name, "gravity": G,
            "window_t_s": [start_s, stop_s], "duration_s": round(float(tsec[-1]), 3)}
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
    vel = vel_raw - err[None, :] * (tsec / tsec[-1])[:, None] if tsec[-1] > 0 else vel_raw.copy()
    info["end_error_before_correction"] = err.round(4).tolist()
    info["implied_accel_bias"] = (err / max(tsec[-1], 1e-9)).round(5).tolist()

    yaw = _yaw_enu(q)
    c, s_ = np.cos(yaw), np.sin(yaw)

    def heading_frame(v):
        return v[:, 0] * c + v[:, 1] * s_, -v[:, 0] * s_ + v[:, 1] * c, v[:, 2]

    fwd, side, vert = heading_frame(vel)
    qc = q * np.array([-1.0, -1.0, -1.0, 1.0])            # world -> body (FLU, like mavros base_link)
    body, body_raw, acc_body = _rotate(qc, vel), _rotate(qc, vel_raw), _rotate(qc, a_w)
    fwd_raw, _, _ = heading_frame(vel_raw)
    afwd, aside, avert = heading_frame(a_w)
    disp = np.zeros_like(vel)
    disp[1:] = np.cumsum(0.5 * (vel[1:] + vel[:-1]) * dt, axis=0)
    speed = np.hypot(vel[:, 0], vel[:, 1])
    dist = np.zeros(len(t))
    dist[1:] = np.cumsum(0.5 * (speed[1:] + speed[:-1]) * dt[:, 0])
    fdist = np.zeros(len(t))
    fdist[1:] = np.cumsum(0.5 * (fwd[1:] + fwd[:-1]) * dt[:, 0])
    info["distance_m"] = round(float(dist[-1]), 3)
    info["forward_distance_m"] = round(float(fdist[-1]), 3)
    info["peak_forward_m_s"] = round(float(fwd[np.argmax(np.abs(fwd))]), 3)
    info["mean_forward_m_s"] = round(float(np.mean(fwd)), 3)

    cols = {"forward_speed": fwd, "sideways_speed": side, "vertical_speed": vert, "speed": speed,
            "distance": dist, "forward_distance": fdist, "forward_speed_raw": fwd_raw,
            "acc_forward": afwd, "acc_sideways": aside, "acc_vertical": avert}
    for i, a in enumerate(("surge", "sway", "heave")):
        cols[a] = body[:, i]
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
    return Series(DERIVED_IMU, "derived/imu_vel", "derived", t, cols,
                  {k: "float" for k in cols}, s_imu.time_source, {}, info)


def imu_summary(info: dict) -> str:
    """One-line description of an integration result."""
    err = np.round(info.get("end_error_before_correction", []), 3).tolist()
    return (f"IMU velocity {info['window_t_s'][0]:.2f}–{info['window_t_s'][1]:.2f} s ({info['mode']} mode): "
            f"forward distance {info.get('forward_distance_m', 0):.2f} m, "
            f"peak forward speed {info.get('peak_forward_m_s', 0):.2f} m/s, drift removed {err} m/s")


# --------------------------------------------------------------------------- #
# Kalman smoothing of tag positions (tf translation)
# --------------------------------------------------------------------------- #
KF_Q = None     # process noise (m/s^2); None = choose automatically per axis (maximum likelihood)
KF_GATE = 5.0   # measurements more than this many sigmas from the prediction are treated as glitches
REFERENCE_TAGS = (0, 20)   # tags fixed relative to the camera: used as a noise reference, not smoothed


def _robust_noise(z: np.ndarray) -> float:
    """Measurement noise (std) from the jitter around a straight line through neighbours."""
    z = z[np.isfinite(z)]
    if len(z) < 5:
        return 1e-3
    d = z[1:-1] - 0.5 * (z[:-2] + z[2:])
    mad = np.median(np.abs(d - np.median(d)))
    return max(1.4826 * mad / np.sqrt(1.5), 1e-6)


def _kf_cv(t: np.ndarray, z: np.ndarray, q: float, r: float, smooth: bool, want_ll: bool = False):
    """
    Constant-velocity Kalman filter (+ Rauch-Tung-Striebel smoother) for one axis.
    Written with scalar maths (2x2 state) so it is fast enough to run many times for auto-tuning.
    Returns (position, velocity) or, with want_ll, the log-likelihood of the measurements.
    """
    n = len(t)
    good = np.isfinite(z)
    if not good.any():
        nan = np.full(n, np.nan)
        return -np.inf if want_ll else (nan, nan)
    k0 = int(np.argmax(good))
    R = r * r
    q2 = q * q
    tl, zl, gl = t.tolist(), z.tolist(), good.tolist()
    # filtered state (x0 pos, x1 vel), covariance (a b; b c), predicted ones for the smoother
    X0 = [np.nan] * n; X1 = [np.nan] * n
    Pa = [0.0] * n; Pb = [0.0] * n; Pc = [0.0] * n
    pX0 = [0.0] * n; pX1 = [0.0] * n; pPa = [0.0] * n; pPb = [0.0] * n; pPc = [0.0] * n; DT = [0.0] * n
    x0, x1 = zl[k0], 0.0
    a, b, c = R, 0.0, 1.0
    ll = 0.0
    for k in range(k0, n):
        dt = tl[k] - tl[k - 1] if k > k0 else 0.0
        # predict: F = [[1, dt], [0, 1]], Q = q^2 [[dt^4/4, dt^3/2], [dt^3/2, dt^2]]
        x0 = x0 + dt * x1
        a, b, c = (a + 2 * dt * b + dt * dt * c + q2 * dt ** 4 / 4,
                   b + dt * c + q2 * dt ** 3 / 2,
                   c + q2 * dt * dt)
        pX0[k], pX1[k], pPa[k], pPb[k], pPc[k], DT[k] = x0, x1, a, b, c, dt
        if gl[k]:
            S = a + R
            nu = zl[k] - x0
            if k > k0 + 10 and nu * nu > KF_GATE * KF_GATE * S:
                X0[k], X1[k], Pa[k], Pb[k], Pc[k] = x0, x1, a, b, c   # glitch: skip this measurement
                continue
            if want_ll and k > k0:
                ll -= 0.5 * (np.log(2 * np.pi * S) + nu * nu / S)
            K0, K1 = a / S, b / S
            x0, x1 = x0 + K0 * nu, x1 + K1 * nu
            a, b, c = a - K0 * a, b - K0 * b, c - K1 * b
        X0[k], X1[k], Pa[k], Pb[k], Pc[k] = x0, x1, a, b, c
    if want_ll:
        return ll
    if smooth:
        for k in range(n - 2, k0 - 1, -1):
            dt = DT[k + 1]
            # C = P_k F^T inv(Pp_{k+1})
            fa, fb, fc = pPa[k + 1], pPb[k + 1], pPc[k + 1]
            det = fa * fc - fb * fb
            if det <= 0:
                continue
            ia, ib, ic = fc / det, -fb / det, fa / det
            m00, m01 = Pa[k] + dt * Pb[k], Pb[k]          # P_k F^T
            m10, m11 = Pb[k] + dt * Pc[k], Pc[k]
            C00, C01 = m00 * ia + m01 * ib, m00 * ib + m01 * ic
            C10, C11 = m10 * ia + m11 * ib, m10 * ib + m11 * ic
            d0, d1 = X0[k + 1] - pX0[k + 1], X1[k + 1] - pX1[k + 1]
            X0[k] += C00 * d0 + C01 * d1
            X1[k] += C10 * d0 + C11 * d1
            # covariance (needed for the next step back)
            e00, e01, e11 = Pa[k + 1] - fa, Pb[k + 1] - fb, Pc[k + 1] - fc
            Pa[k] += C00 * (C00 * e00 + C01 * e01) + C01 * (C00 * e01 + C01 * e11)
            Pb[k] += C00 * (C10 * e00 + C11 * e01) + C01 * (C10 * e01 + C11 * e11)
            Pc[k] += C10 * (C10 * e00 + C11 * e01) + C11 * (C10 * e01 + C11 * e11)
    return np.array(X0), np.array(X1)


def auto_q(t: np.ndarray, z: np.ndarray, r: float) -> float:
    """Process noise that makes the measurements most likely (coarse grid, then refine)."""
    grid = np.logspace(-6, 1, 22)
    lls = [_kf_cv(t, z, q, r, False, want_ll=True) for q in grid]
    i = int(np.argmax(lls))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    fine = np.logspace(np.log10(lo), np.log10(hi), 9)
    lls = [_kf_cv(t, z, q, r, False, want_ll=True) for q in fine]
    return float(fine[int(np.argmax(lls))])


def _kf_names(src: str) -> tuple[str, str]:
    """'translation.x' -> ('translation.x_kf', 'velocity_kf.x'); 'translation.x_px' -> (..._px_kf, velocity_px_kf.x)."""
    base, axis = src.rsplit(".", 1)
    tag = "_px" if axis.endswith("_px") else ""
    axis = axis.replace("_px", "")
    return f"{src}_kf", f"velocity{tag}_kf.{axis}"


def kalman_smooth_tf(ser: Series, q: float | None = None, smooth: bool = True, axes: str = "xyz",
                     include_px: bool = True) -> dict:
    """
    Smooth a tf series' translation with a constant-velocity Kalman filter and add the result
    as new columns of that same series (metres, camera optical frame: x right, y down, z forward):
      translation.x_kf / y_kf / z_kf      smoothed pose position (m)
      translation.x_px_kf / y_px_kf       smoothed pixel-centre position (m), if tag_pixel_position ran
      velocity_kf.* / velocity_px_kf.*    the filter's velocity estimates (m/s)
    Measurement noise is estimated from the data. q (m/s^2) sets how much motion is allowed;
    q=None picks it per axis by maximum likelihood (auto). smooth=True also runs the backward
    (RTS) pass: no lag, uses future samples (post-processing only).
    """
    t = (ser.t - ser.t[0]) / 1e9
    info = {"smoother": "RTS (zero lag)" if smooth else "forward only", "noise_std_m": {}, "q_m_s2": {}}
    calc = ser.info.setdefault("calc_cols", [])
    srcs = [f"translation.{a}" for a in axes]
    if include_px:
        srcs += [f"translation.{a}_px" for a in "xy"]
    for src in srcs:
        if src not in ser.cols:
            continue
        z = ser.cols[src].astype(float)
        r = _robust_noise(z)
        qa = auto_q(t, z, r) if q is None else q
        pos, vel = _kf_cv(t, z, qa, r, smooth)
        for name, arr in zip(_kf_names(src), (pos, vel)):
            ser.cols[name] = arr
            ser.kinds[name] = "float"
            if name not in calc:
                calc.append(name)
        key = src.replace("translation.", "")
        info["noise_std_m"][key] = float(f"{r:.3g}")
        info["q_m_s2"][key] = float(f"{qa:.3g}")
    ser.info["kalman"] = info
    return info


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", help="bag folder (with metadata.yaml) or the .mcap/.db3 inside it")
    ap.add_argument("--info", action="store_true", help="print topics, rates, time sources and exit")
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
    ap.add_argument("--thrusters", default=",".join(map(str, THRUSTER_CHANNELS)),
                    help="rc/out channel indexes of the horizontal thrusters (default 0,1,2,3)")
    ap.add_argument("--tag-px", "--kalman", dest="tag_px", nargs="*", metavar="TF",
                    help="add unfiltered pixel-centre x/y (translation.x_px / y_px, full bandwidth) to tf "
                         "series containing these words (e.g. tag_18). No smoothing is applied.")
    ap.add_argument("--panels", action="store_true",
                    help="add panel signals (open/close, shifts; measured, _imu, _leftover) as derived/panels.*")
    ap.add_argument("--sidecar", action="store_true", help="also write <csv>.sync.json with the settings used")
    ap.add_argument("--rebuild", action="store_true", help="ignore the cache and re-read the bag")
    a = ap.parse_args(argv)

    t_load = time.time()
    data = load_bag(a.bag, camera_topic=a.camera, progress=_cli_progress, rebuild=a.rebuild,
                    msg_dir=a.msg_dir, include_covariance=a.include_covariance)
    print(f"Loaded in {time.time() - t_load:.1f} s")
    if a.info:
        print(bag_summary(data))
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
        s_int = integrate_imu(data, lo, hi, a.integrate_imu)
        if s_int is None:
            print("Could not integrate the IMU (no IMU with orientation + linear_acceleration in the window,"
                  " or no odometry for 'odom' mode); skipping.")
        else:
            data.series[s_int.name] = s_int
            print(imu_summary(s_int.info))

    if a.panels:
        from panels import panel_analysis
        lo = a.start if a.start is not None else 0.0
        hi = a.stop if a.stop is not None else float(data.rel(data.frame_t[-1])) if data.n_frames else 1e9
        ps = panel_analysis(data, lo, hi, REFERENCE_TAGS)
        if ps is None:
            print("Panel signals not added: need tags on two panels in the window.")
        else:
            data.series[ps.name] = ps
            print("Panel signals: " + ", ".join(f"{k} IMU explains {v['imu_explains_heldout_pct']:.0f}%"
                                               for k, v in ps.info["scores"].items() if k.startswith("both.")))

    columns = None
    if a.topics:
        a.topics = list(a.topics) + (["derived/imu_vel"] if a.integrate_imu else []) + \
            (["derived/panels"] if a.panels else []) + ["derived/thrusters",
                                                                                         "derived/heading"]
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
                         "warnings": data.meta.get("warnings", [])}, sidecar=a.sidecar)
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
