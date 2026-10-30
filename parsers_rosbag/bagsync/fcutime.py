"""
fcutime.py - give MAVROS topics the autopilot's own sample times.

With MAVROS timesync not locked, every MAVROS message is stamped with the Jetson's clock when it
ARRIVES, and the link delivers messages in bursts, so stamps bunch up. But the raw MAVLink stream
(recorded on .../mavlink_source) still carries the autopilot's sample timestamp in most messages
(time_boot_ms in ATTITUDE, GLOBAL_POSITION_INT, ...; time_usec in RAW_IMU, ...). This module:

  1. reads every MAVLink message from the bag (straight from the bag; the cache can't hold the
     raw payload) and decodes the autopilot time t_fcu of the timestamped ones;
  2. per message type, finds the offset between the autopilot clock and the Jetson clock from the
     FASTEST arrivals: the lowest arrival-minus-t_fcu in blocks of a few seconds, with a straight
     line through those minimums (follows slow drift between the two clocks; a fresh line after
     an autopilot reboot). Corrected time = t_fcu + offset, on the Jetson (MAVROS header) clock;
  3. pairs each message of a MAVROS topic with the MAVLink message it was made from (by header
     stamp; topics without a header such as rel_alt: by the recorder's arrival time);
  4. replaces the topic's stamps with the corrected times, mapped onto the camera's clock the same
     way the topic's header clock already was.

Which MAVLink message feeds which MAVROS topic is listed in TOPIC_SOURCES. Topics made from messages that carry no timestamp (state = HEARTBEAT, battery =
SYS_STATUS / BATTERY_STATUS, vfr_hud) are left as recorded. Messages that can't be paired are
dropped from that topic (they would otherwise sit at a wrong time) and reported.

What remains is the link's smallest delay (a few ms, the same for every message): the offset is
measured from the fastest arrivals, which still took that long. It is the same for every topic.

The result is cached in the bag's cache folder (fcu_time.npz / fcu_time.json).
"""
from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import numpy as np

FCU_VERSION = 1
MAVLINK_TOPIC_SUFFIX = "mavlink_source"

# msgid -> (name, byte offset, struct format, ns per unit, wraps). The time field is the first
# field on the wire in every one of these (MAVLink orders fields by size; all start with it).
TIME_FIELDS = {
    24: ("GPS_RAW_INT", 0, "<Q", 1_000, False),
    26: ("SCALED_IMU", 0, "<I", 1_000_000, False),
    27: ("RAW_IMU", 0, "<Q", 1_000, False),
    29: ("SCALED_PRESSURE", 0, "<I", 1_000_000, False),
    30: ("ATTITUDE", 0, "<I", 1_000_000, False),
    31: ("ATTITUDE_QUATERNION", 0, "<I", 1_000_000, False),
    32: ("LOCAL_POSITION_NED", 0, "<I", 1_000_000, False),
    33: ("GLOBAL_POSITION_INT", 0, "<I", 1_000_000, False),
    36: ("SERVO_OUTPUT_RAW", 0, "<I", 1_000, True),       # uint32 microseconds: wraps every ~71 min
    65: ("RC_CHANNELS", 0, "<I", 1_000_000, False),
    105: ("HIGHRES_IMU", 0, "<Q", 1_000, False),
    116: ("SCALED_IMU2", 0, "<I", 1_000_000, False),
    129: ("SCALED_IMU3", 0, "<I", 1_000_000, False),
    132: ("DISTANCE_SENSOR", 0, "<I", 1_000_000, False),
    137: ("SCALED_PRESSURE2", 0, "<I", 1_000_000, False),
    141: ("ALTITUDE", 0, "<Q", 1_000, False),
    143: ("SCALED_PRESSURE3", 0, "<I", 1_000_000, False),
}

# (regex on the topic's short name, MAVLink message ids it can come from, in preference order).
# First matching pattern wins. When several ids are listed, the one that pairs up best is used.
TOPIC_SOURCES = [
    # MAVROS imu plugin: imu/data is published on ATTITUDE (or ATTITUDE_QUATERNION if streamed)
    (r"imu/data$", (31, 30)),
    (r"imu/(data_raw|mag)$", (105, 27, 26)),
    (r"imu/(static_pressure|temperature_baro|diff_pressure)$", (29, 137, 143)),
    # MAVROS global_position plugin: all from GLOBAL_POSITION_INT
    (r"global_position/(local|global|rel_alt|compass_hdg|gp_vel)$", (33,)),
    (r"global_position/raw/(fix|gps_vel)$", (24,)),
    (r"local_position/(pose|velocity_local|velocity_body|odom|accel)$", (32,)),
    (r"rc/in$", (65,)),
    (r"rc/out$", (36,)),
    (r"(^|/)altitude$", (141,)),
    (r"(rangefinder|distance_sensor)", (132,)),
]

# Values copied from a MAVLink field into a ROS message, used to confirm each pairing exactly:
# (topic regex, msgid, ROS column, ROS value -> int key, MAVLink field (struct fmt, byte offset), field -> int key)
_F32 = lambda v: np.asarray(v, np.float32).view(np.int32).astype(np.int64)  # noqa: E731
VALUE_KEYS = [
    (r"global_position/rel_alt$", 33, "data", lambda v: np.round(np.asarray(v) * 1000), ("<i", 16), lambda f: f),
    (r"global_position/local$", 33, "pose.pose.position.z", lambda v: np.round(np.asarray(v) * 1000),
     ("<i", 16), lambda f: f),
    (r"global_position/compass_hdg$", 33, "data", lambda v: np.round(np.asarray(v) * 100) % 36000,
     ("<H", 26), lambda f: f % 36000),
    (r"imu/data$", 30, "angular_velocity.x", _F32, ("<f", 16), _F32),        # ATTITUDE.rollspeed
    (r"imu/data$", 31, "angular_velocity.x", _F32, ("<f", 20), _F32),        # ATTITUDE_QUATERNION
]
VALUE_FIELDS = {}                       # msgid -> set of (fmt, offset) to decode
for _p, _m, _c, _tk, _f, _fk in VALUE_KEYS:
    VALUE_FIELDS.setdefault(_m, set()).add(_f)


def _value_key(short, msgid):
    for pat, mid, col, tk, fld, fk in VALUE_KEYS:
        if mid == msgid and re.search(pat, short):
            return col, tk, fld, fk
    return None


BLOCK_S = 5.0            # offset: lowest delay per block of this many seconds of autopilot time
HDR_TOL_MS = 3.0         # pairing tolerance when matching header stamps (same MAVROS callback)
RECV_TOL_MS = 25.0       # ... when matching recorder arrival times (topics without a header)
MIN_MATCH = 0.9          # a topic is re-stamped only if at least this share of it pairs up


def _sources_for(short: str):
    for pat, ids in TOPIC_SOURCES:
        if re.search(pat, short):
            return ids
    return None


# --------------------------------------------------------------------------- #
# Reading the bag
# --------------------------------------------------------------------------- #
def _read(bag_dir, mav_topic, targets: dict, msg_dir=None, progress=None):
    """Raw stamps from the bag. Returns (mavlink dict of arrays, {topic: (header_ns or None, recv_ns)})."""
    from rosbags.rosbag2 import Reader
    from bagcache import build_typestore, stamp_ns

    mav = {"h": [], "r": [], "id": [], "sys": [], "comp": [], "t": [], "v": []}
    h_all = []                       # header stamp of EVERY mavlink message (the topic's cache rows)
    tops = {t: ([], []) for t in targets}
    with Reader(bag_dir) as reader:
        conns = [c for c in reader.connections if c.topic == mav_topic or c.topic in targets]
        store = build_typestore(conns, msg_dir)
        total = max(sum(c.msgcount for c in conns), 1)
        for k, (c, recv, raw) in enumerate(reader.messages(connections=conns)):
            if progress and k % 2000 == 0:
                progress(k / total, f"Autopilot timing: reading {k:,}/{total:,}")
            msg = store.deserialize_cdr(raw, c.msgtype)
            if c.topic == mav_topic:
                h_all.append(stamp_ns(msg.header.stamp))
                tf = TIME_FIELDS.get(int(msg.msgid))
                if tf is None:
                    continue
                n = int(msg.len)
                buf = np.asarray(msg.payload64, dtype="<u8").tobytes()[:n]
                size = struct.calcsize(tf[2])
                buf = buf + b"\0" * max(0, tf[1] + size - len(buf))    # MAVLink 2 trims trailing zeros
                mav["h"].append(stamp_ns(msg.header.stamp))
                mav["r"].append(recv)
                mav["id"].append(int(msg.msgid))
                mav["sys"].append(int(msg.sysid))
                mav["comp"].append(int(msg.compid))
                mav["t"].append(struct.unpack_from(tf[2], buf, tf[1])[0])
                vals = {}
                for fmt, off in VALUE_FIELDS.get(int(msg.msgid), ()):
                    b2 = buf + b"\0" * max(0, off + struct.calcsize(fmt) - len(buf))
                    vals[(fmt, off)] = struct.unpack_from(fmt, b2, off)[0]
                mav["v"].append(vals)
            else:
                h = stamp_ns(msg.header.stamp) if hasattr(msg, "header") else None
                tops[c.topic][0].append(h)
                tops[c.topic][1].append(recv)
    vals = mav.pop("v")
    mav = {k: np.asarray(v, np.int64) for k, v in mav.items()}
    mav["v"] = vals
    mav["h_all"] = np.asarray(h_all, np.int64)
    out = {}
    for t, (h, r) in tops.items():
        hh = None if (not h or h[0] is None) else np.asarray(h, np.int64)
        out[t] = (hh, np.asarray(r, np.int64))
    return mav, out


# --------------------------------------------------------------------------- #
# Autopilot clock -> Jetson clock
# --------------------------------------------------------------------------- #
def _fcu_ns(raw: np.ndarray, msgid: int) -> np.ndarray:
    _, _, _, scale, wraps = TIME_FIELDS[msgid]
    v = raw.astype(np.int64)
    if wraps:   # uint32 microseconds: add 2^32 whenever it rolls over
        d = np.diff(v)
        v = v + np.concatenate([[0], np.cumsum(d < -(1 << 31))]) * (1 << 32)
    return v * scale


def _offset_line(t_fcu: np.ndarray, delay: np.ndarray):
    """Lower envelope of delay(t_fcu) (= arrival - t_fcu): a line through the lowest delay of each
    BLOCK_S block, moved down onto the lowest block. Returns (intercept_ns, slope) about t_fcu[0]."""
    x = (t_fcu - t_fcu[0]) / 1e9
    blk = np.floor(x / BLOCK_S).astype(int)
    xs, ys = [], []
    for b in np.unique(blk):
        m = blk == b
        j = np.argmin(delay[m])
        xs.append(x[m][j])
        ys.append(delay[m][j] / 1e6)                     # ms
    xs, ys = np.asarray(xs), np.asarray(ys)
    if len(xs) < 3 or np.ptp(xs) < 2 * BLOCK_S:
        return float(ys.min() * 1e6), 0.0
    a, c = np.polyfit(xs, ys, 1)
    res = ys - (a * xs + c)
    mad = np.median(np.abs(res - np.median(res))) + 1e-3
    keep = res < 3 * 1.4826 * mad + 1.0                   # drop blocks with no fast arrival
    if keep.sum() >= 3:
        a, c = np.polyfit(xs[keep], ys[keep], 1)
    if abs(a) > 1.0:                                      # > 1000 ppm: not a clock drift, use a constant
        a, c = 0.0, float(np.min(ys))
    c += float(np.min(ys - (a * xs + c)))                 # onto the fastest arrival (never early)
    return float(c * 1e6), float(a * 1e-3)                # ns, ns per ns


def _stream_times(t_fcu_raw, h, msgid):
    """Corrected times (Jetson header clock, ns) for one MAVLink stream in arrival order, plus info."""
    t = _fcu_ns(t_fcu_raw, msgid)
    out = np.empty(len(t), np.int64)
    # a fresh line after an autopilot reboot (time jumps back) or a long silence
    cuts = np.where((np.diff(t) < -1_000_000_000) | (np.diff(t) > 60_000_000_000))[0] + 1
    seg_info = []
    for s, e in zip(np.r_[0, cuts], np.r_[cuts, len(t)]):
        ts, hs = t[s:e], h[s:e]
        c, a = _offset_line(ts, hs - ts)
        off = c + a * (ts - ts[0])
        out[s:e] = ts + np.round(off).astype(np.int64)
        d = (hs - out[s:e]) / 1e6                         # each message's extra delay over the fastest
        seg_info.append({"messages": int(e - s), "drift_ppm": round(a * 1e6, 2),
                         "extra_delay_ms": {"median": round(float(np.median(d)), 2),
                                            "p95": round(float(np.percentile(d, 95)), 2),
                                            "max": round(float(d.max()), 2)}})
    return out, {"segments": seg_info, "reboots": int(len(cuts))}


# --------------------------------------------------------------------------- #
# Pairing ROS messages with MAVLink messages
# --------------------------------------------------------------------------- #
def _pair(a: np.ndarray, b: np.ndarray, tol_ns: float, va=None, vb=None, look: int = 4):
    """For each a (topic stamps, arrival order) the index of its b (stream stamps, arrival order),
    or -1. One-to-one and in order. The typical lag is taken out first; then each a looks at the
    next few unused b's within tol: with values (va, vb: a field copied from the MAVLink message
    into the ROS message) it takes the first whose value matches exactly, so lost messages on
    either side can't shift the pairing; without values it takes the nearest in time."""
    if len(a) == 0 or len(b) == 0:
        return np.full(len(a), -1)
    j = np.clip(np.searchsorted(b, a), 1, len(b) - 1)
    near = np.where(np.abs(a - b[j - 1]) <= np.abs(a - b[j]), b[j - 1], b[j])
    lag = float(np.median(a - near))
    out = np.full(len(a), -1)
    p = 0
    for i, ai in enumerate(a - lag):
        while p < len(b) and b[p] < ai - tol_ns:          # stream message with no topic message
            p += 1
        if p >= len(b):
            break
        cand = [k for k in range(p, min(p + look, len(b))) if abs(b[k] - ai) <= tol_ns]
        if not cand:
            continue
        if va is not None:
            hit = [k for k in cand if vb[k] == va[i]]
            if not hit:
                continue
            q = hit[0]
        else:
            q = min(cand, key=lambda k: abs(b[k] - ai))
        out[i] = q
        p = q + 1
    return out


def _pair_by_value(a, b, tol_ns, va, vb):
    """Like _pair, for topics paired by the recorder's arrival times, whose order transport delays
    can shuffle: each a takes the nearest unused b within tol that carries exactly its value."""
    if len(a) == 0 or len(b) == 0:
        return np.full(len(a), -1)
    j = np.clip(np.searchsorted(b, a), 1, len(b) - 1)
    near = np.where(np.abs(a - b[j - 1]) <= np.abs(a - b[j]), b[j - 1], b[j])
    lag = float(np.median(a - near))
    out = np.full(len(a), -1)
    used = np.zeros(len(b), bool)
    lo_all = np.searchsorted(b, a - lag - tol_ns)
    hi_all = np.searchsorted(b, a - lag + tol_ns, side="right")
    for i in range(len(a)):
        lo, hi = lo_all[i], hi_all[i]
        if hi <= lo:
            continue
        c = np.arange(lo, hi)
        c = c[(~used[lo:hi]) & (vb[lo:hi] == va[i])]
        if len(c):
            q = c[np.argmin(np.abs(b[c] - (a[i] - lag)))]
            out[i] = q
            used[q] = True
    return out


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #
def _cache_paths(data):
    d = Path(data._thumbs_path).parent
    return d / "fcu_time.npz", d / "fcu_time.json"


def _linear_map(src: np.ndarray, dst: np.ndarray):
    """dst ≈ dst0 + k (src - src0) fitted on paired arrays (both int64 ns)."""
    s0, d0 = int(src[0]), int(dst[0])
    if len(src) < 2 or np.ptp(src) == 0:
        return lambda v: np.asarray(v, np.int64) - s0 + d0
    k, c = np.polyfit((src - s0) / 1e9, (dst - d0) / 1e9, 1)
    return lambda v: d0 + np.round((k * (np.asarray(v, np.int64) - s0) / 1e9 + c) * 1e9).astype(np.int64)


def _row_keys(key: np.ndarray, n_rows: int):
    """Raw stamps -> (order of messages that became cache rows, the sorted unique keys)."""
    order = np.argsort(key, kind="stable")
    k = key[order]
    keep = np.ones(len(k), bool)
    keep[1:] = np.diff(k) > 0
    if keep.sum() != n_rows:
        return None
    return order[keep]


def fcu_time_series(data, msg_dir=None, progress=None, use_cache=True) -> list[dict]:
    """Re-stamp every MAVROS topic that has a timestamped MAVLink source with the autopilot's
    sample times, in place. Run right after load_bag, before deriving anything.
    Returns one report per topic considered."""
    mav_series = next((s for s in data.series.values()
                       if s.name.rstrip("/").endswith(MAVLINK_TOPIC_SUFFIX) and s.msgtype != "derived"), None)
    if mav_series is None:
        return [{"series": "(all)", "applied": False,
                 "reason": f"no ...{MAVLINK_TOPIC_SUFFIX} topic in this bag (record the raw MAVLink stream)"}]
    targets = {s.name: s for s in data.series.values()
               if s.msgtype != "derived" and s.info.get("kind") == "topic" and s is not mav_series
               and _sources_for(s.short)}
    # MAVROS topics with no timestamped MAVLink source: reported, left as recorded
    skipped = sorted(s.short for s in data.series.values()
                     if s.msgtype != "derived" and s.info.get("kind") == "topic" and s is not mav_series
                     and s.name not in targets and re.search(r"(^|/)mavros(/|$)", s.short))
    npz_p, json_p = _cache_paths(data)
    key = {"version": FCU_VERSION, "fingerprint": data.meta.get("fingerprint"),
           "targets": sorted(targets)}
    if use_cache and npz_p.exists() and json_p.exists():
        try:
            meta = json.loads(json_p.read_text())
            if meta.get("key") == key:
                z = np.load(npz_p)
                return _apply(data, targets, {n: (z[f"{i}_rows"], z[f"{i}_t"]) for i, n in
                                              enumerate(meta["names"]) if f"{i}_rows" in z}, meta["reports"],
                              skipped)
        except Exception as e:  # noqa: BLE001
            print(f"[bagsync] autopilot-timing cache unreadable, recomputing: {e}")

    mav, raw = _read(data.bag_dir, mav_series.name, set(targets), msg_dir, progress)
    reports, results = [], {}
    if len(mav["t"]) == 0:
        reports.append({"series": mav_series.short, "applied": False,
                        "reason": "no timestamped MAVLink messages decoded"})
    else:
        # the autopilot = the (sysid, compid) sending most timestamped messages
        pairs, counts = np.unique(np.stack([mav["sys"], mav["comp"]], 1), axis=0, return_counts=True)
        sysid, compid = pairs[np.argmax(counts)]
        own = (mav["sys"] == sysid) & (mav["comp"] == compid)
        streams = {}
        for mid in np.unique(mav["id"][own]):
            # MAVROS's arrival order is the header-stamp order (the bag's order is the recorder's,
            # which transport delays can shuffle slightly)
            sel = np.nonzero(own & (mav["id"] == mid))[0]
            sel = sel[np.argsort(mav["h"][sel], kind="stable")]
            T, info = _stream_times(mav["t"][sel], mav["h"][sel], int(mid))
            streams[int(mid)] = {"h": mav["h"][sel], "r": mav["r"][sel], "T": T, "info": info,
                                 "v": [mav["v"][i] for i in sel]}
        hdr_to_cam = _hdr_to_camera(mav_series, mav["h_all"])
        recv_rate = _recv_rate(mav)
        for name, s in sorted(targets.items(), key=lambda kv: kv[1].short):
            h, r = raw[name]
            rep = {"series": s.short}
            hdr_rows = h is not None and s.time_source in ("header", "header-converted")
            # the cache rows were made from the header stamps (or, without a usable header, arrival)
            rows = _row_keys(h if hdr_rows else r, len(s.t))
            if rows is None:
                rep.update(applied=False, reason="couldn't line the bag's messages up with the cached rows")
                reports.append(rep)
                continue
            # pair by header stamp when there is one, else (or if that pairs poorly) by arrival time
            modes = ([True] if h is not None else []) + [False]
            best = None
            for use_hdr in modes:
                k = (h if use_hdr else r)[rows]
                tol = HDR_TOL_MS * 1e6 if use_hdr else RECV_TOL_MS * 1e6 * recv_rate
                for mid in _sources_for(s.short):
                    st = streams.get(mid)
                    if st is None:
                        continue
                    vk = _value_key(s.short, mid)
                    va = vb = None
                    if vk is not None and vk[0] in s.cols:
                        col, tk, fld, fk = vk
                        va = np.asarray(tk(s.cols[col]), np.int64)         # cache rows = k's order
                        vb = np.asarray(fk(np.array([d.get(fld, np.nan) for d in st["v"]])), np.int64)
                    if use_hdr:
                        idx = _pair(k, st["h"], tol, va, vb)
                    else:                                           # by the recorder's arrival times
                        o = np.argsort(st["r"], kind="stable")
                        if va is not None:
                            idx = _pair_by_value(k, st["r"][o], tol, va, vb[o])
                        else:
                            idx = _pair(k, st["r"][o], tol)
                        idx = np.where(idx >= 0, o[np.clip(idx, 0, None)], -1)
                    frac = float(np.mean(idx >= 0))
                    if best is None or frac > best[1] + 1e-9:
                        best = (mid, frac, idx, use_hdr)
                if best is not None and best[1] >= 0.98:
                    break
            if best is None:
                rep.update(applied=False, reason="its MAVLink source ("
                           + "/".join(TIME_FIELDS[m][0] for m in _sources_for(s.short))
                           + ") isn't in the recorded MAVLink stream")
            elif best[1] < MIN_MATCH:
                rep.update(applied=False, reason=f"only {best[1]:.0%} of its messages pair up with "
                           f"{TIME_FIELDS[best[0]][0]} - left as recorded")
            else:
                mid, frac, idx, use_hdr = best
                ok = idx >= 0
                new_t = hdr_to_cam(streams[mid]["T"][idx[ok]])
                order = np.argsort(new_t, kind="stable")
                new_t = new_t[order]
                kept_rows = np.nonzero(ok)[0][order]
                uniq = np.ones(len(new_t), bool)
                uniq[1:] = np.diff(new_t) > 0
                old = s.t[kept_rows[uniq]]
                shift = (new_t[uniq] - old) / 1e6
                results[name] = (kept_rows[uniq], new_t[uniq])
                rep.update(applied=True, source=TIME_FIELDS[mid][0], matched=int(ok.sum()), messages=int(len(idx)),
                           dropped=int(len(idx) - uniq.sum()), key="header stamp" if use_hdr else "arrival time",
                           shift_ms={"median": round(float(np.median(shift)), 2),
                                     "min": round(float(shift.min()), 2), "max": round(float(shift.max()), 2)},
                           link=streams[mid]["info"])
            reports.append(rep)
    try:
        names = sorted(results)
        np.savez(npz_p, **{f"{i}_rows": results[n][0] for i, n in enumerate(names)},
                 **{f"{i}_t": results[n][1] for i, n in enumerate(names)})
        json_p.write_text(json.dumps({"key": key, "names": names, "reports": reports}, indent=1))
    except Exception as e:  # noqa: BLE001
        print(f"[bagsync] couldn't cache autopilot timing: {e}")
    return _apply(data, targets, results, reports, skipped)


def _hdr_to_camera(mav_series, hs):
    """MAVROS header clock -> the clock the cache uses (the camera's). The mavlink topic's cache rows
    were made from its header stamps the same way as every other MAVROS topic's, so fit that map."""
    if mav_series.time_source == "header":
        return lambda v: np.asarray(v, np.int64)
    rows = _row_keys(hs, len(mav_series.t))
    if rows is None:
        raise RuntimeError("mavlink topic rows don't line up with its header stamps")
    return _linear_map(hs[rows], mav_series.t)


def _recv_rate(mav) -> float:
    """Recorder seconds per header second (4 for a bag re-recorded from a 0.25x playback)."""
    if len(mav["h"]) < 10 or np.ptp(mav["h"]) == 0:
        return 1.0
    k = np.polyfit((mav["h"] - mav["h"][0]) / 1e9, (mav["r"] - mav["r"][0]) / 1e9, 1)[0]
    return float(k) if 0.01 < k < 100 else 1.0


def _apply(data, targets, results, reports, skipped=()):
    for name, (rows, new_t) in results.items():
        s = data.series.get(name)
        if s is None or len(rows) == 0 or rows.max() >= len(s.t):
            continue
        s.t = np.asarray(new_t, np.int64)
        s.cols = {c: v[rows] for c, v in s.cols.items()}
        rep = next((r for r in reports if r.get("series") == s.short), {})
        s.info["fcu_timed"] = {k: v for k, v in rep.items() if k != "series"}
        s.time_source = "autopilot"
    return reports + [{"series": n, "applied": False, "reason": "its MAVLink message carries no autopilot "
                       "timestamp (e.g. HEARTBEAT, SYS_STATUS, VFR_HUD) - left as recorded"} for n in skipped]


def fcu_text(reports: list[dict]) -> str:
    lines = []
    for r in reports:
        if r.get("applied"):
            seg = r["link"]["segments"]
            ed = seg[0]["extra_delay_ms"] if seg else {}
            lines.append(f"{r['series']}: autopilot sample times from {r['source']} (by {r['key']}), "
                         f"{r['matched']} of {r['messages']} msgs paired"
                         + (f", {r['dropped']} unpaired dropped" if r["dropped"] else "")
                         + f". Stamps moved by median {r['shift_ms']['median']:+.1f} ms "
                         f"(range {r['shift_ms']['min']:+.1f} to {r['shift_ms']['max']:+.1f}). Link delay over "
                         f"the fastest: median {ed.get('median', 0):.1f} ms, 95 % {ed.get('p95', 0):.1f} ms; "
                         f"clock drift {seg[0]['drift_ppm'] if seg else 0:+.1f} ppm"
                         + (f"; {r['link']['reboots']} autopilot reboot(s)" if r["link"].get("reboots") else ""))
        else:
            lines.append(f"{r['series']}: NOT re-stamped - {r['reason']}")
    return "\n".join(lines) if lines else "no MAVROS topics with an autopilot timestamp"
