"""
bagcache.py - read a ROS 2 bag once, flatten it into NumPy arrays, cache it.

The first time a bag is opened every message is deserialized, the useful
numeric/string fields are flattened into per-topic columns, and camera frames
are stored as small JPEG thumbnails.  Everything is written to a cache folder
inside the bag directory (`.bagsync_cache/`) so the viewer and the exporter
start in well under a second afterwards.

Works with ROS 2 Humble bags in MCAP or SQLite (.db3) format.  No ROS install
is needed - it uses the pure-Python `rosbags` library.

Timestamps
----------
Every series gets one time base, in int64 nanoseconds:
  * header.stamp, when the message has a header and the stamp is sane
  * otherwise the bag receive time (e.g. std_msgs/Float64 topics)
If a topic's header stamps are more than HEADER_SANITY_S away from its
receive times (unset stamps, or a different clock), it falls back to receive
time and a warning is recorded.

Special handling
----------------
  sensor_msgs/Image, CompressedImage -> camera frames (thumbnails, stamps)
  sensor_msgs/CameraInfo             -> intrinsics saved in metadata only
  tf2_msgs/TFMessage                 -> one series per parent->child pair
  /tf_static                         -> saved in metadata (one entry per parent->child edge);
                                        the viewer/exporter turn every edge into a series and
                                        measure it against /tf (see sync_export.tf_tree_series)
  apriltag_msgs/AprilTagDetectionArray -> a count series + one series per tag id
  quaternion fields (orientation / rotation) get derived roll/pitch/yaw columns
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

try:
    import cv2
except ImportError:  # thumbnails are optional
    cv2 = None

from rosbags.interfaces import MessageDefinitionFormat
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_types_from_idl, get_types_from_msg, get_typestore

CACHE_VERSION = 9
VERSION = "bagsync v11"
CACHE_DIRNAME = ".bagsync_cache"
HEADER_SANITY_S = 5.0        # max |receive - header| before falling back to receive time
MAX_ARRAY_ELEMS = 64         # numeric arrays longer than this are skipped (blobs)
DEFAULT_THUMB_WIDTH = 960
# Topics under these namespaces are not read at all (bluerov_sensor_node.py relays of MAVROS topics;
# everything is taken from MAVROS directly). Listed under "Skipped" in Bag info.
IGNORED_NAMESPACES = ("bluerov",)

# Fallback definitions for non-standard types used by this bag, in case the
# bag was recorded without embedded message definitions.  Definitions embedded
# in the bag always take priority.  Add more here, or pass --msg-dir.
FALLBACK_MSGDEFS = {
    "apriltag_msgs/msg/Point": "float64 x\nfloat64 y\n",
    "apriltag_msgs/msg/AprilTagDetection": (
        "string family\nint32 id\nint32 hamming\nfloat32 goodness\n"
        "float32 decision_margin\napriltag_msgs/Point centre\n"
        "apriltag_msgs/Point[4] corners\nfloat64[9] homography\n"
    ),
    "apriltag_msgs/msg/AprilTagDetectionArray": (
        "std_msgs/Header header\napriltag_msgs/AprilTagDetection[] detections\n"
    ),
    "mavros_msgs/msg/State": (
        "std_msgs/Header header\nbool connected\nbool armed\nbool guided\n"
        "bool manual_input\nstring mode\nuint8 system_status\n"
    ),
    "mavros_msgs/msg/RCIn": "std_msgs/Header header\nuint8 rssi\nuint16[] channels\n",
    "mavros_msgs/msg/RCOut": "std_msgs/Header header\nuint16[] channels\n",
}

IMAGE_TYPES = {"sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"}
TF_TYPE = "tf2_msgs/msg/TFMessage"
TAG_ARRAY_TYPE = "apriltag_msgs/msg/AprilTagDetectionArray"
CAMINFO_TYPE = "sensor_msgs/msg/CameraInfo"

ProgressFn = Callable[[float, str], None]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def short_name(topic: str) -> str:
    """'/mavros/mavros/imu/data' -> 'mavros/imu/data' (drop leading / and repeated segments)."""
    parts = [p for p in topic.strip("/").split("/") if p]
    out: list[str] = []
    for p in parts:
        if not out or out[-1] != p:
            out.append(p)
    return "/".join(out) or topic


def measured_rate_hz(t_ns) -> float:
    """Frames (or messages) per second, counted: (n - 1) intervals / (last - first stamp).
    n stamps span n - 1 intervals, so this is the true average rate over the recording,
    dropped frames included. No median, no interpolation."""
    t = np.asarray(t_ns)
    if len(t) < 2 or t[-1] <= t[0]:
        return 0.0
    return (len(t) - 1) / ((int(t[-1]) - int(t[0])) / 1e9)


def static_tf_edges(meta: dict) -> list[dict]:
    """/tf_static edges, one per parent->child (the last one published wins), frame ids without
    a leading '/'."""
    out: dict = {}
    for tf in meta.get("tf_static", []):
        p, c = tf["parent"].strip("/"), tf["child"].strip("/")
        out[(p, c)] = {**tf, "parent": p, "child": c}
    return list(out.values())


def stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def quat_to_euler_deg(x, y, z, w):
    """ZYX (yaw-pitch-roll) Euler angles in degrees. Works on arrays. NaN for zero quats."""
    x, y, z, w = (np.asarray(a, dtype=float) for a in (x, y, z, w))
    n = np.sqrt(x * x + y * y + z * z + w * w)
    with np.errstate(invalid="ignore", divide="ignore"):
        x, y, z, w = x / n, y / n, z / n, w / n
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    bad = ~(n > 1e-9)
    out = [np.degrees(a) for a in (roll, pitch, yaw)]
    for a in out:
        a[bad] = np.nan
    return out


def resolve_bag_dir(path: str | os.PathLike) -> Path:
    """Accept the bag folder, its metadata.yaml, or the .mcap/.db3 file inside it."""
    p = Path(path).expanduser().resolve()
    if p.is_file():
        p = p.parent
    if not (p / "metadata.yaml").exists():
        raise FileNotFoundError(
            f"No metadata.yaml in {p}. Point at the bag folder that holds metadata.yaml "
            "and the .mcap/.db3 file."
        )
    return p


# --------------------------------------------------------------------------- #
# Image decoding
# --------------------------------------------------------------------------- #
def image_msg_to_bgr(msg, msgtype: str) -> np.ndarray | None:
    """Convert sensor_msgs/Image or CompressedImage to a BGR (or gray) uint8 array."""
    if cv2 is None:
        return None
    if msgtype.endswith("CompressedImage"):
        return cv2.imdecode(np.asarray(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)

    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    enc = msg.encoding.lower()
    buf = np.asarray(msg.data, dtype=np.uint8)
    if h == 0 or w == 0 or buf.size < h * step:
        return None
    rows = buf[: h * step].reshape(h, step)

    def chans(n, dtype=np.uint8):
        itemsize = np.dtype(dtype).itemsize
        a = rows[:, : w * n * itemsize]
        a = np.ascontiguousarray(a).view(dtype)
        if dtype != np.uint8 and msg.is_bigendian:
            a = a.byteswap()
        return a.reshape(h, w, n) if n > 1 else a.reshape(h, w)

    if enc in ("bgr8", "8uc3"):
        return chans(3)
    if enc == "rgb8":
        return cv2.cvtColor(chans(3), cv2.COLOR_RGB2BGR)
    if enc == "bgra8":
        return cv2.cvtColor(chans(4), cv2.COLOR_BGRA2BGR)
    if enc == "rgba8":
        return cv2.cvtColor(chans(4), cv2.COLOR_RGBA2BGR)
    if enc in ("mono8", "8uc1"):
        return chans(1)
    if enc in ("mono16", "16uc1"):
        a = chans(1, np.uint16).astype(np.float32)
        lo, hi = np.percentile(a, (1, 99))
        return np.clip((a - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
    if enc in ("yuv422", "uyvy"):
        return cv2.cvtColor(rows[:, : w * 2].reshape(h, w, 2), cv2.COLOR_YUV2BGR_UYVY)
    if enc in ("yuv422_yuy2", "yuyv", "yuy2"):
        return cv2.cvtColor(rows[:, : w * 2].reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUY2)
    bayer = {
        "bayer_rggb8": cv2.COLOR_BayerBG2BGR, "bayer_bggr8": cv2.COLOR_BayerRG2BGR,
        "bayer_gbrg8": cv2.COLOR_BayerGR2BGR, "bayer_grbg8": cv2.COLOR_BayerGB2BGR,
    }
    if enc in bayer:
        return cv2.cvtColor(chans(1), bayer[enc])
    # Unknown encoding: show the first byte plane as grayscale rather than nothing.
    return rows[:, :w].copy()


# --------------------------------------------------------------------------- #
# Type store
# --------------------------------------------------------------------------- #
def _msg_name_from_path(p: Path) -> str:
    # .../<pkg>/msg/<Name>.msg  ->  <pkg>/msg/<Name>
    if p.parent.name == "msg":
        return f"{p.parent.parent.name}/msg/{p.stem}"
    return f"{p.parent.name}/msg/{p.stem}"


def build_typestore(connections, msg_dir: str | None = None):
    store = get_typestore(Stores.ROS2_HUMBLE)
    typs: dict = {}

    def add(new):
        for k, v in new.items():
            if k not in store.fielddefs:
                typs[k] = v

    for name, text in FALLBACK_MSGDEFS.items():
        try:
            add(get_types_from_msg(text, name))
        except Exception:
            pass
    if msg_dir:
        for p in sorted(Path(msg_dir).expanduser().rglob("*.msg")):
            try:
                add(get_types_from_msg(p.read_text(), _msg_name_from_path(p)))
            except Exception as e:  # noqa: BLE001
                print(f"[bagsync] could not parse {p}: {e}")
    for c in connections:  # definitions embedded in the bag win
        fmt, data = c.msgdef.format, c.msgdef.data
        try:
            if fmt == MessageDefinitionFormat.MSG and data:
                add(get_types_from_msg(data, c.msgtype))
            elif fmt == MessageDefinitionFormat.IDL and data:
                sep = "=" * 80 + "\n"
                if data.startswith(f"{sep}IDL: "):
                    for block in data.split(sep)[1:]:
                        add(get_types_from_idl(block.split("\n", 1)[1]))
                else:
                    add(get_types_from_idl(data))
        except Exception as e:  # noqa: BLE001
            print(f"[bagsync] embedded definition for {c.msgtype} not usable: {e}")
    store.register(typs)
    return store


# --------------------------------------------------------------------------- #
# Generic flattening
# --------------------------------------------------------------------------- #
def _is_time(v) -> bool:
    t = getattr(v, "__msgtype__", "")
    return t in ("builtin_interfaces/msg/Time", "builtin_interfaces/msg/Duration")


def flatten_msg(msg, out: dict, kinds: dict, prefix: str = "", include_cov: bool = False) -> None:
    for fname in msg.__dataclass_fields__:
        if fname.startswith("__") or fname.isupper() or fname == "header" or fname.endswith("frame_id") or (not include_cov and "covariance" in fname):
            continue
        _flatten_value(getattr(msg, fname), prefix + fname, out, kinds, include_cov)


def _flatten_value(v, name, out, kinds, include_cov):
    if isinstance(v, (bool, np.bool_)):
        out[name] = float(v)
        kinds.setdefault(name, "bool")
    elif isinstance(v, (int, np.integer)):
        out[name] = float(v)
        kinds.setdefault(name, "int")
    elif isinstance(v, (float, np.floating)):
        out[name] = float(v)
        kinds.setdefault(name, "float")
    elif isinstance(v, str):
        out[name] = v
        kinds.setdefault(name, "str")
    elif isinstance(v, np.ndarray):
        if v.ndim != 1 or v.size > MAX_ARRAY_ELEMS or v.dtype.kind not in "biuf":
            return
        k = "float" if v.dtype.kind == "f" else ("bool" if v.dtype.kind == "b" else "int")
        for i, x in enumerate(v.tolist()):
            out[f"{name}[{i}]"] = float(x)
            kinds.setdefault(f"{name}[{i}]", k)
    elif is_dataclass(v):
        if _is_time(v):
            out[name] = v.sec + v.nanosec * 1e-9
            kinds.setdefault(name, "float")
        else:
            flatten_msg(v, out, kinds, name + ".", include_cov)
    # lists of sub-messages (variable-length complex arrays) are skipped generically


# --------------------------------------------------------------------------- #
# Series
# --------------------------------------------------------------------------- #
@dataclass
class Series:
    name: str                       # unique key, e.g. '/mavros/mavros/imu/data' or '/tf[map->base]'
    short: str                      # CSV/GUI prefix, e.g. 'mavros/imu/data'
    msgtype: str
    t: np.ndarray                   # int64 ns, sorted, unique
    cols: dict                      # column name -> np.ndarray (float64, or str)
    kinds: dict                     # column name -> 'float' | 'int' | 'bool' | 'str'
    time_source: str = "header"
    derived: dict = field(default_factory=dict)   # euler col -> {'quat': prefix, 'part': 'roll'|...}
    info: dict = field(default_factory=dict)      # stats, tag id, etc.

    @property
    def rate_hz(self) -> float:
        """Average rate = messages / time: (n - 1) intervals over the span first -> last stamp."""
        return measured_rate_hz(self.t)


# --------------------------------------------------------------------------- #
# Clocks: header stamps vs the recorder's receive times
# --------------------------------------------------------------------------- #
# A bag recorded live has receive ≈ header + a few ms. A bag recorded while PLAYING another bag
# (e.g. running the AprilTag detector on a replay at 0.25x) keeps the original header stamps
# but its receive times are the re-recording's, possibly slowed or sped up. Then the header
# stamps are the real timing and receive times must not be used.
CLOCK_RATE_TOL = 2e-3        # header and receive clocks run at the same rate within 0.2 %
CLOCK_SAME_S = 1.0           # two topics share a header clock if they agree within 1 s


@dataclass
class Clock:
    """recv ≈ recv0 + rate * (hdr - hdr0) + c   (all ns); fitted robustly."""
    hdr0: int
    recv0: int
    rate: float              # receive seconds per header second (4.0 for a 0.25x replay)
    c: float                 # ns
    resid_ms: float

    def to_recv(self, hdr):
        return self.recv0 + self.rate * (np.asarray(hdr, np.int64) - self.hdr0) + self.c

    def to_hdr(self, recv):
        return np.round(self.hdr0 + ((np.asarray(recv, np.int64) - self.recv0) - self.c) / self.rate).astype(np.int64)

    def is_identity(self) -> bool:
        off = (self.recv0 + self.c - self.hdr0) / 1e9
        return abs(self.rate - 1.0) < CLOCK_RATE_TOL and abs(off) < HEADER_SANITY_S

    def same_as(self, other: "Clock", at_hdr: int) -> bool:
        return (abs(self.rate / other.rate - 1.0) < CLOCK_RATE_TOL
                and abs(float(self.to_recv(at_hdr) - other.to_recv(at_hdr))) / 1e9 < CLOCK_SAME_S * other.rate)


def fit_clock(hdr, recv) -> Clock | None:
    """Linear map header -> receive, or None if the header stamps are unusable (unset, constant,
    or not increasing with receive time)."""
    hdr, recv = np.asarray(hdr, np.int64), np.asarray(recv, np.int64)
    good = hdr != 0
    if good.sum() < 3:
        return None
    hdr, recv = hdr[good], recv[good]
    h0, r0 = int(hdr[0]), int(recv[0])
    x, y = (hdr - h0) / 1e9, (recv - r0) / 1e9
    if np.ptp(x) <= 0 or np.ptp(y) <= 0:
        return None
    a, c = np.polyfit(x, y, 1)
    res = y - (a * x + c)
    mad = np.median(np.abs(res - np.median(res))) + 1e-6
    keep = np.abs(res) < 6 * 1.4826 * mad + 0.05
    if keep.sum() >= 3:
        a, c = np.polyfit(x[keep], y[keep], 1)
        res = y - (a * x + c)
    if not (0.01 < a < 100) or np.corrcoef(x, y)[0, 1] < 0.99:
        return None
    # c is put at the low edge of the residuals (receive can only be late, never early)
    c_ns = (c + float(np.percentile(res, 5))) * 1e9
    return Clock(h0, r0, float(a), float(c_ns), float(np.median(np.abs(res)) * 1e3))


class _Accum:
    """Collects rows for one series while reading."""

    def __init__(self, name, short, msgtype, info=None):
        self.name, self.short, self.msgtype = name, short, msgtype
        self.hdr: list[int] = []
        self.recv: list[int] = []
        self.rows: list[dict] = []
        self.kinds: dict = {}
        self.has_header = False
        self.info = info or {}

    def add(self, recv_ns, hdr_ns, row):
        self.recv.append(recv_ns)
        self.hdr.append(hdr_ns if hdr_ns is not None else recv_ns)
        if hdr_ns is not None:
            self.has_header = True
        self.rows.append(row)

    def finish(self, warnings: list, ref: "Clock | None" = None) -> Series | None:
        """ref: the bag's reference header clock when the bag was recorded from a playback
        (receive times are not real time). None = a normal live recording."""
        if not self.rows:
            return None
        recv = np.asarray(self.recv, dtype=np.int64)
        hdr = np.asarray(self.hdr, dtype=np.int64)
        info = dict(self.info)
        source = "receive"
        t = recv
        if ref is not None:
            own = fit_clock(hdr, recv) if self.has_header else None
            if own is not None and own.same_as(ref, int(np.median(hdr))):
                source, t = "header", hdr
            elif own is not None:
                source, t = "header-converted", ref.to_hdr(own.to_recv(hdr))
                warnings.append(f"{self.name}: header clock differs from the camera's - converted onto it.")
            else:
                source, t = "receive-mapped", ref.to_hdr(recv)
            info["recv_minus_header_ms_median"] = float(np.median((ref.to_hdr(recv) - t) / 1e6))
        elif self.has_header:
            off = (recv - hdr) / 1e9
            info["recv_minus_header_ms_median"] = float(np.median(off) * 1e3)
            info["recv_minus_header_ms_p95"] = float(np.percentile(np.abs(off), 95) * 1e3)
            if np.all(hdr == 0) or abs(np.median(off)) > HEADER_SANITY_S:
                warnings.append(
                    f"{self.name}: header.stamp is {np.median(off):.1f} s from receive time "
                    "(unset or different clock) - using receive time instead."
                )
            else:
                source, t = "header", hdr
        order = np.argsort(t, kind="stable")
        t = t[order]
        keep = np.ones(len(t), dtype=bool)
        keep[1:] = np.diff(t) > 0           # drop duplicate stamps (keep first)
        idx = order[keep]
        t = t[keep]

        names: list[str] = []
        seen = set()
        for r in self.rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    names.append(k)
        cols = {}
        for k in names:
            if self.kinds.get(k) == "str":
                cols[k] = np.array([str(self.rows[i].get(k, "")) for i in idx])
            else:
                cols[k] = np.array([self.rows[i].get(k, np.nan) for i in idx], dtype=float)
        s = Series(self.name, self.short, self.msgtype, t, cols, dict(self.kinds), source, info=info)
        make_quats_continuous(s)
        add_euler_columns(s)
        return s


def quat_groups(cols) -> list[str]:
    """Column prefixes that form quaternions: '<p>.x/.y/.z/.w' with p ending in orientation/rotation."""
    out = []
    for c in cols:
        if c.endswith(".w"):
            p = c[:-2]
            if p.split(".")[-1] in ("orientation", "rotation", "quaternion") and all(
                f"{p}.{a}" in cols for a in "xyz"
            ):
                out.append(p)
    return out


def make_quats_continuous(s: Series) -> None:
    """q and -q are the same rotation. Detectors (AprilTag, tf) pick either sign
    per frame, so when w ~ 0 (tag seen ~180 deg rotated) the x/y/z/w plots jump
    between +/- values without any physical motion. Pick one sign consistently:
    the first valid sample has its largest component positive, every later
    sample is flipped if its dot product with the previous valid one is < 0."""
    for p in quat_groups(s.cols):
        q = np.stack([s.cols[f"{p}.{a}"] for a in "xyzw"], axis=1).astype(float)
        ok = np.where(np.isfinite(q).all(axis=1) & (np.abs(q).sum(axis=1) > 1e-9))[0]
        if len(ok) == 0:
            continue
        prev = q[ok[0]]
        if prev[np.argmax(np.abs(prev))] < 0:
            prev = -prev
        q[ok[0]] = prev
        for i in ok[1:]:
            if np.dot(q[i], prev) < 0:
                q[i] = -q[i]
            prev = q[i]
        for j, a in enumerate("xyzw"):
            s.cols[f"{p}.{a}"] = q[:, j]


def add_euler_columns(s: Series) -> None:
    for p in quat_groups(s.cols):
        q = [s.cols[f"{p}.{a}"] for a in "xyzw"]
        if np.all(np.nan_to_num(np.abs(q[0]) + np.abs(q[1]) + np.abs(q[2]) + np.abs(q[3])) < 1e-9):
            continue  # all-zero quaternion (e.g. imu/data_raw has no orientation)
        for part, arr in zip(("roll", "pitch", "yaw"), quat_to_euler_deg(*q)):
            if part != "pitch":
                # unwrap so an angle sitting near +/-180 (tags seen upside-down
                # through the dome have roll ~180) doesn't jump by 360 deg
                ok = np.isfinite(arr)
                if ok.any():
                    arr = arr.copy()
                    arr[ok] = np.degrees(np.unwrap(np.radians(arr[ok])))
                    # keep the series centred in (-180, 180] on average
                    arr[ok] -= 360.0 * np.round(np.median(arr[ok]) / 360.0)
            c = f"{p}.{part}_deg"
            s.cols[c] = arr
            s.kinds[c] = "float"
            s.derived[c] = {"quat": p, "part": part}


# --------------------------------------------------------------------------- #
# Bag data (cached)
# --------------------------------------------------------------------------- #
class BagData:
    def __init__(self, bag_dir: Path, meta: dict, series: dict, frames: dict, thumbs_path: Path):
        self.bag_dir = bag_dir
        self.meta = meta
        self.series: dict[str, Series] = series
        self.frame_t: np.ndarray = frames["t"]          # header (or receive) ns, sorted
        self.frame_recv: np.ndarray = frames["recv"]
        self._thumb_off: np.ndarray = frames["off"]
        self._thumb_len: np.ndarray = frames["len"]
        self._thumbs_path = thumbs_path
        self._thumb_fh = None
        self.camera_topic: str = meta["camera_topic"]
        self.t0: int = int(self.frame_t[0]) if len(self.frame_t) else int(meta["bag_start_ns"])
        self._tags_per_frame: list | None = None
        # the camera rate is always counted from the frame stamps (frames / time), also for
        # caches written by older versions that stored a median-based estimate
        self.meta.setdefault("frame_info", {})["rate_hz"] = self.fps

    @property
    def fps(self) -> float:
        """Camera frame rate = frames / time over the whole recording."""
        return measured_rate_hz(self.frame_t)

    # ---- frames ------------------------------------------------------------
    @property
    def n_frames(self) -> int:
        return len(self.frame_t)

    @property
    def frame_period_s(self) -> float:
        return float(np.median(np.diff(self.frame_t))) / 1e9 if self.n_frames > 1 else 0.0

    def rel(self, t_ns) -> np.ndarray:
        return (np.asarray(t_ns, dtype=np.int64) - self.t0) / 1e9

    def abs_ns(self, rel_s: float) -> int:
        return int(round(self.t0 + rel_s * 1e9))

    def frame_at(self, rel_s: float) -> int:
        """Index of the frame nearest to a time (seconds since first frame)."""
        if self.n_frames == 0:
            return -1
        t = self.abs_ns(rel_s)
        i = int(np.searchsorted(self.frame_t, t))
        if i <= 0:
            return 0
        if i >= self.n_frames:
            return self.n_frames - 1
        return i if (self.frame_t[i] - t) < (t - self.frame_t[i - 1]) else i - 1

    def thumb(self, i: int) -> np.ndarray | None:
        """Decoded thumbnail (BGR or gray) for frame i."""
        if cv2 is None or not (0 <= i < self.n_frames) or self._thumb_len[i] == 0:
            return None
        if self._thumb_fh is None:
            self._thumb_fh = open(self._thumbs_path, "rb")
        self._thumb_fh.seek(int(self._thumb_off[i]))
        buf = np.frombuffer(self._thumb_fh.read(int(self._thumb_len[i])), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)

    @property
    def thumb_scale(self) -> float:
        f = self.meta.get("frame_info", {})
        return float(f.get("thumb_width", 1)) / float(f.get("width", 1) or 1)

    # ---- tags --------------------------------------------------------------
    def tags_in_frame(self, i: int) -> list[tuple[int, np.ndarray]]:
        """[(tag_id, corners(4,2) in original pixels)] for frame i.

        Each frame takes the nearest detection of each tag within half a frame period, so a
        small offset between detection and image stamps can't make tags blink on and off.
        """
        if self._tags_per_frame is None:
            self._tags_per_frame = [[] for _ in range(self.n_frames)]
            self.tag_stamp_offset_ms = None
            if self.n_frames == 0:
                return []
            tol = int(max(1e6, 0.5 * self.frame_period_s * 1e9))
            offs = []
            for s in self.series.values():
                if s.info.get("kind") != "tag" or len(s.t) == 0:
                    continue
                cx = [s.cols.get(f"corners[{k}].{a}") for k in range(4) for a in "xy"]
                if any(c is None for c in cx):
                    continue
                corners = np.stack(cx, axis=1).reshape(-1, 4, 2)
                j = np.clip(np.searchsorted(s.t, self.frame_t), 0, len(s.t) - 1)
                jm = np.clip(j - 1, 0, len(s.t) - 1)
                best = np.where(np.abs(s.t[j] - self.frame_t) <= np.abs(s.t[jm] - self.frame_t), j, jm)
                d = s.t[best] - self.frame_t
                ok = np.abs(d) <= tol
                offs.append(d[ok])
                for f in np.nonzero(ok)[0]:
                    self._tags_per_frame[f].append((s.info["tag_id"], corners[best[f]]))
            if offs and sum(len(o) for o in offs):
                self.tag_stamp_offset_ms = float(np.median(np.concatenate(offs)) / 1e6)
        return self._tags_per_frame[i] if 0 <= i < self.n_frames else []

    def tags_held(self, i: int, hold: int = 0) -> list[tuple[int, np.ndarray, int]]:
        """[(tag_id, corners, age)] for frame i. age 0 = detected in this frame. A tag the
        detector missed for a few frames keeps its last outline for up to `hold` frames
        (age = frames since it was last seen), so the overlay doesn't blink on every miss."""
        out = [(k, c, 0) for k, c in self.tags_in_frame(i)]
        if hold <= 0:
            return out
        seen = {k for k, _, _ in out}
        for age in range(1, hold + 1):
            if i - age < 0:
                break
            for k, c in self.tags_in_frame(i - age):
                if k not in seen:
                    seen.add(k)
                    out.append((k, c, age))
        return out

    # ---- full-resolution frames straight from the bag ------------------------
    def iter_full_frames(self, i0: int, i1: int, msg_dir: str | None = None) -> Iterator[tuple[int, int, np.ndarray]]:
        """Yield (frame_idx, stamp_ns, image) for frames i0..i1 inclusive, read from the bag."""
        if self.n_frames == 0:
            return
        lo = int(self.frame_recv[i0:i1 + 1].min())
        hi = int(self.frame_recv[i0:i1 + 1].max()) + 1
        with Reader(self.bag_dir) as reader:
            conns = [c for c in reader.connections if c.topic == self.camera_topic]
            store = build_typestore(conns, msg_dir)
            use_hdr = self.meta["frame_info"].get("time_source") == "header"
            for c, recv, raw in reader.messages(connections=conns, start=lo, stop=hi):
                msg = store.deserialize_cdr(raw, c.msgtype)
                t = stamp_ns(msg.header.stamp) if use_hdr else recv
                k = int(np.searchsorted(self.frame_t, t))
                if k < self.n_frames and self.frame_t[k] == t and i0 <= k <= i1:
                    img = image_msg_to_bgr(msg, c.msgtype)
                    if img is not None:
                        yield k, t, img

    # ---- lens distortion ----------------------------------------------------
    @property
    def has_distortion(self) -> bool:
        ci = self.meta.get("camera_info") or {}
        return bool(ci) and any(abs(x) > 1e-9 for x in ci.get("d", []))

    def undistort_maps(self, size_wh: tuple[int, int]):
        """
        Remap tables that turn a raw thumbnail of size_wh into the rectified image
        that image_proc / apriltag would see (camera_info K, D, R -> P).
        None if there is no usable camera_info.
        """
        ci = self.meta.get("camera_info")
        if cv2 is None or not ci or not ci.get("width"):
            return None
        key = tuple(size_wh)
        if getattr(self, "_umaps_key", None) == key:
            return self._umaps
        w, h = size_wh
        s = w / float(ci["width"])
        K = np.array(ci["k"], float).reshape(3, 3)
        D = np.array(ci["d"], float)
        R = np.array(ci.get("r") or np.eye(3).ravel(), float).reshape(3, 3)
        if not np.any(R):
            R = np.eye(3)
        P = np.array(ci["p"], float).reshape(3, 4)[:, :3] if any(ci.get("p", [])) else K.copy()
        S = np.diag([s, s, 1.0])
        Ks, Ps = S @ K, S @ P
        if ci.get("distortion_model", "") in ("equidistant", "fisheye"):
            m1, m2 = cv2.fisheye.initUndistortRectifyMap(Ks, D[:4], R, Ps, (w, h), cv2.CV_16SC2)
        else:
            m1, m2 = cv2.initUndistortRectifyMap(Ks, D, R, Ps, (w, h), cv2.CV_16SC2)
        self._umaps_key, self._umaps = key, (m1, m2)
        return self._umaps

    def close(self):
        if self._thumb_fh:
            self._thumb_fh.close()
            self._thumb_fh = None


# --------------------------------------------------------------------------- #
# Build / load cache
# --------------------------------------------------------------------------- #
def _fingerprint(bag_dir: Path, settings: dict) -> dict:
    files = {}
    for p in sorted(bag_dir.iterdir()):
        if p.is_file() and (p.suffix in (".mcap", ".db3", ".yaml") or p.name.endswith(".zstd")):
            st = p.stat()
            files[p.name] = [st.st_size, int(st.st_mtime)]
    return {"version": CACHE_VERSION, "files": files, "settings": settings}


def load_bag(
    path,
    camera_topic: str | None = None,
    progress: ProgressFn | None = None,
    rebuild: bool = False,
    msg_dir: str | None = None,
    include_covariance: bool = False,
    thumb_width: int = DEFAULT_THUMB_WIDTH,
    cache_dir: str | None = None,
) -> BagData:
    """Open a bag, using the cache if it is up to date, otherwise building it."""
    bag_dir = resolve_bag_dir(path)
    cdir = Path(cache_dir) if cache_dir else bag_dir / CACHE_DIRNAME
    if not cache_dir and not os.access(bag_dir, os.W_OK):
        # read-only bag folder (e.g. a mounted drive): cache in the user's home instead
        cdir = Path.home() / ".cache" / "bagsync" / f"{bag_dir.name}_{hashlib.md5(str(bag_dir).encode()).hexdigest()[:8]}"
    settings = {"camera_topic": camera_topic, "include_covariance": include_covariance,
                "thumb_width": thumb_width}
    fp = _fingerprint(bag_dir, settings)
    meta_p, data_p, thumbs_p = cdir / "meta.json", cdir / "data.npz", cdir / "thumbs.bin"

    if not rebuild and meta_p.exists() and data_p.exists():
        try:
            meta = json.loads(meta_p.read_text())
            if meta.get("fingerprint") == fp:
                if progress:
                    progress(1.0, "Loaded from cache")
                return _load_cache(bag_dir, meta, data_p, thumbs_p)
        except Exception as e:  # noqa: BLE001
            print(f"[bagsync] cache unreadable, rebuilding: {e}")

    cdir.mkdir(parents=True, exist_ok=True)
    meta = _build_cache(bag_dir, fp, camera_topic, progress, msg_dir, include_covariance,
                        thumb_width, meta_p, data_p, thumbs_p)
    return _load_cache(bag_dir, meta, data_p, thumbs_p)


def _build_cache(bag_dir, fp, camera_topic, progress, msg_dir, include_cov, thumb_width,
                 meta_p, data_p, thumbs_p) -> dict:
    warnings: list[str] = []
    skipped: dict[str, str] = {}
    accums: dict[str, _Accum] = {}
    camera_info = None
    tf_static: list[dict] = []
    frame_hdr, frame_recv, thumb_off, thumb_len = [], [], [], []
    frame_info: dict = {}

    with Reader(bag_dir) as reader:
        conns = list(reader.connections)
        store = build_typestore(conns, msg_dir)
        topics = {}
        for c in conns:
            topics.setdefault(c.topic, c.msgtype)

        # pick camera topic
        cams = [t for t, ty in topics.items() if ty in IMAGE_TYPES]
        if camera_topic is None:
            pref = [t for t in cams if t.endswith("image_raw")] or cams
            camera_topic = pref[0] if pref else None
        elif camera_topic not in topics:
            raise ValueError(f"Camera topic {camera_topic!r} not in bag. Image topics: {cams}")
        if camera_topic is None:
            warnings.append("No image topic found - syncing will use a fixed rate instead.")

        usable = []
        for c in conns:
            if c.topic.strip("/").split("/")[0] in IGNORED_NAMESPACES:
                skipped[c.topic] = "ignored (bluerov_sensor_node topic; the MAVROS topic it relays is used)"
                continue
            if c.msgtype not in store.fielddefs:
                skipped[c.topic] = (f"type {c.msgtype} unknown (no embedded definition) - "
                                    "pass --msg-dir with its .msg files")
                continue
            if c.msgtype in IMAGE_TYPES and c.topic != camera_topic:
                skipped[c.topic] = "image topic other than the sync camera"
                continue
            usable.append(c)

        total = max(sum(c.msgcount for c in usable), 1)
        done = 0
        thumbs_fh = open(thumbs_p, "wb")
        try:
            for c, recv, raw in reader.messages(connections=usable):
                done += 1
                if progress and done % 500 == 0:
                    progress(done / total, f"Reading messages {done:,}/{total:,}")
                try:
                    msg = store.deserialize_cdr(raw, c.msgtype)
                except Exception as e:  # noqa: BLE001
                    skipped.setdefault(c.topic, f"deserialize failed: {e}")
                    continue
                hdr = stamp_ns(msg.header.stamp) if hasattr(msg, "header") else None

                if c.topic == camera_topic:
                    frame_hdr.append(hdr if hdr is not None else recv)
                    frame_recv.append(recv)
                    img = image_msg_to_bgr(msg, c.msgtype)
                    blob = b""
                    if img is not None:
                        if not frame_info:
                            frame_info = {"width": int(img.shape[1]), "height": int(img.shape[0]),
                                          "encoding": getattr(msg, "encoding", getattr(msg, "format", ""))}
                        if img.shape[1] > thumb_width:
                            s = thumb_width / img.shape[1]
                            img = cv2.resize(img, (thumb_width, max(1, int(round(img.shape[0] * s)))),
                                             interpolation=cv2.INTER_AREA)
                        frame_info.setdefault("thumb_width", int(img.shape[1]))
                        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
                        blob = enc.tobytes() if ok else b""
                    thumb_off.append(thumbs_fh.tell())
                    thumb_len.append(len(blob))
                    thumbs_fh.write(blob)
                    continue

                if c.msgtype == CAMINFO_TYPE:
                    if camera_info is None:
                        camera_info = {
                            "topic": c.topic, "width": int(msg.width), "height": int(msg.height),
                            "distortion_model": msg.distortion_model,
                            "k": np.asarray(msg.k).tolist(), "d": np.asarray(msg.d).tolist(),
                            "p": np.asarray(msg.p).tolist(),
                            "r": np.asarray(msg.r).tolist(),
                        }
                    continue

                if c.msgtype == TF_TYPE:
                    for tr in msg.transforms:
                        if c.topic == "/tf_static":
                            tf_static.append({
                                "parent": tr.header.frame_id, "child": tr.child_frame_id,
                                "translation": [tr.transform.translation.x, tr.transform.translation.y,
                                                tr.transform.translation.z],
                                "rotation_xyzw": [tr.transform.rotation.x, tr.transform.rotation.y,
                                                  tr.transform.rotation.z, tr.transform.rotation.w],
                            })
                            continue
                        key = f"{c.topic}[{tr.header.frame_id}->{tr.child_frame_id}]"
                        a = accums.get(key)
                        if a is None:
                            a = accums[key] = _Accum(key, f"{short_name(c.topic)}[{tr.header.frame_id}->{tr.child_frame_id}]",
                                                     "geometry_msgs/msg/TransformStamped",
                                                     {"kind": "tf", "source_topic": c.topic})
                        row: dict = {}
                        flatten_msg(tr.transform, row, a.kinds)
                        a.add(recv, stamp_ns(tr.header.stamp), row)
                    continue

                if c.msgtype == TAG_ARRAY_TYPE:
                    key = c.topic
                    a = accums.get(key)
                    if a is None:
                        a = accums[key] = _Accum(key, short_name(key), c.msgtype, {"kind": "tag_count"})
                    a.kinds["count"] = "int"
                    a.add(recv, hdr, {"count": float(len(msg.detections))})
                    for d in msg.detections:
                        tkey = f"{c.topic}[tag{int(d.id)}]"
                        ta = accums.get(tkey)
                        if ta is None:
                            ta = accums[tkey] = _Accum(tkey, f"{short_name(c.topic)}.tag{int(d.id)}", c.msgtype,
                                                       {"kind": "tag", "tag_id": int(d.id),
                                                        "family": d.family, "source_topic": c.topic})
                        row = {"centre.x": float(d.centre.x), "centre.y": float(d.centre.y),
                               "decision_margin": float(d.decision_margin), "hamming": float(d.hamming)}
                        for k, p in enumerate(d.corners):
                            row[f"corners[{k}].x"] = float(p.x)
                            row[f"corners[{k}].y"] = float(p.y)
                        for k2 in row:
                            ta.kinds.setdefault(k2, "int" if k2 == "hamming" else "float")
                        ta.add(recv, hdr, row)
                    continue

                a = accums.get(c.topic)
                if a is None:
                    a = accums[c.topic] = _Accum(c.topic, short_name(c.topic), c.msgtype,
                                                 {"kind": "topic"})
                row = {}
                flatten_msg(msg, row, a.kinds, include_cov=include_cov)
                a.add(recv, hdr, row)
        finally:
            thumbs_fh.close()
        bag_start = int(reader.start_time)
        bag_end = int(reader.end_time)

    if progress:
        progress(0.98, "Building arrays")

    # Reference clock. Camera header stamps if usable, otherwise the headered topic with most
    # messages. If it is not "receive ≈ header" the bag was recorded from a playback: use the
    # header stamps everywhere and map receive-only topics onto them.
    ref = fit_clock(frame_hdr, frame_recv) if frame_hdr else None
    ref_from = camera_topic
    if ref is None:
        cands = sorted((a for a in accums.values() if a.has_header and len(a.hdr) > 10),
                       key=lambda a: -len(a.hdr))
        for a in cands:
            ref = fit_clock(a.hdr, a.recv)
            if ref is not None:
                ref_from = a.name
                break
    playback = ref is not None and not ref.is_identity()
    clock_info = {"reference": ref_from, "playback_recording": bool(playback)}
    if ref is not None:
        clock_info.update({"receive_per_header_s": round(ref.rate, 5),
                           "offset_s": round((ref.recv0 + ref.c - ref.hdr0) / 1e9, 3)})
    if playback:
        warnings.append(
            f"This bag was recorded from a playback: receive time runs {ref.rate:.3f}x header time "
            f"(about {1 / ref.rate:.2f}x playback speed), offset {clock_info['offset_s']:.0f} s. "
            "Using the original header stamps for all topics; topics without a header are mapped onto "
            f"that clock (reference: {ref_from}).")

    series: dict[str, Series] = {}
    for k, a in accums.items():
        s = a.finish(warnings, ref if playback else None)
        if s is not None and s.cols:
            series[k] = s

    # Headerless topics (std_msgs/Float64 etc.) only have the receive time, which is late by the
    # transport delay. If a topic in the same namespace has a header (e.g. global_position/local
    # next to global_position/rel_alt and compass_hdg, all from the same MAVLink message), shift
    # the headerless one back by that sibling's median receive-minus-header delay.
    for s_ in series.values():
        if s_.time_source not in ("receive", "receive-mapped") or s_.info.get("kind") != "topic":
            continue
        parent = s_.name.rsplit("/", 1)[0]
        if parent.count("/") < 2:          # need a real shared namespace, not the root
            continue
        sib = [o for o in series.values() if o is not s_ and o.time_source in ("header", "header-converted")
               and o.name.rsplit("/", 1)[0] == parent and "recv_minus_header_ms_median" in o.info]
        if not sib:
            continue
        lat_ms = float(np.median([o.info["recv_minus_header_ms_median"] for o in sib]))
        s_.t = s_.t - int(round(lat_ms * 1e6))
        s_.time_source = "receive-corrected"
        s_.info["latency_corrected_ms"] = round(lat_ms, 2)
        s_.info["latency_from"] = [o.name for o in sib]

    # camera frames: sort by stamp, sanity check header, drop duplicates
    frames = {"t": np.zeros(0, np.int64), "recv": np.zeros(0, np.int64),
              "off": np.zeros(0, np.int64), "len": np.zeros(0, np.int64)}
    if frame_hdr:
        hdr = np.asarray(frame_hdr, np.int64)
        recv = np.asarray(frame_recv, np.int64)
        off = (recv - hdr) / 1e9
        frame_info["time_source"] = "header"
        frame_info["recv_minus_header_ms_median"] = float(np.median(off) * 1e3)
        t = hdr
        if playback and ref_from == camera_topic:
            frame_info["recv_minus_header_ms_median"] = float(np.median((ref.to_hdr(recv) - hdr) / 1e6))
        elif playback:
            own = fit_clock(hdr, recv)
            if own is not None and own.same_as(ref, int(np.median(hdr))):
                pass
            else:
                frame_info["time_source"] = "receive-mapped"
                t = ref.to_hdr(recv)
        elif np.all(hdr == 0) or abs(np.median(off)) > HEADER_SANITY_S:
            warnings.append(f"{camera_topic}: header.stamp unusable - frames use receive time.")
            frame_info["time_source"] = "receive"
            t = recv
        order = np.argsort(t, kind="stable")
        t = t[order]
        keep = np.ones(len(t), bool)
        keep[1:] = np.diff(t) > 0
        if (~keep).sum():
            warnings.append(f"{camera_topic}: {int((~keep).sum())} frames with duplicate stamps dropped.")
        idx = order[keep]
        frames = {"t": t[keep], "recv": recv[idx],
                  "off": np.asarray(thumb_off, np.int64)[idx], "len": np.asarray(thumb_len, np.int64)[idx]}
        dt = np.diff(frames["t"]) / 1e9
        if len(dt):
            med = float(np.median(dt))
            drops = int(np.sum(np.round(dt / med) - 1).clip(0)) if med > 0 else 0
            frame_info.update({"count": int(len(frames["t"])), "rate_hz": measured_rate_hz(frames["t"]),
                               "dropped_estimate": drops,
                               "max_gap_ms": float(dt.max() * 1e3)})
            if drops:
                warnings.append(f"{camera_topic}: about {drops} dropped frames "
                                f"(largest gap {dt.max() * 1e3:.0f} ms).")

    meta = {
        "fingerprint": fp,
        "bag_dir": str(bag_dir),
        "bag_start_ns": bag_start,
        "bag_end_ns": bag_end,
        "camera_topic": camera_topic,
        "frame_info": frame_info,
        "camera_info": camera_info,
        "clock": clock_info,
        "tf_static": tf_static,
        "skipped": skipped,
        "warnings": warnings,
        "series": [],
    }
    arrays = {"frame_t": frames["t"], "frame_recv": frames["recv"],
              "thumb_off": frames["off"], "thumb_len": frames["len"]}
    for i, (k, s) in enumerate(sorted(series.items())):
        colnames = list(s.cols)
        meta["series"].append({
            "name": s.name, "short": s.short, "msgtype": s.msgtype, "time_source": s.time_source,
            "columns": colnames, "kinds": s.kinds, "derived": s.derived, "info": s.info,
            "count": int(len(s.t)), "rate_hz": s.rate_hz,
        })
        arrays[f"s{i}_t"] = s.t
        for j, cn in enumerate(colnames):
            arrays[f"s{i}_c{j}"] = s.cols[cn]
    np.savez(data_p, **arrays)
    meta_p.write_text(json.dumps(meta, indent=1))
    if progress:
        progress(1.0, "Cache built")
    return meta


def _load_cache(bag_dir: Path, meta: dict, data_p: Path, thumbs_p: Path) -> BagData:
    z = np.load(data_p, allow_pickle=False)
    series = {}
    for i, sm in enumerate(meta["series"]):
        cols = {cn: z[f"s{i}_c{j}"] for j, cn in enumerate(sm["columns"])}
        series[sm["name"]] = Series(sm["name"], sm["short"], sm["msgtype"], z[f"s{i}_t"], cols,
                                    sm["kinds"], sm["time_source"], sm.get("derived", {}),
                                    sm.get("info", {}))
    frames = {"t": z["frame_t"], "recv": z["frame_recv"], "off": z["thumb_off"], "len": z["thumb_len"]}
    return BagData(bag_dir, meta, series, frames, thumbs_p)


# --------------------------------------------------------------------------- #
# Field status: empty / constant fields, and autopilot estimates (vs sensor readings)
# --------------------------------------------------------------------------- #
# (topic short-name regex, column regex, note). These are values the autopilot COMPUTES (its
# EKF/attitude estimate), not direct sensor readings. Edit to suit your setup.
AUTOPILOT_ESTIMATES = [
    (r"imu/data$", r"^orientation\.", "autopilot attitude estimate (EKF), not a sensor reading"),
    (r"global_position/local$", r".", "autopilot position/velocity estimate (EKF; no GPS/DVL on board)"),
    (r"global_position/global$", r".", "autopilot global position estimate (EKF)"),
    (r"global_position/compass_hdg$", r".", "autopilot heading estimate (EKF, from the compass)"),
    (r"global_position/rel_alt$", r".", "autopilot depth estimate (EKF, mostly the depth sensor)"),
    (r"local_position/", r".", "autopilot local position estimate (EKF)"),
    (r"vfr_hud$", r".", "autopilot summary values"),
    (r"^derived/heading$", r".", "calculated from the autopilot's attitude estimate"),
]


def autopilot_note(topic_short: str, col: str) -> str:
    import re
    for tp, cp, note in AUTOPILOT_ESTIMATES:
        if re.search(tp, topic_short) and re.search(cp, col):
            return note
    return ""


def field_status(s: "Series", col: str) -> tuple[str, str]:
    """('empty' | 'constant' | '', detail) for a recorded field. Calculated series are skipped."""
    if s.msgtype == "derived" or s.info.get("kind") == "tf_static":
        return "", ""
    v = s.cols[col]
    if v.dtype.kind in "US":
        vals = set(v.tolist())
        if vals <= {""}:
            return "empty", "no data (always blank)"
        return ("constant", f"always '{next(iter(vals))}'") if len(vals) == 1 else ("", "")
    fin = v[np.isfinite(v)]
    if len(fin) == 0:
        return "empty", "no data (always blank)"
    # mavros fills orientation with (0, 0, 0, 1) when the message carries none (e.g. imu/data_raw)
    import re
    m = re.match(r"(.*orientation)\.", col)
    if m and all(f"{m.group(1)}.{a}" in s.cols for a in "xyzw"):
        q = np.stack([s.cols[f"{m.group(1)}.{a}"] for a in "xyzw"], 1)
        qf = q[np.isfinite(q).all(1)]
        if len(qf) and np.all(qf[:, :3] == 0):
            return "empty", "no orientation in this message (placeholder 0, 0, 0, 1)"
    if np.all(fin == 0):
        # 0 is "no data" for measurements and PWM outputs (0 = output unused), but a real value for
        # counts, error tallies (hamming) and true/false flags
        kind = s.kinds.get(col, "float")
        if kind == "float" or re.fullmatch(r"channels\[\d+\]", col):
            return "empty", "no data (always 0)"
        return "constant", "always 0"
    if np.ptp(fin) == 0:
        return "constant", f"always {fin[0]:g}"
    return "", ""


def bag_summary(data: BagData) -> str:
    """Human-readable overview of what was read and how it is timed."""
    m = data.meta
    lines = [f"Bag: {m['bag_dir']}",
             f"Duration: {(m['bag_end_ns'] - m['bag_start_ns']) / 1e9:.2f} s"]
    fi = m.get("frame_info", {})
    if data.camera_topic:
        lines.append(
            f"Sync camera: {data.camera_topic}  {data.n_frames} frames @ {data.fps:.3f} fps (frames / time), "
            f"{fi.get('width')}x{fi.get('height')} {fi.get('encoding')}, stamps from {fi.get('time_source')}, "
            f"~{fi.get('dropped_estimate', 0)} dropped, max gap {fi.get('max_gap_ms', 0):.0f} ms, "
            f"latency {fi.get('recv_minus_header_ms_median', float('nan')):.1f} ms"
        )
    ck = m.get("clock") or {}
    if ck.get("playback_recording"):
        lines.append(f"Clock: recorded from a playback at ~{1 / ck['receive_per_header_s']:.2f}x speed "
                     f"(receive runs {ck['receive_per_header_s']:.3f}x header time). All timing uses the "
                     f"original header stamps; reference {ck.get('reference')}.")
    elif ck:
        lines.append(f"Clock: live recording (header ≈ receive), reference {ck.get('reference')}.")
    lines.append("")
    lines.append(f"{'series':58s} {'n':>6s} {'Hz':>7s}  time    latency(ms)")
    for s in sorted(data.series.values(), key=lambda s: s.name):
        lat = s.info.get("recv_minus_header_ms_median")
        lat_s = f"{lat:8.1f}" if lat is not None else "       -"
        lc = s.info.get("latency_corrected_ms")
        extra = f"   (shifted -{lc:.1f} ms, delay of {short_name(s.info['latency_from'][0])})" if lc is not None else ""
        lines.append(f"{s.name[:58]:58s} {len(s.t):6d} {s.rate_hz:7.1f}  {s.time_source:7s} {lat_s}{extra}")
    empty, const, auto = [], [], {}
    for s in sorted(data.series.values(), key=lambda s: s.name):
        for c in s.cols:
            st, _ = field_status(s, c)
            if st == "empty":
                empty.append(f"{s.short}.{c}")
            elif st == "constant":
                const.append(f"{s.short}.{c}")
            note = autopilot_note(s.short, c)
            if note:
                auto.setdefault(f"{s.short}  ({note})", 0)
                auto[f"{s.short}  ({note})"] += 1
    if empty:
        lines.append(f"\nEmpty fields (no data, shown in red): {len(empty)}")
        lines += [f"  {e}" for e in empty]
    if const:
        lines.append(f"\nConstant fields (never change, shown in grey): {len(const)}")
        lines += [f"  {e}" for e in const]
    if auto:
        lines.append("\nAutopilot estimates, not sensor readings (shown in orange):")
        lines += [f"  {k}: {n} fields" for k, n in auto.items()]
    if m.get("skipped"):
        lines.append("\nSkipped:")
        lines += [f"  {k}: {v}" for k, v in m["skipped"].items()]
    if m.get("warnings"):
        lines.append("\nWarnings:")
        lines += [f"  {w}" for w in m["warnings"]]
    return "\n".join(lines)
