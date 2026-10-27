#!/usr/bin/env python3
"""
clock_check.py - are the camera and IMU stamped on the same clock?

Reads header.stamp and the recorder's receive time of every camera frame and IMU message,
straight from the bag (not the cache). For each topic it fits
    receive time = start + rate x (header time) + delay
(arrivals are only ever late, so the fit follows the earliest arrivals). Two topics stamped on
the same clock have the same rate and nearly the same line; the gap between their lines is the
clock offset (plus the difference between their smallest delays, a few ms).

Usage:  python clock_check.py BAG_FOLDER [--imu imu/data] [--camera image_raw]
"""
import argparse
import sys

import numpy as np
from rosbags.rosbag2 import Reader

from bagcache import build_typestore, fit_clock, resolve_bag_dir, stamp_ns


def read_stamps(bag, want):
    """{topic: (header_ns, receive_ns)} for topics whose name ends with any of `want`."""
    out = {}
    with Reader(bag) as r:
        conns = [c for c in r.connections if any(c.topic.endswith(w) for w in want)]
        store = build_typestore(conns)
        for c, recv, raw in r.messages(connections=conns):
            msg = store.deserialize_cdr(raw, c.msgtype)
            if not hasattr(msg, "header"):
                continue
            h, rv = out.setdefault(c.topic, ([], []))
            h.append(stamp_ns(msg.header.stamp))
            rv.append(recv)
    return {k: (np.array(h, np.int64), np.array(rv, np.int64)) for k, (h, rv) in out.items()}


def main():
    ap = argparse.ArgumentParser(description="Check whether camera and IMU stamps share one clock")
    ap.add_argument("bag")
    ap.add_argument("--imu", default="imu/data")
    ap.add_argument("--camera", default="image_raw")
    a = ap.parse_args()
    bag = resolve_bag_dir(a.bag)
    st = read_stamps(bag, [a.camera, a.imu])
    cam = next((k for k in st if k.endswith(a.camera)), None)
    imu = next((k for k in st if k.endswith(a.imu)), None)
    if cam is None or imu is None:
        print(f"need both topics; found {list(st)}")
        return 1

    fits = {}
    for name in (cam, imu):
        h, rv = st[name]
        f = fit_clock(h, rv)
        if f is None:
            print(f"{name}: header stamps unusable (unset, constant or not increasing)")
            return 1
        fits[name] = f
        print(f"{name}: {len(h)} msgs, header stamps {h[0] / 1e9:.3f} -> {h[-1] / 1e9:.3f} s; "
              f"receive runs {f.rate:.4f}x header time; typical extra delay {f.resid_ms:.1f} ms")

    fc, fi = fits[cam], fits[imu]
    rate_diff = (fi.rate / fc.rate - 1) * 100
    # at the middle of the recording: which camera header time has the same receive time as the IMU's?
    h_mid = int(np.median(st[imu][0]))
    offset_ms = (h_mid - fc.to_hdr(fi.to_recv(h_mid))) / 1e6
    playback = abs(fc.rate - 1) > 2e-3
    print()
    if playback:
        print(f"Note: this bag was recorded while playing another bag at ~{1 / fc.rate:.2f}x, so receive times "
              "are the re-recording's. The comparison still works, but its precision is lower.")
    print(f"Clock rates differ by {rate_diff:+.3f} %   (same clock: within about 0.2 %)")
    print(f"IMU header clock vs camera header clock: {offset_ms:+.1f} ms   "
          f"(+ = IMU stamps are later than the camera's for the same moment)")
    cam_h, imu_h = st[cam][0], st[imu][0]
    looks_boot = lambda h: h[0] < 1e9 * 86400 * 365       # under a year since zero: time since boot  # noqa: E731
    looks_unix = lambda h: h[0] > 1e9 * 1.5e9              # after 2017: Unix (date-and-time) clock  # noqa: E731
    if abs(rate_diff) < 0.2 and abs(offset_ms) < 20:
        print("-> SAME clock. The difference is within what the transport delays explain.")
    elif abs(rate_diff) < 0.2 and ((looks_boot(cam_h) and looks_unix(imu_h)) or (looks_unix(cam_h) and looks_boot(imu_h))):
        which = "camera" if looks_boot(cam_h) else "IMU"
        other = "IMU" if which == "camera" else "camera"
        print(f"-> DIFFERENT clocks: the {which} is stamped with time since the computer booted, the {other} with "
              f"date-and-time (Unix). They tick at the same rate, so the offset is fixed: it is the moment the "
              f"computer booted.\n   bagsync's export already maps the IMU onto the camera's clock through the "
              f"recorder's times (time source 'header-converted'), so exported CSVs are NOT off by this amount;\n"
              f"   at most by a fixed few tens of ms. To remove the mismatch at the source, stamp the {which} with ROS "
              f"time (e.g. the camera driver's use_ros_time parameter).")
    elif abs(rate_diff) < 0.2:
        print(f"-> Same rate but offset by {offset_ms / 1000:+.3f} s: DIFFERENT clocks (or one of them "
              "is stamped late by a fixed amount). Tags and IMU would be misaligned by this much.")
    else:
        print("-> DIFFERENT clocks: they run at different rates.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
