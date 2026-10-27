#!/usr/bin/env python3
"""
imu_timing.py - how evenly is the IMU stamped?

Prints the IMU's message count and rate, the spacing between consecutive stamps, and how the
spacings are spread. A topic stamped when measured shows spacings close to its period (10 ms at
100 Hz). One stamped on arrival through a busy link shows many near-0 spacings followed by
20-80 ms pauses, at the same overall rate.

Usage:  python imu_timing.py BAG_FOLDER [--topic imu/data]
"""
import argparse
import sys

import numpy as np

from bagcache import load_bag


def main():
    ap = argparse.ArgumentParser(description="Show how evenly the IMU is stamped")
    ap.add_argument("bag")
    ap.add_argument("--topic", default="imu/data", help="series whose name ends with this (default imu/data)")
    a = ap.parse_args()
    d = load_bag(a.bag)
    s = next((v for k, v in d.series.items() if k.endswith("/" + a.topic.strip("/"))), None)
    if s is None:
        print(f"no topic ending in {a.topic}")
        return 1
    dt = np.diff(s.t) / 1e6
    span = (s.t[-1] - s.t[0]) / 1e9
    print(f"{s.short}: {len(s.t)} msgs over {span:.1f} s = {(len(s.t) - 1) / span:.1f} Hz, time source: {s.time_source}")
    p10, p50, p90 = np.percentile(dt, [10, 50, 90])
    print(f"spacing ms: median {p50:.1f}   10% {p10:.1f}   90% {p90:.1f}   max {dt.max():.1f}")
    edges = [0, 2, 6, 9, 11, 20, 40, 80, np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        n = int(((dt >= lo) & (dt < hi)).sum())
        print(f"  {lo:>3g}-{'' if hi == np.inf else f'{hi:g}':<3} ms: {n:6d}  ({100 * n / len(dt):5.1f} %)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
