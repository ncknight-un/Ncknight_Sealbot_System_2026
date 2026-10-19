#!/usr/bin/env python3
"""
Convert a Ping Viewer Ping360 log (.bin) into a ROS 2 bag of
ping360_sonar_msgs/msg/SonarEcho messages, the same type the
ping360_sonar driver publishes on /scan_echo.

Usage (on the Jetson, with your workspace sourced):
    source ~/Sealbot_ROS/ros_ws/install/setup.bash
    python3 ping360_log_to_bag.py Ping360-pool-20191218-124752636.bin

Options:
    --out DIR              output bag folder (default: <log name>_bag)
    --topic NAME           topic to write (default: /scan_echo)
    --frame-id NAME        header.frame_id (default: sonar)
    --speed-of-sound M_S   used to compute range (default: 1500)
    --start-time ISO       wall-clock time of the first record, e.g.
                           2019-12-18T12:47:52.636 (default: taken from the
                           file name if it has the Ping Viewer date stamp,
                           otherwise the current time)
    --info                 only print a summary, no ROS needed
    --csv FILE             also write one row per beam to a CSV (no ROS needed)

Log format (from Ping Viewer's source, Qt QDataStream, big-endian):
    header : QString "PingViewer sensor log file", uint32 version,
             5 x QString build/OS info, int32 sensor family, int32 sensor type
    records: QString time "hh:mm:ss.zzz" (time since recording started),
             QByteArray raw bytes received from the sensor
The raw bytes are a Ping Protocol stream; messages can be split across
records, so they are reassembled here.
"""

import argparse
import array
import csv
import math
import os
import re
import struct
import sys
from datetime import datetime, timedelta

PING360_DEVICE_DATA = 2300
PING360_AUTO_DEVICE_DATA = 2301
SAMPLE_PERIOD_TICK = 25e-9  # seconds per sample_period unit
VALID_HEADER = "PingViewer sensor log file"


# ---------------------------------------------------------------- log file
class LogReader:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.buf = f.read()
        self.pos = 0
        self.header = self._read_header()

    def _u32(self):
        (v,) = struct.unpack_from(">I", self.buf, self.pos)
        self.pos += 4
        return v

    def _i32(self):
        (v,) = struct.unpack_from(">i", self.buf, self.pos)
        self.pos += 4
        return v

    def _bytes(self):
        n = self._u32()
        if n == 0xFFFFFFFF:  # null QByteArray / QString
            return b""
        if self.pos + n > len(self.buf):
            raise EOFError
        b = self.buf[self.pos:self.pos + n]
        self.pos += n
        return b

    def _qstring(self):
        return self._bytes().decode("utf-16-be", errors="replace")

    def _read_header(self):
        h = {"header": self._qstring()}
        if h["header"] != VALID_HEADER:
            raise ValueError(f"not a Ping Viewer log (header was {h['header']!r})")
        h["version"] = self._u32()
        for k in ("hash", "date", "tag", "os_name", "os_version"):
            h[k] = self._qstring()
        h["family"] = self._i32()
        h["type"] = self._i32()
        return h

    def records(self):
        """Yield (seconds_since_start, raw_bytes)."""
        day_offset = 0.0
        last = -1.0
        while self.pos < len(self.buf):
            try:
                t = self._qstring()
                data = self._bytes()
            except (EOFError, struct.error):
                break  # truncated final record
            if not t:
                break
            m = re.match(r"(\d+):(\d+):(\d+)\.(\d+)", t)
            if not m:
                continue
            hh, mm, ss, ms = (int(x) for x in m.groups())
            sec = hh * 3600 + mm * 60 + ss + ms / 1000.0
            if sec + day_offset < last - 3600:  # clock wrapped past 24 h
                day_offset += 86400.0
            sec += day_offset
            last = sec
            yield sec, data


# ------------------------------------------------------- ping protocol
class PingStream:
    """Reassembles Ping Protocol messages from arbitrary byte chunks."""

    def __init__(self):
        self.buf = bytearray()
        self.bad_checksums = 0

    def feed(self, chunk):
        self.buf += chunk
        out = []
        while True:
            start = self.buf.find(b"BR")
            if start < 0:
                # keep a trailing 'B' that may start the next message
                self.buf = self.buf[-1:] if self.buf.endswith(b"B") else bytearray()
                return out
            if start:
                del self.buf[:start]
            if len(self.buf) < 8:
                return out
            plen, mid = struct.unpack_from("<HH", self.buf, 2)
            total = 8 + plen + 2
            if len(self.buf) < total:
                return out
            (csum,) = struct.unpack_from("<H", self.buf, 8 + plen)
            if (sum(self.buf[:8 + plen]) & 0xFFFF) != csum:
                self.bad_checksums += 1
                del self.buf[:1]  # resync on the next 'BR'
                continue
            out.append((mid, bytes(self.buf[8:8 + plen])))
            del self.buf[:total]


def decode_device_data(mid, p):
    """Return a dict for 2300/2301 payloads, else None."""
    if mid == PING360_DEVICE_DATA:
        fmt = "<BBHHHHHH"
        if len(p) < struct.calcsize(fmt):
            return None
        (mode, gain, angle, tx_dur, sp, freq, nsamp, dlen) = struct.unpack_from(fmt, p)
    elif mid == PING360_AUTO_DEVICE_DATA:
        fmt = "<BBHHHHHHBBHH"
        if len(p) < struct.calcsize(fmt):
            return None
        (mode, gain, angle, tx_dur, sp, freq, _start, _stop, _steps, _delay,
         nsamp, dlen) = struct.unpack_from(fmt, p)
    else:
        return None
    off = struct.calcsize(fmt)
    data = p[off:off + dlen]
    if not data:
        return None
    return dict(gain=gain, angle=angle, transmit_duration=tx_dur,
                sample_period=sp, transmit_frequency=freq,
                number_of_samples=nsamp, data=data)


def beams(log_path):
    log = LogReader(log_path)
    stream = PingStream()
    counts = {}

    def gen():
        for t, chunk in log.records():
            for mid, payload in stream.feed(chunk):
                counts[mid] = counts.get(mid, 0) + 1
                d = decode_device_data(mid, payload)
                if d:
                    d["t"] = t
                    yield d

    return log, stream, counts, gen()


def grad_to_rad(grad):
    # same convention as the ping360_sonar driver: -200..199 gradians
    if grad >= 200:
        grad -= 400
    return grad * math.pi / 200.0


def range_m(d, c):
    return d["sample_period"] * SAMPLE_PERIOD_TICK * d["number_of_samples"] * c / 2.0


def start_time_from_name(path):
    m = re.search(r"(\d{8})-(\d{6})(\d{3})", os.path.basename(path))
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S") + \
        timedelta(milliseconds=int(m.group(3)))


# -------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--out")
    ap.add_argument("--topic", default="/scan_echo")
    ap.add_argument("--frame-id", default="sonar")
    ap.add_argument("--speed-of-sound", type=float, default=1500.0)
    ap.add_argument("--start-time")
    ap.add_argument("--info", action="store_true")
    ap.add_argument("--csv")
    a = ap.parse_args()

    if a.start_time:
        t0 = datetime.fromisoformat(a.start_time)
    else:
        t0 = start_time_from_name(a.log) or datetime.now()
    t0_ns = int(t0.timestamp() * 1e9)

    log, stream, counts, gen = beams(a.log)
    h = log.header
    print(f"Log version {h['version']}, Ping Viewer {h['tag'] or h['hash']} "
          f"({h['os_name']} {h['os_version']}), sensor family {h['family']} type {h['type']}")

    writer = None
    if not a.info:
        try:
            import rclpy.serialization  # noqa: F401
            from rclpy.serialization import serialize_message
            from rosbag2_py import (SequentialWriter, StorageOptions,
                                    ConverterOptions, TopicMetadata)
            from ping360_sonar_msgs.msg import SonarEcho
        except ImportError as e:
            sys.exit(f"ROS 2 import failed ({e}).\nSource ROS and your workspace first:\n"
                     "  source ~/Sealbot_ROS/ros_ws/install/setup.bash\n"
                     "or run with --info / --csv to inspect the log without ROS.")
        out = a.out or os.path.splitext(os.path.basename(a.log))[0] + "_bag"
        if os.path.exists(out):
            sys.exit(f"{out} already exists; delete it or pass --out")
        writer = SequentialWriter()
        writer.open(StorageOptions(uri=out, storage_id="sqlite3"),
                    ConverterOptions(input_serialization_format="cdr",
                                     output_serialization_format="cdr"))
        writer.create_topic(TopicMetadata(name=a.topic,
                                          type="ping360_sonar_msgs/msg/SonarEcho",
                                          serialization_format="cdr"))

    csv_file = csv_w = None
    if a.csv:
        csv_file = open(a.csv, "w", newline="")
        csv_w = csv.writer(csv_file)
        csv_w.writerow(["t_sec", "angle_rad", "range_m", "gain",
                        "number_of_samples", "intensities"])

    n = 0
    first_t = last_t = None
    ranges = set()
    for d in gen:
        n += 1
        first_t = d["t"] if first_t is None else first_t
        last_t = d["t"]
        r = range_m(d, a.speed_of_sound)
        ranges.add(round(r, 2))
        angle = grad_to_rad(d["angle"])

        if writer:
            msg = SonarEcho()
            stamp_ns = t0_ns + int(d["t"] * 1e9)
            msg.header.stamp.sec = stamp_ns // 1_000_000_000
            msg.header.stamp.nanosec = stamp_ns % 1_000_000_000
            msg.header.frame_id = a.frame_id
            msg.angle = float(angle)
            msg.gain = int(d["gain"])
            msg.number_of_samples = int(len(d["data"]))
            msg.transmit_frequency = int(d["transmit_frequency"])
            msg.speed_of_sound = int(round(a.speed_of_sound))
            msg.range = max(0, min(255, int(round(r))))
            msg.intensities = array.array("B", d["data"])
            writer.write(a.topic, serialize_message(msg), stamp_ns)

        if csv_w:
            csv_w.writerow([f"{d['t']:.3f}", f"{angle:.5f}", f"{r:.3f}", d["gain"],
                            len(d["data"]), " ".join(map(str, d["data"]))])

    if csv_file:
        csv_file.close()
    wrote_bag = writer is not None
    if writer is not None:
        del writer  # closes the bag

    print(f"Messages by id: {dict(sorted(counts.items()))}"
          f"{f', bad checksums: {stream.bad_checksums}' if stream.bad_checksums else ''}")
    if n == 0:
        sys.exit("No Ping360 beams (message 2300/2301) found in this log.")
    print(f"Beams: {n} over {last_t - first_t:.1f} s; range(s) at "
          f"{a.speed_of_sound:g} m/s: {sorted(ranges)} m")
    if wrote_bag:
        print(f"Wrote {a.out or os.path.splitext(os.path.basename(a.log))[0] + '_bag'} "
              f"(topic {a.topic}), start time {t0.isoformat()}")
    if a.csv:
        print(f"Wrote {a.csv}")


if __name__ == "__main__":
    main()
