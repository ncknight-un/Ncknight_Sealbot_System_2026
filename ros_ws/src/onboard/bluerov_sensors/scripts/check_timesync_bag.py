#!/usr/bin/env python3
"""Summarize recorded MAVROS timesync and FCU/sample timing from a ROS 2 bag."""

import argparse
import math
import statistics
import struct
from collections import defaultdict
from pathlib import Path

from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message


TIMESYNC_TOPIC = "/mavros/mavros/timesync_status"
SENSOR_TOPICS = (
    "/mavros/mavros/imu/data",
    "/mavros/mavros/imu/static_pressure",
)
RAW_TOPIC_SUFFIX = "/mavlink_source"
FCU_FIELDS = {
    27: ("RAW_IMU", "<Q", 1_000),
    29: ("SCALED_PRESSURE", "<I", 1_000_000),
    30: ("ATTITUDE", "<I", 1_000_000),
}


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def summary(values, unit="ms"):
    if not values:
        return "n/a"
    values = sorted(values)
    p95 = values[max(0, math.ceil(0.95 * len(values)) - 1)]
    return (
        f"mean={statistics.fmean(values):.3f}{unit} "
        f"p50={statistics.median(values):.3f}{unit} "
        f"p95={p95:.3f}{unit} max={max(values):.3f}{unit}"
    )


def intervals_ns(values):
    return [(b - a) / 1_000_000.0 for a, b in zip(values, values[1:])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="ROS 2 bag directory")
    parser.add_argument(
        "--timesync-topic", default=TIMESYNC_TOPIC,
        help=f"Timesync status topic (default: {TIMESYNC_TOPIC})",
    )
    args = parser.parse_args()
    if not args.bag.is_dir():
        parser.error(f"bag directory does not exist: {args.bag}")

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(args.bag), storage_id="mcap"),
        ConverterOptions(input_serialization_format="cdr",
                         output_serialization_format="cdr"),
    )
    type_by_topic = {
        entry.name: entry.type for entry in reader.get_all_topics_and_types()
    }
    relevant = {
        name for name in SENSOR_TOPICS if name in type_by_topic
    }
    raw_topics = [
        name for name in type_by_topic if name.endswith(RAW_TOPIC_SUFFIX)
    ]
    relevant.update(raw_topics)
    if args.timesync_topic in type_by_topic:
        relevant.add(args.timesync_topic)
    if not relevant:
        parser.error("none of the requested MAVROS topics are present in this bag")

    message_types = {
        name: get_message(type_by_topic[name]) for name in relevant
    }
    topic_data = defaultdict(lambda: {"bag": [], "header": []})
    raw_fcu = defaultdict(list)
    rtts_ms = []
    offsets_ns = []

    while reader.has_next():
        topic, serialized, bag_time_ns = reader.read_next()
        if topic not in relevant:
            continue
        msg = deserialize_message(serialized, message_types[topic])
        if topic == args.timesync_topic:
            rtts_ms.append(float(msg.round_trip_time_ms))
            offsets_ns.append(int(msg.estimated_offset_ns))
            continue
        if topic in SENSOR_TOPICS:
            topic_data[topic]["bag"].append(bag_time_ns)
            topic_data[topic]["header"].append(stamp_ns(msg.header.stamp))
            continue
        if topic in raw_topics:
            mid = int(msg.msgid)
            field = FCU_FIELDS.get(mid)
            if field is None or int(msg.len) < struct.calcsize(field[1]):
                continue
            name, fmt, scale = field
            payload = b"".join(
                int(word).to_bytes(8, byteorder="little")
                for word in msg.payload64
            )
            raw_fcu[name].append(struct.unpack_from(fmt, payload)[0])

    print(f"Bag: {args.bag}")
    if rtts_ms:
        print(f"Timesync samples: {len(rtts_ms)}")
        print(f"RTT: {summary(rtts_ms)}")
        nonzero_offsets = [value for value in offsets_ns if value != 0]
        if nonzero_offsets:
            offset_ms = [value / 1_000_000.0 for value in nonzero_offsets]
            print(f"Nonzero estimated offsets: {len(nonzero_offsets)}/{len(offsets_ns)}")
            print(f"Estimated offset: {summary(offset_ms)}")
            print(
                f"Estimated offset p95-p5 spread: "
                f"{(sorted(offset_ms)[max(0, math.ceil(.95 * len(offset_ms)) - 1)] - sorted(offset_ms)[int(.05 * (len(offset_ms) - 1))]):.3f}ms"
            )
        else:
            print("Estimated offset: all zero or no usable estimate")
    else:
        print(f"No messages on timesync topic {args.timesync_topic}")

    for topic in SENSOR_TOPICS:
        if topic not in topic_data:
            print(f"{topic}: not present")
            continue
        data = topic_data[topic]
        print(f"{topic}: messages={len(data['header'])}")
        print(f"  Header intervals: {summary(intervals_ns(data['header']))}")
        print(f"  Bag receive intervals: {summary(intervals_ns(data['bag']))}")

    for name in ("RAW_IMU", "SCALED_PRESSURE", "ATTITUDE"):
        samples = raw_fcu[name]
        _, _, ns_per_unit = next(
            field for field in FCU_FIELDS.values() if field[0] == name
        )
        diffs = []
        for a, b in zip(samples, samples[1:]):
            delta = b - a
            if name != "RAW_IMU" and delta < -(1 << 31):
                delta += 1 << 32
            diffs.append(delta * ns_per_unit / 1_000_000.0)
        if samples:
            print(f"FCU {name}: messages={len(samples)}")
            print(f"  Sample intervals: {summary(diffs)}")
        else:
            print(f"FCU {name}: no messages")


if __name__ == "__main__":
    main()
