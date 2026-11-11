#!/usr/bin/env python3
"""Measure egress/ingress intervals for selected MAVLink messages in UDP traffic."""

import argparse
import math
import socket
import statistics
import struct
import time
from collections import defaultdict


MESSAGE_NAMES = {
    27: "RAW_IMU",
    29: "SCALED_PRESSURE",
    30: "ATTITUDE",
}


def ipv4_udp_payload(frame):
    if len(frame) < 14:
        return None

    ether_type = struct.unpack_from("!H", frame, 12)[0]
    offset = 14
    while ether_type in (0x8100, 0x88A8):
        if len(frame) < offset + 4:
            return None
        ether_type = struct.unpack_from("!H", frame, offset + 2)[0]
        offset += 4

    if ether_type != 0x0800 or len(frame) < offset + 20:
        return None

    ip_header_len = (frame[offset] & 0x0F) * 4
    if frame[offset + 9] != 17 or len(frame) < offset + ip_header_len + 8:
        return None

    fragment = struct.unpack_from("!H", frame, offset + 6)[0]
    if fragment & 0x1FFF:
        return None

    src = socket.inet_ntoa(frame[offset + 12:offset + 16])
    dst = socket.inet_ntoa(frame[offset + 16:offset + 20])
    udp_offset = offset + ip_header_len
    src_port, dst_port, udp_len = struct.unpack_from("!HHH", frame, udp_offset)
    if udp_len < 8 or len(frame) < udp_offset + udp_len:
        return None
    payload = frame[udp_offset + 8:udp_offset + udp_len]
    return src, dst, src_port, dst_port, payload


def mavlink_ids(payload):
    offset = 0
    while offset < len(payload):
        start = payload.find(b"\xfe", offset)
        v2_start = payload.find(b"\xfd", offset)
        starts = [value for value in (start, v2_start) if value >= 0]
        if not starts:
            return
        offset = min(starts)
        version = payload[offset]

        if version == 0xFE:
            if len(payload) < offset + 6:
                return
            payload_len = payload[offset + 1]
            frame_len = 6 + payload_len + 2
            message_id = payload[offset + 5]
        else:
            if len(payload) < offset + 10:
                return
            payload_len = payload[offset + 1]
            incompat_flags = payload[offset + 2]
            frame_len = 10 + payload_len + 2 + (13 if incompat_flags & 1 else 0)
            message_id = int.from_bytes(payload[offset + 7:offset + 10], "little")

        if len(payload) < offset + frame_len:
            return
        yield message_id
        offset += frame_len


def report(label, timestamps):
    print(f"{label}: messages={len(timestamps)}")
    if len(timestamps) < 2:
        print("  Intervals: not enough samples")
        return

    intervals = [(b - a) / 1e6 for a, b in zip(timestamps, timestamps[1:])]
    ordered = sorted(intervals)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    print(
        f"  Intervals ms: mean={statistics.fmean(intervals):.2f} "
        f"p50={statistics.median(intervals):.2f} "
        f"p95={p95:.2f} max={max(intervals):.2f}"
    )
    print(f"  Bunches (<5 ms): {sum(value < 5 for value in intervals)}")
    print(f"  Gaps (>15 ms): {sum(value > 15 for value in intervals)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", required=True, help="Capture interface (eth0/enP8p1s0)")
    parser.add_argument("--source", default="192.168.2.2")
    parser.add_argument("--destination", default="192.168.2.3")
    parser.add_argument("--port", type=int, default=14551)
    parser.add_argument("--duration", type=float, default=30)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(3))
    try:
        sock.bind((args.interface, 0))
        sock.settimeout(0.5)
        print(
            f"Capturing {args.source} -> {args.destination}:{args.port} "
            f"on {args.interface} for {args.duration:g} seconds..."
        )
        deadline = time.monotonic() + args.duration
        timestamps = defaultdict(list)
        packet_count = 0

        while time.monotonic() < deadline:
            try:
                frame, _ = sock.recvfrom(65535)
            except socket.timeout:
                continue

            decoded = ipv4_udp_payload(frame)
            if decoded is None:
                continue
            src, dst, _, dst_port, payload = decoded
            if src != args.source or dst != args.destination or dst_port != args.port:
                continue

            packet_count += 1
            received_ns = time.monotonic_ns()
            for message_id in mavlink_ids(payload):
                if message_id in MESSAGE_NAMES:
                    timestamps[message_id].append(received_ns)
    finally:
        sock.close()

    print(f"Matching UDP packets: {packet_count}")
    for message_id, name in MESSAGE_NAMES.items():
        report(name, timestamps[message_id])


if __name__ == "__main__":
    main()
