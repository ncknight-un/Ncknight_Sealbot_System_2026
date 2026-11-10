#!/usr/bin/env python3
"""Measure MAVROS sensor callback timing and message header timestamps."""

import argparse
import math
import statistics
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from mavros_msgs.msg import Mavlink
from sensor_msgs.msg import FluidPressure, Imu


QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


class TopicTiming:
    def __init__(self):
        self.count = 0
        self.arrival_intervals_ms = []
        self.header_intervals_ms = []
        self.header_age_ms = []
        self.nonmonotonic_headers = 0
        self.last_arrival_ns = None
        self.last_header_ns = None

    def record(self, msg, now_ros_ns):
        arrival_ns = time.monotonic_ns()
        header_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        self.count += 1

        if self.last_arrival_ns is not None:
            self.arrival_intervals_ms.append(
                (arrival_ns - self.last_arrival_ns) / 1_000_000.0
            )
        self.last_arrival_ns = arrival_ns

        if header_ns > 0:
            if self.last_header_ns is not None:
                delta_ns = header_ns - self.last_header_ns
                if delta_ns <= 0:
                    self.nonmonotonic_headers += 1
                self.header_intervals_ms.append(delta_ns / 1_000_000.0)
            self.last_header_ns = header_ns
            self.header_age_ms.append((now_ros_ns - header_ns) / 1_000_000.0)


def describe(values):
    if not values:
        return "n/a"
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return (
        f"mean={statistics.fmean(values):.2f} "
        f"p50={statistics.median(ordered):.2f} "
        f"p95={ordered[p95_index]:.2f} max={max(ordered):.2f}"
    )


class TimingDiagnostic(Node):
    def __init__(self, prefix, raw_topic, report_period, expected_hz):
        super().__init__("mavros_timing_diagnostic")
        self._expected_hz = expected_hz
        self._report_period = report_period
        self._sensor_topics = {
            "imu": TopicTiming(),
            "pressure": TopicTiming(),
        }
        self._raw_topics = {
            27: TopicTiming(),
            29: TopicTiming(),
            30: TopicTiming(),
            31: TopicTiming(),
        }

        imu_topic = f"{prefix}/imu/data"
        pressure_topic = f"{prefix}/imu/static_pressure"
        self.create_subscription(
            Imu, imu_topic, lambda msg: self._record("imu", msg), QOS
        )
        self.create_subscription(
            FluidPressure,
            pressure_topic,
            lambda msg: self._record("pressure", msg),
            QOS,
        )
        self.create_subscription(Mavlink, raw_topic, self._record_raw, QOS)
        self.create_timer(report_period, self._report)
        self.get_logger().info(
            f"Monitoring {imu_topic}, {pressure_topic}, and {raw_topic}; "
            f"expected rate={expected_hz:g} Hz, report period={report_period:g} s"
        )
        if self.get_parameter("use_sim_time").value:
            self.get_logger().warn(
                "use_sim_time is enabled; header age compares against simulation time"
            )

    def _record(self, name, msg):
        self._sensor_topics[name].record(msg, self.get_clock().now().nanoseconds)

    def _record_raw(self, msg):
        topic = self._raw_topics.get(msg.msgid)
        if topic is not None:
            topic.record(msg, self.get_clock().now().nanoseconds)

    def _report(self):
        expected_period_ms = 1000.0 / self._expected_hz
        raw_names = {
            27: "raw/RAW_IMU(27)",
            29: "raw/SCALED_PRESSURE(29)",
            30: "raw/ATTITUDE(30)",
            31: "raw/ATTITUDE_QUATERNION(31)",
        }
        for name, topic in (
            list(self._sensor_topics.items())
            + [(raw_names[msgid], topic) for msgid, topic in self._raw_topics.items()]
        ):
            rate = topic.count / self._report_period
            bursts = sum(
                interval < expected_period_ms * 0.5
                for interval in topic.arrival_intervals_ms
            )
            gaps = sum(
                interval > expected_period_ms * 1.5
                for interval in topic.arrival_intervals_ms
            )
            self.get_logger().info(
                f"{name}: messages={topic.count} rate={rate:.1f}Hz | "
                f"arrival interval ms [{describe(topic.arrival_intervals_ms)}] "
                f"bursts(<{expected_period_ms * 0.5:.1f}ms)={bursts} "
                f"gaps(>{expected_period_ms * 1.5:.1f}ms)={gaps} | "
                f"header interval ms [{describe(topic.header_intervals_ms)}] "
                f"nonmonotonic={topic.nonmonotonic_headers} | "
                f"header age ms [{describe(topic.header_age_ms)}]"
            )
            topic.count = 0
            topic.arrival_intervals_ms.clear()
            topic.header_intervals_ms.clear()
            topic.header_age_ms.clear()
            topic.nonmonotonic_headers = 0


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix",
        default="/mavros/mavros",
        help="MAVROS topic prefix (default: /mavros/mavros)",
    )
    parser.add_argument(
        "--expected-hz",
        type=float,
        default=100.0,
        help="Expected rate used to flag burst/gap intervals (default: 100)",
    )
    parser.add_argument(
        "--raw-topic",
        default="/uas1/mavlink_source",
        help="Raw MAVLink topic (default: /uas1/mavlink_source)",
    )
    parser.add_argument(
        "--report-period",
        type=float,
        default=10.0,
        help="Seconds per report (default: 10)",
    )
    options, ros_args = parser.parse_known_args(args)
    if options.expected_hz <= 0.0 or options.report_period <= 0.0:
        parser.error("--expected-hz and --report-period must be positive")

    prefix = "/" + options.prefix.strip("/")
    rclpy.init(args=ros_args)
    node = TimingDiagnostic(
        prefix, options.raw_topic, options.report_period, options.expected_hz
    )
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
