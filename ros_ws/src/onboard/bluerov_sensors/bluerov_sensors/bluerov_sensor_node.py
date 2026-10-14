#!/usr/bin/env python3
"""
BlueROV2 MAVROS Sensor Node  —  ROS 2 Humble / ArduSub
-------------------------------------------------------
Relays MAVROS topics to /bluerov/*, computes attitude (roll/pitch/yaw),
heading and depth, requests MAVLink stream rates once connected, and
prints a periodic status report showing which topics are actually live.

Timing:
  * On connect, MAVROS timesync is switched off (timesync_mode = NONE), so
    every MAVROS message is stamped with the Jetson's clock on arrival —
    the same time base as the camera (use-ros-time).
  * Link latency is measured once a second by sending the autopilot a
    MAVLink REQUEST_MESSAGE through MAVROS and timing the COMMAND_ACK.
    One-way latency (RTT / 2, ms) is published on /MAVROS_latency.

Published:
  /MAVROS_latency     std_msgs/Float64              (one-way link latency estimate, ms)
  /bluerov/imu        sensor_msgs/Imu
  /bluerov/attitude   geometry_msgs/Vector3Stamped  (x=roll, y=pitch, z=yaw, degrees, ENU)
  /bluerov/heading    std_msgs/Float64              (compass, degrees, 0 = north)
  /bluerov/pressure   sensor_msgs/FluidPressure
  /bluerov/depth      std_msgs/Float64              (metres, from rel_alt; positive = down)
  /bluerov/battery    sensor_msgs/BatteryState
  /bluerov/gps        sensor_msgs/NavSatFix         (surface only)
  /bluerov/rc_in      mavros_msgs/RCIn
  /bluerov/rc_out     mavros_msgs/RCOut             (thruster/servo outputs)
  /bluerov/state      mavros_msgs/State
  /bluerov/altitude   mavros_msgs/Altitude
  /bluerov/velocity   geometry_msgs/TwistStamped    (only with a DVL / position source)

Run directly:
    python3 bluerov_sensor_node.py
Or with parameters:
    python3 bluerov_sensor_node.py --ros-args -p water_density:=1025.0
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Imu, FluidPressure, BatteryState, NavSatFix
from mavros_msgs.msg import RCIn, RCOut, State, Altitude
from mavros_msgs.srv import StreamRate, CommandLong
from std_msgs.msg import Float64
from geometry_msgs.msg import TwistStamped, Vector3Stamped
from rcl_interfaces.srv import ListParameters, SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType


# Best-effort subscribers connect to both best-effort and reliable publishers.
SUB_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)

PUB_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)

GRAVITY = 9.80665
MAV_CMD_REQUEST_MESSAGE = 512
MAVLINK_MSG_ID_SYSTEM_TIME = 2      # small message; cheap for the autopilot to send
SURFACE_CAL_SAMPLES = 20            # ~1 s of static_pressure at 20 Hz


def quat_to_rpy_deg(q):
    """Quaternion -> (roll, pitch, yaw) in degrees."""
    sinr = 2.0 * (q.w * q.x + q.y * q.z)
    cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr, cosr)

    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny, cosy)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


class BlueROVSensorNode(Node):

    def __init__(self):
        super().__init__("bluerov_sensor_node")

        # ---------------- Parameters ----------------
        # Prefix MAVROS topics live under. With mavros apm.launch in namespace
        # "mavros", topics appear as /mavros/mavros/state, /mavros/mavros/imu/data, ...
        self.declare_parameter("vehicle_ns", "mavros/mavros")
        self.declare_parameter("low_battery_threshold", 13.5)   # volts (4S pack)
        self.declare_parameter("water_density", 997.0)          # 997 fresh, 1025 sea
        self.declare_parameter("surface_pressure", 0.0)         # Pa; 0 = auto-zero at startup
        self.declare_parameter("stream_rate_hz", 10)            # 0 = don't request
        self.declare_parameter("status_period", 5.0)            # seconds
        self.declare_parameter("disable_timesync", True)        # stamp MAVROS data with Jetson time
        self.declare_parameter("latency_period", 1.0)           # seconds between latency probes; 0 = off
        self.declare_parameter("latency_timeout", 2.0)          # seconds before a probe counts as lost

        self._ns = self.get_parameter("vehicle_ns").value.strip("/")
        self._low_batt = float(self.get_parameter("low_battery_threshold").value)
        self._rho = float(self.get_parameter("water_density").value)
        self._p0 = float(self.get_parameter("surface_pressure").value)
        self._stream_hz = int(self.get_parameter("stream_rate_hz").value)
        self._status_period = float(self.get_parameter("status_period").value)
        self._disable_timesync = bool(self.get_parameter("disable_timesync").value)
        self._latency_period = float(self.get_parameter("latency_period").value)
        self._latency_timeout = float(self.get_parameter("latency_timeout").value)

        # ---------------- State cache ----------------
        self._connected = False
        self._armed = False
        self._mode = "UNKNOWN"
        self._rpy = None
        self._heading = None
        self._depth = None
        self._press_depth = None
        self._batt_v = None
        self._streams_requested = False
        self._timesync_done = False
        self._p0_samples = [] if self._p0 <= 0.0 else None

        # Latency probe state
        self._lat_pending = None          # (future, t0) while a probe is in flight
        self._lat_last_ms = None          # last one-way estimate
        self._lat_window = []             # one-way samples since last status report
        self._lat_lost = 0

        # Keep service clients alive while their calls are in flight
        self._param_clients = []

        # topic -> message count since last status report
        self._counts = {}

        # ---------------- Relays ----------------
        ns = self._ns
        self._relay(Imu,           f"/{ns}/imu/data",                     "/bluerov/imu",      self._on_imu)
        self._relay(FluidPressure, f"/{ns}/imu/static_pressure",          "/bluerov/pressure", self._on_pressure)
        self._relay(Float64,       f"/{ns}/global_position/rel_alt",      None,                self._on_rel_alt)
        self._relay(Float64,       f"/{ns}/global_position/compass_hdg",  "/bluerov/heading",  self._on_heading)
        self._relay(BatteryState,  f"/{ns}/battery",                      "/bluerov/battery",  self._on_battery)
        self._relay(NavSatFix,     f"/{ns}/global_position/global",       "/bluerov/gps")
        self._relay(RCIn,          f"/{ns}/rc/in",                        "/bluerov/rc_in")
        self._relay(RCOut,         f"/{ns}/rc/out",                       "/bluerov/rc_out")
        self._relay(State,         f"/{ns}/state",                        "/bluerov/state",    self._on_state)
        self._relay(Altitude,      f"/{ns}/altitude",                     "/bluerov/altitude")
        self._relay(TwistStamped,  f"/{ns}/local_position/velocity_body", "/bluerov/velocity")

        self._pub_attitude = self.create_publisher(Vector3Stamped, "/bluerov/attitude", PUB_QOS)
        self._pub_depth = self.create_publisher(Float64, "/bluerov/depth", PUB_QOS)
        self._pub_latency = self.create_publisher(Float64, "/MAVROS_latency", PUB_QOS)

        self._stream_cli = self.create_client(StreamRate, f"/{ns}/set_stream_rate")
        self._cmd_cli = self.create_client(CommandLong, f"/{ns}/cmd/command")

        self.create_timer(self._status_period, self._report_status)
        if self._latency_period > 0.0:
            self.create_timer(self._latency_period, self._probe_latency)

        self.get_logger().info(f"BlueROV2 sensor node started (MAVROS topics under /{ns}). "
                               f"Waiting for FCU connection...")

    # ------------------------------------------------------------------
    def _relay(self, msg_type, src, dst, extra=None):
        """Subscribe to src, optionally republish on dst, count messages, run extra(msg)."""
        pub = self.create_publisher(msg_type, dst, PUB_QOS) if dst else None
        self._counts[src] = 0

        def cb(msg):
            self._counts[src] += 1
            if pub is not None:
                pub.publish(msg)
            if extra is not None:
                extra(msg)

        self.create_subscription(msg_type, src, cb, SUB_QOS)

    # ---------------- Per-topic handlers ----------------
    def _on_state(self, msg: State):
        if msg.connected and not self._connected:
            self.get_logger().info("FCU CONNECTED")
            self._request_streams()
            self._configure_timesync()
        elif not msg.connected and self._connected:
            self.get_logger().warn("FCU DISCONNECTED")
        self._connected = msg.connected

        if msg.armed != self._armed or msg.mode != self._mode:
            self._armed, self._mode = msg.armed, msg.mode
            self.get_logger().info(f"Vehicle state | armed={self._armed} mode={self._mode}")

    def _on_imu(self, msg: Imu):
        self._rpy = quat_to_rpy_deg(msg.orientation)
        att = Vector3Stamped()
        att.header = msg.header
        att.vector.x, att.vector.y, att.vector.z = self._rpy
        self._pub_attitude.publish(att)

    def _on_heading(self, msg: Float64):
        self._heading = msg.data

    def _on_pressure(self, msg: FluidPressure):
        if self._p0_samples is not None:
            # Auto-zero: average the first readings as the surface reference.
            self._p0_samples.append(msg.fluid_pressure)
            if len(self._p0_samples) >= SURFACE_CAL_SAMPLES:
                self._p0 = sum(self._p0_samples) / len(self._p0_samples)
                self._p0_samples = None
                self.get_logger().info(f"Surface pressure zeroed at {self._p0:.1f} Pa "
                                       f"(start the node with the vehicle at the surface)")
            return
        self._press_depth = (msg.fluid_pressure - self._p0) / (self._rho * GRAVITY)

    def _on_rel_alt(self, msg: Float64):
        # ArduSub reports altitude relative to the surface (negative underwater).
        self._depth = -msg.data
        self._pub_depth.publish(Float64(data=self._depth))

    def _on_battery(self, msg: BatteryState):
        self._batt_v = msg.voltage
        if 0.5 < msg.voltage < self._low_batt:
            self.get_logger().warn(f"LOW BATTERY: {msg.voltage:.2f} V",
                                   throttle_duration_sec=10.0)

    # ---------------- Stream rates ----------------
    def _request_streams(self):
        if self._streams_requested or self._stream_hz <= 0:
            return
        if not self._stream_cli.service_is_ready():
            self.get_logger().warn("set_stream_rate service not available; topics may stay silent")
            return
        req = StreamRate.Request()
        req.stream_id = 0          # all streams
        req.message_rate = self._stream_hz
        req.on_off = True
        self._stream_cli.call_async(req)
        self._streams_requested = True
        self.get_logger().info(f"Requested all MAVLink streams at {self._stream_hz} Hz")

    # ---------------- Timesync -> Jetson clock ----------------
    def _configure_timesync(self):
        """Find MAVROS's timesync parameters and switch timesync off.

        The parameter names differ between MAVROS versions, so search every
        node under the MAVROS namespace for names ending in 'timesync_mode'
        (set to NONE) or 'timesync_rate' (set to 0).
        """
        if not self._disable_timesync or self._timesync_done:
            return
        self._timesync_done = True

        root = "/" + self._ns.split("/")[0]
        targets = []
        for name, node_ns in self.get_node_names_and_namespaces():
            fqn = (node_ns.rstrip("/") + "/" + name) if node_ns != "/" else "/" + name
            if fqn.startswith(root + "/") or node_ns.startswith(root):
                targets.append(fqn)

        if not targets:
            self.get_logger().warn(f"No MAVROS nodes found under {root}; timesync left on")
            return

        for fqn in targets:
            cli = self.create_client(ListParameters, f"{fqn}/list_parameters")
            self._param_clients.append(cli)
            if not cli.wait_for_service(timeout_sec=1.0):
                continue
            fut = cli.call_async(ListParameters.Request())
            fut.add_done_callback(lambda f, fqn=fqn: self._on_param_list(fqn, f))

    def _on_param_list(self, fqn, fut):
        try:
            names = fut.result().result.names
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"Could not list parameters on {fqn}: {exc}")
            return

        params = []
        for n in names:
            if n.endswith("timesync_mode"):
                params.append(Parameter(name=n, value=ParameterValue(
                    type=ParameterType.PARAMETER_STRING, string_value="NONE")))
            elif n.endswith("timesync_rate"):
                params.append(Parameter(name=n, value=ParameterValue(
                    type=ParameterType.PARAMETER_DOUBLE, double_value=0.0)))
        if not params:
            return

        cli = self.create_client(SetParameters, f"{fqn}/set_parameters")
        self._param_clients.append(cli)
        if not cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(f"set_parameters not available on {fqn}")
            return
        req = SetParameters.Request(parameters=params)
        fut2 = cli.call_async(req)
        fut2.add_done_callback(lambda f, fqn=fqn, params=params: self._on_param_set(fqn, params, f))

    def _on_param_set(self, fqn, params, fut):
        try:
            results = fut.result().results
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"Setting timesync parameters on {fqn} failed: {exc}")
            return
        for p, r in zip(params, results):
            if r.successful:
                self.get_logger().info(f"Timesync off: {fqn} {p.name} set "
                                       f"(MAVROS data now stamped with Jetson time)")
            else:
                self.get_logger().warn(f"Could not set {fqn} {p.name}: {r.reason}")

    # ---------------- Link latency ----------------
    def _probe_latency(self):
        if not self._connected or not self._cmd_cli.service_is_ready():
            return

        if self._lat_pending is not None:
            fut, t0 = self._lat_pending
            if time.monotonic() - t0 > self._latency_timeout:
                fut.cancel()
                self._lat_pending = None
                self._lat_lost += 1
            return  # one probe in flight at a time

        req = CommandLong.Request()
        req.broadcast = False
        req.command = MAV_CMD_REQUEST_MESSAGE
        req.confirmation = 0
        req.param1 = float(MAVLINK_MSG_ID_SYSTEM_TIME)

        t0 = time.monotonic()
        fut = self._cmd_cli.call_async(req)
        self._lat_pending = (fut, t0)
        fut.add_done_callback(lambda f, t0=t0: self._on_latency(f, t0))

    def _on_latency(self, fut, t0):
        rtt_ms = (time.monotonic() - t0) * 1000.0
        if self._lat_pending is None or self._lat_pending[0] is not fut:
            return  # timed out already
        self._lat_pending = None

        if fut.cancelled():
            return
        try:
            ok = fut.result().success
        except Exception:  # noqa: BLE001
            ok = False
        if not ok:
            self._lat_lost += 1
            return

        one_way = rtt_ms / 2.0
        self._lat_last_ms = one_way
        self._lat_window.append(one_way)
        self._pub_latency.publish(Float64(data=one_way))

    # ---------------- Status report ----------------
    def _report_status(self):
        def fmt(v, spec):
            return "--" if v is None else format(v, spec)

        rpy = ("--" if self._rpy is None
               else f"{self._rpy[0]:+.1f}/{self._rpy[1]:+.1f}/{self._rpy[2]:+.1f}")

        if self._lat_window:
            lat = (f"{sum(self._lat_window) / len(self._lat_window):.1f} ms avg "
                   f"(max {max(self._lat_window):.1f})")
        else:
            lat = "--"
        if self._lat_lost:
            lat += f", {self._lat_lost} lost"
        self._lat_window = []
        self._lat_lost = 0

        self.get_logger().info(
            f"[STATUS] connected={self._connected} armed={self._armed} mode={self._mode} | "
            f"heading={fmt(self._heading, '.1f')} deg | roll/pitch/yaw={rpy} deg | "
            f"depth(rel_alt)={fmt(self._depth, '.2f')} m depth(pressure)={fmt(self._press_depth, '.2f')} m | "
            f"battery={fmt(self._batt_v, '.2f')} V | latency={lat}"
        )

        parts = []
        for topic, n in self._counts.items():
            hz = n / self._status_period
            short = topic.replace(f"/{self._ns}/", "")
            parts.append(f"{short}={hz:.1f}Hz" if n else f"{short}=SILENT")
            self._counts[topic] = 0
        self.get_logger().info("[TOPICS] " + "  ".join(parts))


def main(args=None):
    rclpy.init(args=args)
    node = BlueROVSensorNode()
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
