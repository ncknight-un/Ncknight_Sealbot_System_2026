#!/usr/bin/env python3
"""
BlueROV2 MAVROS Sensor Data Node  —  ROS 2 Humble
--------------------------------------------------
Subscribes to MAVROS topics published by ArduSub and re-publishes them
under /bluerov/ so they can be captured cleanly with `ros2 bag record`.

Topics collected:
  IMU            : /mavros/imu/data
  Pressure/Depth : /mavros/imu/static_pressure
                   /mavros/global_position/rel_alt  (depth from Bar30)
  Battery        : /mavros/battery
  GPS (surface)  : /mavros/global_position/global
  RC in/out      : /mavros/rc/in   /mavros/rc/out
  State          : /mavros/state
  Altitude       : /mavros/altitude
  Velocity       : /mavros/local_position/velocity_body

Run:
    ros2 run bluerov_sensors bluerov_sensor_node

Or via launch file:
    ros2 launch bluerov_sensors bluerov_sensors.launch.py
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import Imu, FluidPressure, BatteryState, NavSatFix
from mavros_msgs.msg import RCIn, RCOut, State, Altitude
from std_msgs.msg import Float64
from geometry_msgs.msg import TwistStamped


# QoS that matches MAVROS publishers (best-effort, volatile)
MAVROS_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)

# QoS for our re-published topics (reliable, keep last)
PUB_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


class BlueROVSensorNode(Node):

    def __init__(self):
        super().__init__("bluerov_sensor_node")

        # ---------------------------------------------------------------
        # Parameters  (set in launch file or via --ros-args -p name:=val)
        # ---------------------------------------------------------------
        self.declare_parameter("vehicle_ns", "mavros")
        self.declare_parameter("low_battery_threshold", 13.5)

        ns  = self.get_parameter("vehicle_ns").get_parameter_value().string_value
        self._low_batt = (
            self.get_parameter("low_battery_threshold")
            .get_parameter_value()
            .double_value
        )

        self.get_logger().info(f"BlueROV2 sensor node starting (namespace: /{ns}) …")

        # ---------------------------------------------------------------
        # Subscribers  (MAVROS → this node)
        # ---------------------------------------------------------------
        self.create_subscription(
            Imu,           f"/{ns}/imu/data",                    self.cb_imu,      MAVROS_QOS)
        self.create_subscription(
            FluidPressure, f"/{ns}/imu/static_pressure",         self.cb_pressure, MAVROS_QOS)
        self.create_subscription(
            BatteryState,  f"/{ns}/battery",                     self.cb_battery,  MAVROS_QOS)
        self.create_subscription(
            NavSatFix,     f"/{ns}/global_position/global",      self.cb_gps,      MAVROS_QOS)
        self.create_subscription(
            Float64,       f"/{ns}/global_position/rel_alt",     self.cb_depth,    MAVROS_QOS)
        self.create_subscription(
            RCIn,          f"/{ns}/rc/in",                       self.cb_rc_in,    MAVROS_QOS)
        self.create_subscription(
            RCOut,         f"/{ns}/rc/out",                      self.cb_rc_out,   MAVROS_QOS)
        self.create_subscription(
            State,         f"/{ns}/state",                       self.cb_state,    10)
        self.create_subscription(
            Altitude,      f"/{ns}/altitude",                    self.cb_altitude, MAVROS_QOS)
        self.create_subscription(
            TwistStamped,  f"/{ns}/local_position/velocity_body",self.cb_velocity, MAVROS_QOS)

        # ---------------------------------------------------------------
        # Re-publishers  → /bluerov/<topic>
        # ---------------------------------------------------------------
        self._pub_imu      = self.create_publisher(Imu,          "/bluerov/imu",      PUB_QOS)
        self._pub_pressure = self.create_publisher(FluidPressure, "/bluerov/pressure", PUB_QOS)
        self._pub_battery  = self.create_publisher(BatteryState,  "/bluerov/battery",  PUB_QOS)
        self._pub_gps      = self.create_publisher(NavSatFix,     "/bluerov/gps",      PUB_QOS)
        self._pub_depth    = self.create_publisher(Float64,       "/bluerov/depth",    PUB_QOS)
        self._pub_rc_in    = self.create_publisher(RCIn,          "/bluerov/rc_in",    PUB_QOS)
        self._pub_rc_out   = self.create_publisher(RCOut,         "/bluerov/rc_out",   PUB_QOS)
        self._pub_altitude = self.create_publisher(Altitude,      "/bluerov/altitude", PUB_QOS)
        self._pub_velocity = self.create_publisher(TwistStamped,  "/bluerov/velocity", PUB_QOS)

        # ---------------------------------------------------------------
        # Internal state cache
        # ---------------------------------------------------------------
        self._armed   = False
        self._mode    = "UNKNOWN"
        self._depth_m = 0.0

        # Throttle timers (track last log time per topic)
        self._last_batt_log  = self.get_clock().now()
        self._last_depth_log = self.get_clock().now()

        self.get_logger().info("BlueROV2 sensor node ready — collecting data.")

    # -------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------

    def cb_imu(self, msg: Imu):
        self._pub_imu.publish(msg)

    def cb_pressure(self, msg: FluidPressure):
        self._pub_pressure.publish(msg)
        depth_approx = (msg.fluid_pressure - 101325.0) / (1025.0 * 9.80665)
        self.get_logger().debug(
            f"Pressure: {msg.fluid_pressure:.1f} Pa  (~{depth_approx:.2f} m)"
        )

    def cb_battery(self, msg: BatteryState):
        self._pub_battery.publish(msg)

        # Throttle normal log to every 10 s
        now = self.get_clock().now()
        if (now - self._last_batt_log).nanoseconds > 10e9:
            self.get_logger().info(
                f"Battery | {msg.voltage:.2f} V  "
                f"{msg.current:.2f} A  "
                f"{msg.percentage * 100:.1f} %"
            )
            self._last_batt_log = now

        if msg.voltage < self._low_batt:
            self.get_logger().warn(f"LOW BATTERY: {msg.voltage:.2f} V !")

    def cb_gps(self, msg: NavSatFix):
        self._pub_gps.publish(msg)
        self.get_logger().debug(
            f"GPS | lat={msg.latitude:.6f}  lon={msg.longitude:.6f}"
        )

    def cb_depth(self, msg: Float64):
        self._pub_depth.publish(msg)
        self._depth_m = msg.data

        now = self.get_clock().now()
        if (now - self._last_depth_log).nanoseconds > 2e9:
            self.get_logger().info(f"Depth: {self._depth_m:.3f} m")
            self._last_depth_log = now

    def cb_rc_in(self, msg: RCIn):
        self._pub_rc_in.publish(msg)

    def cb_rc_out(self, msg: RCOut):
        self._pub_rc_out.publish(msg)

    def cb_state(self, msg: State):
        if msg.armed != self._armed or msg.mode != self._mode:
            self._armed = msg.armed
            self._mode  = msg.mode
            self.get_logger().info(
                f"Vehicle state | armed={self._armed}  mode={self._mode}"
            )

    def cb_altitude(self, msg: Altitude):
        self._pub_altitude.publish(msg)

    def cb_velocity(self, msg: TwistStamped):
        self._pub_velocity.publish(msg)
        self.get_logger().debug(
            f"Vel | vx={msg.twist.linear.x:.3f}  "
            f"vy={msg.twist.linear.y:.3f}  "
            f"vz={msg.twist.linear.z:.3f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = BlueROVSensorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
