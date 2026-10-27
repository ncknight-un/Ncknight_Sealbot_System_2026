"""
bluerov_sensors.launch.py
--------------------------
Launches MAVROS (ArduSub, via mavros/apm.launch) + the BlueROV2 sensor node.

Usage:
    ros2 launch bluerov_sensors bluerov_sensors.launch.py

Override the FCU URL:
    ros2 launch bluerov_sensors bluerov_sensors.launch.py fcu_url:=udp://0.0.0.0:14551@

Run only the sensor node (MAVROS already running elsewhere):
    ros2 launch bluerov_sensors bluerov_sensors.launch.py start_mavros:=false

Network (standard BlueROV2 tether):
    topside = 192.168.2.1, BlueOS/Pi = 192.168.2.2, Jetson = 192.168.2.3
    In BlueOS > MAVLink Endpoints, add a UDP Client endpoint to 192.168.2.3:14551.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    args = [
        DeclareLaunchArgument(
            "fcu_url",
            default_value="udp://0.0.0.0:14551@",
            description="MAVROS FCU URL (listen on 14551 for the BlueOS UDP endpoint)"),
        DeclareLaunchArgument(
            "gcs_url", default_value="",
            description="Optional GCS forward URL (e.g. QGroundControl)"),
        DeclareLaunchArgument(
            "mavros_ns", default_value="mavros",
            description="Namespace MAVROS is launched in"),
        DeclareLaunchArgument(
            "mavros_topic_prefix", default_value="mavros/mavros",
            description="Prefix MAVROS topics actually appear under "
                        "(namespace + MAVROS node name), e.g. /mavros/mavros/state"),
        DeclareLaunchArgument(
            "start_mavros", default_value="true",
            description="Also start MAVROS (false if it is already running)"),
        DeclareLaunchArgument(
            "low_battery_threshold", default_value="13.5",
            description="Voltage (V) below which a warning is logged"),
        DeclareLaunchArgument(
            "water_density", default_value="997.0",
            description="kg/m^3: 997 fresh water, 1025 sea water"),
        DeclareLaunchArgument(
            "stream_rate_hz", default_value="10",
            description="MAVLink stream rate requested on connect (0 = don't request)"),
        DeclareLaunchArgument(
            "fast_rate_hz", default_value="100.0",
            description="Rate (Hz) for IMU, heading, depth and pressure messages; 0 = use stream_rate_hz"),
        DeclareLaunchArgument(
            "surface_pressure", default_value="0.0",
            description="Surface reference pressure in Pa (0 = auto-zero at startup)"),
        DeclareLaunchArgument(
            "disable_timesync", default_value="false",
            description="Turn off MAVROS timesync (data then stamped on arrival, not when measured)"),
        DeclareLaunchArgument(
            "latency_period", default_value="1.0",
            description="Seconds between /MAVROS_latency probes (0 = off)"),
    ]

    # ------------------------------------------------------------------
    # MAVROS: reuse the stock ArduPilot launch so plugin lists and
    # config match what you get from `ros2 launch mavros apm.launch`.
    # ------------------------------------------------------------------
    mavros = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(get_package_share_directory("mavros"), "launch", "apm.launch")),
        launch_arguments={
            "fcu_url": LaunchConfiguration("fcu_url"),
            "gcs_url": LaunchConfiguration("gcs_url"),
            "namespace": LaunchConfiguration("mavros_ns"),
            "tgt_system": "1",
            "tgt_component": "1",
        }.items(),
        condition=IfCondition(LaunchConfiguration("start_mavros")),
    )

    # ------------------------------------------------------------------
    # BlueROV2 sensor node
    # ------------------------------------------------------------------
    sensor_node = Node(
        package="bluerov_sensors",
        executable="bluerov_sensor_node",
        name="bluerov_sensor_node",
        output="screen",
        parameters=[{
            "vehicle_ns": LaunchConfiguration("mavros_topic_prefix"),
            "low_battery_threshold": ParameterValue(
                LaunchConfiguration("low_battery_threshold"), value_type=float),
            "water_density": ParameterValue(
                LaunchConfiguration("water_density"), value_type=float),
            "stream_rate_hz": ParameterValue(
                LaunchConfiguration("stream_rate_hz"), value_type=int),
            "fast_rate_hz": ParameterValue(
                LaunchConfiguration("fast_rate_hz"), value_type=float),
            "surface_pressure": ParameterValue(
                LaunchConfiguration("surface_pressure"), value_type=float),
            "disable_timesync": ParameterValue(
                LaunchConfiguration("disable_timesync"), value_type=bool),
            "latency_period": ParameterValue(
                LaunchConfiguration("latency_period"), value_type=float),
        }],
    )

    return LaunchDescription(args + [mavros, sensor_node])
