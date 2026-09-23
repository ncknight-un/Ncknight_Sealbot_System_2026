"""
bluerov_sensors.launch.py
--------------------------
Launches MAVROS (ArduSub config) + the BlueROV2 sensor collection node.

Usage:
    ros2 launch bluerov_sensors bluerov_sensors.launch.py

Override the FCU URL:
    ros2 launch bluerov_sensors bluerov_sensors.launch.py \
        fcu_url:=udp://:14550@192.168.2.2:14555
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ------------------------------------------------------------------
    # Launch arguments
    # ------------------------------------------------------------------
    fcu_url_arg = DeclareLaunchArgument(
        "fcu_url",
        # Standard BlueROV2 tether network: topside = 192.168.2.1
        #                                   ROV companion = 192.168.2.2
        default_value="udp://:14550@192.168.2.2:14555",
        description="MAVROS FCU connection URL",
    )

    gcs_url_arg = DeclareLaunchArgument(
        "gcs_url",
        default_value="",
        description="Optional GCS forward URL (e.g. QGroundControl)",
    )

    vehicle_ns_arg = DeclareLaunchArgument(
        "vehicle_ns",
        default_value="mavros",
        description="MAVROS ROS namespace",
    )

    low_batt_arg = DeclareLaunchArgument(
        "low_battery_threshold",
        default_value="13.5",
        description="Voltage (V) below which a warning is logged",
    )

    # ------------------------------------------------------------------
    # MAVROS node  (ArduSub / APM plugin set)
    # ------------------------------------------------------------------
    mavros_node = Node(
        package="mavros",
        executable="mavros_node",
        name="mavros",
        output="screen",
        parameters=[
            {
                "fcu_url": LaunchConfiguration("fcu_url"),
                "gcs_url": LaunchConfiguration("gcs_url"),
                "target_system_id": 1,
                "target_component_id": 1,
                "fcu_protocol": "v2.0",
                # Use the APM (ArduSub) plugin list
                "plugin_allowlist": [
                    "imu",
                    "battery",
                    "rc_io",
                    "global_position",
                    "altitude",
                    "local_position",
                    "sys_status",
                    "sys_time",
                ],
            }
        ],
    )

    # ------------------------------------------------------------------
    # BlueROV2 sensor collection node
    # ------------------------------------------------------------------
    sensor_node = Node(
        package="bluerov_sensors",
        executable="bluerov_sensor_node",
        name="bluerov_sensor_node",
        output="screen",
        parameters=[
            {
                "vehicle_ns": LaunchConfiguration("vehicle_ns"),
                "low_battery_threshold": LaunchConfiguration("low_battery_threshold"),
            }
        ],
    )

    # ------------------------------------------------------------------
    # Optional: auto-start ros2 bag recording
    # Uncomment the ExecuteProcess block below to enable.
    # ------------------------------------------------------------------
    # bag_record = ExecuteProcess(
    #     cmd=[
    #         "ros2", "bag", "record",
    #         "-o", "/tmp/bluerov_session",
    #         "/bluerov/imu",
    #         "/bluerov/pressure",
    #         "/bluerov/battery",
    #         "/bluerov/gps",
    #         "/bluerov/depth",
    #         "/bluerov/rc_in",
    #         "/bluerov/rc_out",
    #         "/bluerov/altitude",
    #         "/bluerov/velocity",
    #         "/mavros/state",
    #     ],
    #     output="screen",
    # )

    return LaunchDescription(
        [
            fcu_url_arg,
            gcs_url_arg,
            vehicle_ns_arg,
            low_batt_arg,
            mavros_node,
            sensor_node,
            # bag_record,   # ← uncomment to auto-bag
        ]
    )
