"""Standalone Vimba X camera node, configured from camera_bringup/config/sealbot_camera.yaml."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from camera_bringup.params import default_params_file, load_camera_params


def _launch_setup(context):
    params = load_camera_params(LaunchConfiguration('camera_params_file').perform(context))
    return [
        Node(
            package='vimbax_camera', executable='vimbax_camera_node',
            name='sealbot_camera', namespace='camera', output='screen',
            parameters=[params],
            remappings=[
                ('~/image_raw', '/camera/image_raw'),
                ('~/camera_info', '/camera/camera_info'),
            ],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_params_file', default_value=default_params_file(),
            description='Camera parameter YAML (selects settings and calibration files)'),
        OpaqueFunction(function=_launch_setup),
    ])
