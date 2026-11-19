"""Camera + GPU AprilTag in one container.

CPU rectify (mono8) -> GPU rgb8 -> GPU AprilTag. All camera settings, including
the image size the GPU converter allocates for, come from the camera parameter
file (camera_bringup/config/sealbot_camera.yaml by default).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes
from launch_ros.descriptions import ComposableNode

from camera_bringup.params import default_params_file, image_size, load_camera_params

CONTAINER = '/vision_container'
INTRA = [{'use_intra_process_comms': True}]


def _launch_setup(context):
    cam_params = load_camera_params(LaunchConfiguration('camera_params_file').perform(context))
    width, height = image_size(cam_params)
    tag_params = os.path.join(
        get_package_share_directory('apriltag_bringup'), 'config', 'isaac_tags.yaml')

    camera = LoadComposableNodes(
        target_container=CONTAINER,
        condition=IfCondition(LaunchConfiguration('use_camera')),
        composable_node_descriptions=[
            ComposableNode(
                package='vimbax_camera', plugin='vimbax_camera::VimbaXCameraNode',
                name='sealbot_camera', namespace='camera',
                parameters=[cam_params],
                remappings=[
                    ('/camera/sealbot_camera/image_raw', '/camera/image_raw'),
                    ('/camera/sealbot_camera/camera_info', '/camera/camera_info'),
                ],
                extra_arguments=INTRA,
            ),
        ],
    )

    apriltag = LoadComposableNodes(
        target_container=CONTAINER,
        condition=IfCondition(LaunchConfiguration('use_apriltag')),
        composable_node_descriptions=[
            ComposableNode(
                package='image_proc', plugin='image_proc::RectifyNode', name='rectify',
                parameters=[{'interpolation': 1}],
                remappings=[
                    ('image', '/camera/image_raw'),
                    ('camera_info', '/camera/camera_info'),
                    ('image_rect', '/camera/image_rect_mono'),
                ],
                extra_arguments=INTRA,
            ),
            ComposableNode(
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
                name='to_rgb8',
                parameters=[{
                    'encoding_desired': 'rgb8',
                    'image_width': width,
                    'image_height': height,
                }],
                remappings=[
                    ('image_raw', '/camera/image_rect_mono'),
                    ('image', '/camera/image_rect'),
                ],
            ),
            # tag_detections -> /detections to keep naming consistent with early testing
            ComposableNode(
                package='isaac_ros_apriltag',
                plugin='nvidia::isaac_ros::apriltag::AprilTagNode', name='apriltag',
                parameters=[tag_params],
                remappings=[
                    ('image', '/camera/image_rect'),
                    ('camera_info', '/camera/camera_info'),
                    ('tag_detections', '/detections'),
                ],
            ),
        ],
    )
    return [camera, apriltag]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_camera', default_value='true'),
        DeclareLaunchArgument('use_apriltag', default_value='true'),
        DeclareLaunchArgument(
            'camera_params_file', default_value=default_params_file(),
            description='Camera parameter YAML (selects settings and calibration files)'),
        # One shared container for camera + perception
        ComposableNodeContainer(
            name='vision_container', namespace='/',
            package='rclcpp_components', executable='component_container_mt',
            output='screen',
        ),
        OpaqueFunction(function=_launch_setup),
    ])
