import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/sealbot/Sealbot_ROS/ros_ws/install/camera_bringup'
