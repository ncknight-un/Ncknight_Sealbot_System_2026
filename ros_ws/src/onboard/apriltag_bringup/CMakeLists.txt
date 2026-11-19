cmake_minimum_required(VERSION 3.8)
project(apriltag_bringup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

# isaac_tag_bridge: Isaac ROS AprilTag -> apriltag_ros-style /tf + /detections.
# Only built where Isaac ROS is installed (the Jetson); the CPU launch files
# are installed everywhere.
find_package(isaac_ros_apriltag_interfaces QUIET)
if(isaac_ros_apriltag_interfaces_FOUND)
  find_package(rclcpp REQUIRED)
  find_package(apriltag_msgs REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(tf2 REQUIRED)
  find_package(tf2_ros REQUIRED)
  add_executable(isaac_tag_bridge src/isaac_tag_bridge.cpp)
  target_compile_features(isaac_tag_bridge PUBLIC cxx_std_17)
  ament_target_dependencies(isaac_tag_bridge
    rclcpp apriltag_msgs geometry_msgs tf2 tf2_ros isaac_ros_apriltag_interfaces)
  install(TARGETS isaac_tag_bridge DESTINATION lib/${PROJECT_NAME})
else()
  message(WARNING "isaac_ros_apriltag_interfaces not found: skipping isaac_tag_bridge "
                  "(only the CPU apriltag pipeline will work on this machine)")
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/compare_calibrations.py DESTINATION lib/${PROJECT_NAME})

ament_package()
