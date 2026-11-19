cmake_minimum_required(VERSION 3.8)
project(apriltag_bringup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

# isaac_tag_bridge: Isaac ROS AprilTag -> apriltag_ros-style /tf + /detections,
# with corner refinement + PnP. Built as a component (loaded into vision_container)
# plus a standalone `isaac_tag_bridge` executable. Only built where Isaac ROS is
# installed (the Jetson); the CPU launch files are installed everywhere.
find_package(isaac_ros_apriltag_interfaces QUIET)
if(isaac_ros_apriltag_interfaces_FOUND)
  find_package(rclcpp REQUIRED)
  find_package(rclcpp_components REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(apriltag_msgs REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(tf2 REQUIRED)
  find_package(tf2_ros REQUIRED)
  find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d)

  add_library(isaac_tag_bridge_component SHARED src/isaac_tag_bridge.cpp)
  target_compile_features(isaac_tag_bridge_component PUBLIC cxx_std_17)
  target_include_directories(isaac_tag_bridge_component SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
  ament_target_dependencies(isaac_tag_bridge_component
    rclcpp rclcpp_components sensor_msgs apriltag_msgs geometry_msgs tf2 tf2_ros)
  # Link only the C++ message targets. ament_target_dependencies would also pull in
  # the prebuilt package's __rosidl_generator_py target, whose exported numpy
  # include path (/usr/local/.../numpy/core/include) doesn't exist on the Jetson.
  target_link_libraries(isaac_tag_bridge_component
    ${OpenCV_LIBS}
    isaac_ros_apriltag_interfaces::isaac_ros_apriltag_interfaces__rosidl_generator_cpp
    isaac_ros_apriltag_interfaces::isaac_ros_apriltag_interfaces__rosidl_typesupport_cpp)

  rclcpp_components_register_node(isaac_tag_bridge_component
    PLUGIN "sealbot::IsaacTagBridge"
    EXECUTABLE isaac_tag_bridge)

  install(TARGETS isaac_tag_bridge_component
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)
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
