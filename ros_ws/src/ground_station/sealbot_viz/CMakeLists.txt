cmake_minimum_required(VERSION 3.8)
project(sealbot_viz)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(apriltag_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(ping360_sonar_msgs REQUIRED)

add_executable(tag_viz_node src/tag_viz_node.cpp)
target_compile_features(tag_viz_node PUBLIC cxx_std_17)
ament_target_dependencies(tag_viz_node rclcpp visualization_msgs apriltag_msgs)

add_executable(arena_viz_node src/arena_viz_node.cpp)
target_compile_features(arena_viz_node PUBLIC cxx_std_17)
ament_target_dependencies(arena_viz_node rclcpp visualization_msgs)

add_executable(sonar_cloud_node src/sonar_cloud_node.cpp)
target_compile_features(sonar_cloud_node PUBLIC cxx_std_17)
ament_target_dependencies(sonar_cloud_node rclcpp sensor_msgs ping360_sonar_msgs)

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

install(TARGETS tag_viz_node arena_viz_node sonar_cloud_node
  DESTINATION lib/${PROJECT_NAME})
 
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME})

ament_package()
