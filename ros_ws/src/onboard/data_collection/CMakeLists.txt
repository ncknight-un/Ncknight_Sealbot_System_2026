cmake_minimum_required(VERSION 3.8)
project(data_collection)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

add_executable(panel_calibration src/panel_calibration.cpp)
target_compile_features(panel_calibration PUBLIC cxx_std_17)
# Default output is this package's source config/ folder, so the calibration
# survives rebuilds and gets installed by the next colcon build
target_compile_definitions(panel_calibration PRIVATE
  DATA_COLLECTION_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(panel_calibration PRIVATE ${EIGEN3_INCLUDE_DIRS})
ament_target_dependencies(panel_calibration rclcpp tf2_msgs)

add_executable(tag_zero_frames src/tag_zero_frames.cpp)
target_compile_features(tag_zero_frames PUBLIC cxx_std_17)
ament_target_dependencies(tag_zero_frames rclcpp tf2_ros geometry_msgs)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(DIRECTORY launch urdf meshes config DESTINATION share/${PROJECT_NAME})

install(TARGETS panel_calibration tag_zero_frames
  DESTINATION lib/${PROJECT_NAME})

ament_package()
