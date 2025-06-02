cmake_minimum_required(VERSION 2.8.3)
project(gpr_node)

set(CMAKE_CXX_FLAGS "-std=c++14 -O3 -Wall")

find_package(PCL 1.7 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  visualization_msgs
  cv_bridge
  pcl_ros
  pcl_conversions
  plan_env
  # active_perception
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES gpr_node
  CATKIN_DEPENDS
    roscpp rospy std_msgs visualization_msgs cv_bridge
    pcl_ros pcl_conversions 
    gazebo_radiation_plugins 
    plan_env
    # active_perception
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

add_library(gpr_node
  src/gpr.cpp
)

add_executable(gpr_node_node
  src/gpr_node.cpp
)
target_link_libraries(gpr_node_node
  gpr_node
  # active_perception
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)
