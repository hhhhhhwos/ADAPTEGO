cmake_minimum_required(VERSION 3.0.2)
project(adaptive_planning)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  quadrotor_msgs
  traj_utils
  plan_env
)

find_package(Eigen3 REQUIRED)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES adaptive_planning
 CATKIN_DEPENDS roscpp rospy std_msgs geometry_msgs nav_msgs sensor_msgs quadrotor_msgs traj_utils plan_env
 DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

add_executable(weight_adapter_node src/weight_adapter_node.cpp)
target_link_libraries(weight_adapter_node ${catkin_LIBRARIES})

add_executable(safety_supervisor_node src/safety_supervisor_node.cpp)
target_link_libraries(safety_supervisor_node ${catkin_LIBRARIES})
