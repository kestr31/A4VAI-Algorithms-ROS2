cmake_minimum_required(VERSION 3.5)
project(custom_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)

set(msg_files
  "msg/CustomMsg.msg"
  "msg/WaypointIndex.msg"
)

set(srv_files
  "srv/WaypointSetpoint.srv"
)


rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  DEPENDENCIES builtin_interfaces
)


ament_export_dependencies(rosidl_default_runtime)
ament_package()

