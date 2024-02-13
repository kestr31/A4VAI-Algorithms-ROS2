# A4VAI Algorithms in ROS2 Package



# 1. Build ROS2 Package

px4_ros_ws/src 경로에 git clone

```
$ colcon build --symlink-install --packages-select a4vai
```
```
$ colcon build --symlink-install --packages-select a4vai
```

# 2. Run Node

```
$ ros2 run a4vai Plan2WP  
```
```
$ ros2 run a4vai node_MPPI_output 
```
```
$ ros2 run a4vai node_att_ctrl  
```
```
$ ros2 run a4vai controller
```