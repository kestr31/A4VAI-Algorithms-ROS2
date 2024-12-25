import numpy as np

from rclpy.clock import Clock
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import VehicleAttitudeSetpoint
from px4_msgs.msg import TrajectorySetpoint

from custom_msgs.msg import LocalWaypointSetpoint
from custom_msgs.msg import GlobalWaypointSetpoint

from std_msgs.msg import Float32
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Bool

class PubFuncPX4:
    def __init__(self, node):
        self.node = node

    # publish offboard control mode
    def publish_offboard_control_mode(self, offboard_mode):
        msg = OffboardControlMode()
        msg.position        = offboard_mode.position
        msg.velocity        = offboard_mode.velocity
        msg.acceleration    = offboard_mode.acceleration
        msg.attitude        = offboard_mode.attitude
        msg.body_rate       = offboard_mode.body_rate
        self.node.offboard_control_mode_publisher.publish(msg)

    # publish_vehicle_command
    def publish_vehicle_command(self, modes):
        msg = VehicleCommand()
        msg.param1  = modes.params[0]
        msg.param2  = modes.params[1]
        msg.param3  = modes.params[2]
        msg.command = modes.CMD_mode
        # values below are in [3]
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.node.vehicle_command_publisher.publish(msg)

    # publish attitude offboard command
    def publish_vehicle_attitude_setpoint(self, mode_flag, veh_att_set):
        if mode_flag.is_pf:
            msg = VehicleAttitudeSetpoint()
            msg.timestamp = int(Clock().now().nanoseconds / 1000)  # time in microseconds
            msg.roll_body           = veh_att_set.roll_body
            msg.pitch_body          = veh_att_set.pitch_body
            msg.yaw_body            = veh_att_set.yaw_body
            msg.yaw_sp_move_rate    = veh_att_set.yaw_sp_move_rate
            msg.q_d[0]              = veh_att_set.q_d[0]
            msg.q_d[1]              = veh_att_set.q_d[1]
            msg.q_d[2]              = veh_att_set.q_d[2]
            msg.q_d[3]              = veh_att_set.q_d[3]
            msg.thrust_body[0]      = 0.0
            msg.thrust_body[1]      = 0.0
            msg.thrust_body[2]      = veh_att_set.thrust_body[2]
            self.node.vehicle_attitude_setpoint_publisher.publish(msg)

    def publish_vehicle_velocity_setpoint(self, mode_flag, veh_vel_set):
        if mode_flag.is_ca == True or mode_flag.is_manual == True:
            msg = TrajectorySetpoint()
            msg.timestamp = int(Clock().now().nanoseconds / 1000)  # time in microseconds
            msg.position        = veh_vel_set.position
            msg.acceleration    = veh_vel_set.acceleration
            msg.jerk            = veh_vel_set.jerk
            msg.velocity        = np.float32([veh_vel_set.ned_velocity[0], veh_vel_set.ned_velocity[1], 0])
            msg.yaw             = veh_vel_set.yaw
            msg.yawspeed        = veh_vel_set.yawspeed
            self.node.trajectory_setpoint_publisher.publish(msg)

class PubFuncWaypoint:
    def __init__(self, node):
        self.node = node
        self.guid_var = node.guid_var

    # publish local waypoint
    def local_waypoint_publish(self, plag):
        msg = LocalWaypointSetpoint()
        msg.path_planning_complete = plag
        print(plag)
        msg.waypoint_x = self.guid_var.waypoint_x
        msg.waypoint_y = self.guid_var.waypoint_y
        msg.waypoint_z = self.guid_var.waypoint_z
        self.node.local_waypoint_publisher.publish(msg)

    def global_waypoint_publish(self, publisher):
        msg = GlobalWaypointSetpoint()
        msg.start_point = [self.guid_var.waypoint_x[0], self.guid_var.waypoint_z[0], self.guid_var.waypoint_y[0]]
        msg.goal_point = [self.guid_var.waypoint_x[-1], self.guid_var.waypoint_z[-1], self.guid_var.waypoint_y[-1]]
        print('debug: global_waypoint_publish')
        publisher.publish(msg)

class PubFuncHeartbeat:
    def __init__(self, node):
        self.node = node

    def publish_collision_avoidance_heartbeat(self):
        msg = Bool()
        msg.data = True
        self.node.collision_avoidance_heartbeat_publisher.publish(msg)

    def publish_path_planning_heartbeat(self):
        msg = Bool()
        msg.data = True
        self.node.path_planning_heartbeat_publisher.publish(msg)

    def publish_controller_heartbeat(self):
        msg = Bool()
        msg.data = True
        self.node.controller_heartbeat_publisher.publish(msg)

    def publish_path_following_heartbeat(self):
        msg = Bool()
        msg.data = True
        self.node.path_following_heartbeat_publisher.publish(msg)

class PubFuncPlotter:
    def __init__(self, node):
        self.node = node

    def publish_heading(self, state_var):
        msg = Float32()
        msg.data = float(state_var.yaw)
        self.node.heading_publisher_to_plotter.publish(msg)

    def publish_control_mode(self, mode_flag):
        msg = Bool()
        if mode_flag.is_ca == True:
            msg.data = True
        else:
            msg.data = False
        self.node.control_mode_publisher_to_plotter.publish(msg)

    def publish_obstacle_min_distance(self, ca_var):
        msg = Float64MultiArray()
        msg.data = [float(ca_var.depth_min_distance), float(ca_var.lidar_min_distance)]
        self.node.min_distance_publisher_to_plotter.publish(msg)

    def publish_global_waypoint_to_plotter(self, guid_var):
        msg = GlobalWaypointSetpoint()
        msg.start_point = [guid_var.init_pos[0], guid_var.init_pos[0], guid_var.init_pos[0]]
        msg.goal_point = [guid_var.waypoint_x[-1], guid_var.waypoint_z[-1], guid_var.waypoint_y[-1]]
        self.node.global_waypoint_publisher_to_plotter.publish(msg)

    def publish_local_waypoint_to_plotter(self, guid_var):
        msg = LocalWaypointSetpoint()
        msg.path_planning_complete = True
        msg.waypoint_x = guid_var.real_wp_x
        msg.waypoint_y = guid_var.real_wp_y
        msg.waypoint_z = guid_var.real_wp_z
        self.node.local_waypoint_publisher_to_plotter.publish(msg)