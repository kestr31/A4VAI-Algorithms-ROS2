# Librarys

# Library for common
import numpy as np
import matplotlib.pyplot as plt
import cv2
# ROS libraries
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock

# ----------------------------------------------------------------------------------------#
# common ROS messages
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

# custom msgs libararies
from custom_msgs.msg import Heartbeat

# PX4 msgs libraries
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import VehicleAttitude

# ----------------------------------------------------------------------------------------#

from .init_variable_ca import *
from ..lib.common_fuctions import *


class CollisionAvoidanceTest(Node):
    def __init__(self):
        super().__init__("collision_avoidance_test")

        set_initial_variables(self)

        # ----------------------------------------------------------------------------------------#
        # region PUBLISHERS

        # publisher for vehicle command
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", self.qos_profile
        )

        # publisher for offboard control mode
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", self.qos_profile
        )

        # publisher for vehicle velocity setpoint
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", self.qos_profile
        )

        self.controller_heartbeat_publisher = self.create_publisher(
            Heartbeat, "/controller_heartbeat", 10
        )

        self.path_planning_heartbeat_publisher = self.create_publisher(
            Heartbeat, "/path_planning_heartbeat", 10
        )

        self.path_following_heartbeat_publisher = self.create_publisher(
            Heartbeat, "/path_following_heartbeat", 10
        )

        # end region
        # ----------------------------------------------------------------------------------------#

        # ----------------------------------------------------------------------------------------#
        # region SUBSCRIBERS

        # subscriber for vehicle local position
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position",
            self.vehicle_local_position_callback,
            self.qos_profile,
        )

        self.vehicle_attitude_subscriber = self.create_subscription(
            VehicleAttitude,
            "/fmu/out/vehicle_attitude",
            self.vehicle_attitude_callback,
            self.qos_profile,
        )

        self.CA_velocity_setpoint_subscriber_ = self.create_subscription(
            Twist, "/ca_vel_2_control", self.CA2Control_callback, 10
        )

        self.DepthSubscriber_ = self.create_subscription(
            Image, "/airsim_node/SimpleFlight/Depth_Camera_DepthPerspective/image", self.DepthCallback, 1
        )

        self.collision_avoidance_heartbeat_subscriber = self.create_subscription(
            Heartbeat,
            "/collision_avoidance_heartbeat",
            self.collision_avoidance_heartbeat_call_back,
            10,
        )

        # endregion
        # ----------------------------------------------------------------------------------------#

        # ----------------------------------------------------------------------------------------#
        # region TIMER

        period_offboard_control_mode = 0.2  # required 5Hz for offboard control (proof that the external controller is healthy
        self.offboard_main_timer = self.create_timer(
            period_offboard_control_mode, self.offboard_control_main
        )

        period_offboard_vel_ctrl = 0.02  # required 50Hz at least for velocity control
        self.velocity_control_call_timer = self.create_timer(
            period_offboard_vel_ctrl, self.publish_vehicle_velocity_setpoint
        )

        period_heartbeat_mode = 1
        self.heartbeat_timer = self.create_timer(
            period_heartbeat_mode, self.publish_collision_avoidance_heartbeat
        )
        self.heartbeat_timer = self.create_timer(
            period_heartbeat_mode, self.publish_path_planning_heartbeat
        )
        self.heartbeat_timer = self.create_timer(
            period_heartbeat_mode, self.publish_controller_heartbeat
        )

        # endregion
        # ----------------------------------------------------------------------------------------#

    # ----------------------------------------------------------------------------------------#
    # region MAIN CODE

    def offboard_control_main(self):

        # send offboard mode and arm mode command to px4
        if self.takeoff_start_time == self.offboard_initial_time and self.collision_avoidance_heartbeat == True:
            # offboard mode cmd to px4
            self.publish_vehicle_command(self.prm_offboard_mode)
            # arm cmd to px4
            self.publish_vehicle_command(self.prm_arm_mode)

        # takeoff after a certain period of time
        elif self.offboard_initial_time < self.takeoff_start_time:
            self.offboard_initial_time += 1

        # send offboard heartbeat signal to px4
        self.publish_offboard_control_mode(self.prm_off_con_mod)

        if self.initial_position_flag == False:
            self.veh_trj_set.pos_NED = self.initial_position
            self.takeoff()
        else:
            self.prm_off_con_mod.position = False
            self.prm_off_con_mod.velocity = True

            self.publish_offboard_control_mode(self.prm_off_con_mod)

        self.get_logger().info("-------------------------------------------------------")
        self.get_logger().info(f"position: {self.x}, {self.y}, {self.z}")
        self.get_logger().info(f"attitude: {self.phi*180/np.pi}, {self.theta*180/np.pi}, {self.psi*180/np.pi}")
        self.get_logger().info(f"heading: {self.heading*180/np.pi}")
        self.get_logger().info(f"yaw_cmd: {self.veh_trj_set.yaw_rad*180/np.pi}")
        self.get_logger().info("                            ")
        self.get_logger().info(f"obstacle_check: {self.obstacle_check}")
        self.get_logger().info(f"obstacle_flag: {self.obstacle_flag}")
        self.get_logger().info(f"vel_body_cmd: {np.array((self.DCM_nb @ self.veh_trj_set.vel_NED).tolist())}")
        self.get_logger().info(f"vel_ned__cmd: {self.veh_trj_set.vel_NED}")
        self.get_logger().info(f"yawspeed_cmd: {self.veh_trj_set.yaw_vel_rad}")
        self.get_logger().info("-------------------------------------------------------")
        self.state_logger()

    def takeoff(self):
        self.publish_position_setpoint()
        if abs(self.z - -(self.initial_position[2])) < 0.3:
            self.initial_position_flag = True


    # calculate yaw angle command
    def calculate_yaw_cmd_rad(self):
        dx = self.wp_x[self.cur_wp] - self.x
        dy = self.wp_y[self.cur_wp] - self.y
        self.yaw_cmd_rad = np.arctan2(dy, dx)  # [rad]

    # calculate velocity command
    def calculate_velocity_cmd(self):

        velocity = 2.0

        cos_yaw = velocity * np.cos(self.heading)
        sin_yaw = velocity * np.sin(self.heading)

        return cos_yaw, sin_yaw, 0.0

    # check waypoint
    def check_waypoint(self):
        # calculate distance to waypoint
        self.wp_distance = np.sqrt(
            (self.x - self.wp_x[self.cur_wp]) ** 2
            + (self.y - self.wp_y[self.cur_wp]) ** 2
        )

        if self.wp_distance < 0.5:
            self.cur_wp += 1
            if self.cur_wp == len(self.wp_x):
                self.cur_wp = 0

    def state_logger(self):
        self.get_logger().info("-----------------")
        # self.get_logger().info("sim_time =" + str(self.sim_time) )
        flightlog = "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %s %s \n" % (
            int(Clock().now().nanoseconds / 1000),
            self.obstacle_check, self.obstacle_flag, self.x, self.y, self.z,
            self.psi, self.theta, self.phi, 
            self.vel_body_cmd[0], self.vel_body_cmd[1], self.vel_body_cmd[2],
            self.veh_trj_set.vel_NED[0], self.veh_trj_set.vel_NED[1], self.veh_trj_set.vel_NED[2],
            str(self.veh_trj_set.vel_NED), str(self.veh_trj_set.yaw_vel_rad))
        
        self.flightlogFile.write(flightlog)

    # ----------------------------------------------------------------------------------------#
    # endregion
        # self.get_logger().info(f"position: {self.x}, {self.y}, {self.z}")
        # self.get_logger().info(f"attitude: {self.phi*180/np.pi}, {self.theta*180/np.pi}, {self.psi*180/np.pi}")
        # self.get_logger().info(f"heading: {self.heading*180/np.pi}")
        # self.get_logger().info(f"yaw_cmd: {self.veh_trj_set.yaw_rad*180/np.pi}")
        # self.get_logger().info("                            ")
        # self.get_logger().info(f"obstacle_check: {self.obstacle_check}")
        # self.get_logger().info(f"obstacle_flag: {self.obstacle_flag}")
        # self.get_logger().info(f"vel_body_cmd: {self.vel_body_cmd}")
        # self.get_logger().info(f"vel_ned__cmd: {self.veh_trj_set.vel_NED}")
        # self.get_logger().info(f"yawspeed_cmd: {self.veh_trj_set.yaw_vel_rad}")



    # ----------------------------------------------------------------------------------------#
    # region PUB FUNC

    # publish_vehicle_command
    def publish_vehicle_command(self, prm_veh_com):
        msg = VehicleCommand()
        msg.param1 = prm_veh_com.params[0]
        msg.param2 = prm_veh_com.params[1]
        msg.command = prm_veh_com.CMD_mode
        # values below are in [3]
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.vehicle_command_publisher.publish(msg)

    # publish position offboard command
    def publish_position_setpoint(self):
        msg = TrajectorySetpoint()
        msg.timestamp = int(Clock().now().nanoseconds / 1000)  # time in microseconds
        msg.position = self.veh_trj_set.pos_NED
        msg.yaw = self.veh_trj_set.yaw_rad
        self.trajectory_setpoint_publisher.publish(msg)

    # publish_vehicle_velocity_setpoint
    def publish_vehicle_velocity_setpoint(self):
        if self.initial_position_flag:
            msg = TrajectorySetpoint()
            msg.timestamp = int(
                Clock().now().nanoseconds / 1000
            )  # time in microseconds
            msg.position = [np.NaN, np.NaN, np.NaN]
            msg.acceleration = [np.NaN, np.NaN, np.NaN]
            msg.jerk = [np.NaN, np.NaN, np.NaN]

            if self.obstacle_check == False:


                
                msg.yaw = self.yaw_cmd_rad
                msg.yawspeed = np.NaN
                self.veh_trj_set.vel_NED = self.vel_ned_cmd_normal
                msg.velocity = np.float32(self.veh_trj_set.vel_NED)
            else:
                # msg.yaw = self.yaw_cmd_rad
                # msg.yawspeed = np.NaN
                # self.veh_trj_set.vel_NED = self.vel_ned_cmd_normal
                # msg.velocity = [np.NaN, np.NaN, np.NaN]


        
                self.veh_trj_set.vel_NED = self.total_ned_cmd
                # self.veh_trj_set.vel_NED = self.vel_ned_cmd_ca + self.vel_ned_cmd_normal

                
                self.veh_trj_set.yaw_rad = np.NaN
                self.veh_trj_set.yaw_vel_rad = self.collision_avoidance_yaw_vel_rad

                msg.velocity = np.float32([self.veh_trj_set.vel_NED[0], self.veh_trj_set.vel_NED[1], 0])
                msg.yaw = self.veh_trj_set.yaw_rad
                msg.yawspeed = self.veh_trj_set.yaw_vel_rad

            self.trajectory_setpoint_publisher.publish(msg)

    # publish offboard control mode
    def publish_offboard_control_mode(self, prm_off_con_mod):
        msg = OffboardControlMode()
        msg.position = prm_off_con_mod.position
        msg.velocity = prm_off_con_mod.velocity
        msg.acceleration = prm_off_con_mod.acceleration
        msg.attitude = prm_off_con_mod.attitude
        msg.body_rate = prm_off_con_mod.body_rate
        self.offboard_control_mode_publisher.publish(msg)

    # heartbeat publish
    def publish_collision_avoidance_heartbeat(self):
        msg = Heartbeat()
        msg.heartbeat = True
        self.path_following_heartbeat_publisher.publish(msg)

    def publish_path_planning_heartbeat(self):
        msg = Heartbeat()
        msg.heartbeat = True
        self.path_planning_heartbeat_publisher.publish(msg)

    def publish_controller_heartbeat(self):
        msg = Heartbeat()
        msg.heartbeat = True
        self.controller_heartbeat_publisher.publish(msg)

    # ----------------------------------------------------------------------------------------#
    # endregion

    # ----------------------------------------------------------------------------------------#
    # region CALLBACK FUNC

    # callback function for vehicle local position
    def vehicle_local_position_callback(self, msg):
        # update NED position
        self.x = msg.x
        self.y = msg.y
        self.z = -msg.z

        self.heading = msg.heading

        # check if the vehicle reached the waypoint
        self.check_waypoint()

        # calculate yaw angle command
        self.calculate_yaw_cmd_rad()

    # callback function for vehicle attitude
    def vehicle_attitude_callback(self, msg):
        self.phi, self.theta, self.psi = convert_quaternion2euler(
            msg.q[0], msg.q[1], msg.q[2], msg.q[3]
        )
        self.DCM_nb = DCM_from_euler_angle([self.phi, self.theta, self.psi])
        self.DCM_bn = np.transpose(self.DCM_nb)

        self.vel_ned_cmd_normal  = BodytoNED(self.vel_body_cmd_normal, 0, 0, self.DCM_bn)
        self.vel_body_cmd  = np.array((self.DCM_nb @ self.veh_trj_set.vel_NED).tolist())
        
        # self.DCM_body2ned = cal_DCM_body2ned(self.psi, self.theta, self.phi)

    def CA2Control_callback(self, msg):
        self.vel_cmd_body_x = msg.linear.x
        self.vel_cmd_body_y = msg.linear.y
        self.vel_cmd_body_z = msg.linear.z
        self.collision_avoidance_yaw_vel_rad = -msg.angular.z

        self_total_body_cmd = [self.vel_cmd_body_x + 2.0, self.vel_cmd_body_y, self.vel_cmd_body_z]
        if self_total_body_cmd[0] > 3.0:
            self_total_body_cmd[0] = 3.0
        if self_total_body_cmd[1] > 3.0:
            self_total_body_cmd[1] = 3.0
        if self_total_body_cmd[2] > 3.0:
            self_total_body_cmd[2] = 3.0

        self.total_ned_cmd  = BodytoNED(self_total_body_cmd[0], self_total_body_cmd[1], self_total_body_cmd[2], self.DCM_bn)


        self.vel_ned_cmd_ca  = BodytoNED(self.vel_cmd_body_x, self.vel_cmd_body_y, self.vel_cmd_body_z, self.DCM_bn)

    def DepthCallback(self, msg):
        try:
            # Convert the ROS Image message to OpenCV format
            self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            return
        
        self.valid_image = np.ones(self.image.shape)*12.0
        # np.savetxt('/home/user/workspace/ros2/ros2_ws/src/algorithm_test/algorithm_test/collosion_avoidance_unit_test/sample1.csv',self.valid_image,delimiter=",")
        valid_mask = (self.image <= 12)
        # np.savetxt('/home/user/workspace/ros2/ros2_ws/src/algorithm_test/algorithm_test/collosion_avoidance_unit_test/sample2.csv',valid_mask,delimiter=",")
        self.valid_image[valid_mask] = self.image[valid_mask]
        # np.savetxt('/home/user/workspace/ros2/ros2_ws/src/algorithm_test/algorithm_test/collosion_avoidance_unit_test/sample3.csv',self.valid_image,delimiter=",")

        self.min_distance = self.valid_image.min()

        if self.initial_position_flag:
            if self.min_distance < 7.0:
                self.obstacle_check = True
                np.savetxt('/home/user/workspace/ros2/ros2_ws/src/algorithm_test/algorithm_test/collosion_avoidance_unit_test/sample1.csv',self.valid_image,delimiter=",")
            else:
                self.obstacle_flag = False

            if self.obstacle_check == True and self.min_distance < 8.0:
                self.obstacle_flag = True
            else:
                self.obstacle_check = False
                self.obstacle_flag = False

    def collision_avoidance_heartbeat_call_back(self,msg):
        self.collision_avoidance_heartbeat = msg.heartbeat

    # endregion
    # ----------------------------------------------------------------------------------------#


def main(args=None):
    rclpy.init(args=args)
    path_planning_test = CollisionAvoidanceTest()
    rclpy.spin(path_planning_test)
    path_planning_test.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
