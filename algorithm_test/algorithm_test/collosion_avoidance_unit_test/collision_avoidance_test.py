# Librarys

# Library for common
import numpy as np
import os

# ROS libraries
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
# ----------------------------------------------------------------------------------------#
# PX4 msgs libraries
from px4_msgs.msg import TrajectorySetpoint

from ..lib.timer import HeartbeatTimer, MainTimer, CommandPubTimer
from ..lib.subscriber import PX4Subscriber, CmdSubscriber, HeartbeatSubscriber, MissionSubscriber
from ..lib.publisher import PX4Publisher, HeartbeatPublisher, PlotterPublisher
from ..lib.publish_function import PubFuncHeartbeat, PubFuncPX4, PubFuncWaypoint, PubFuncPlotter
from ..lib.common_fuctions import set_initial_variables, state_logger, publish_to_plotter, BodytoNED
# ----------------------------------------------------------------------------------------#
class CollisionAvoidanceTest(Node):
    def __init__(self):
        super().__init__("collision_avoidance_test")
        # ----------------------------------------------------------------------------------------#
        # region INITIALIZE
        dir = os.path.dirname(os.path.abspath(__file__))
        sim_name = "ca_unit_test"
        set_initial_variables(self, dir, sim_name)

        self.offboard_mode.velocity = True

        self.yaw_cmd_rad = 0.0
        self.vel_ned_cmd_normal = np.zeros(3)
        # endregion
        # ----------------------------------------------------------------------------------------#
        # region PUBLISHERS
        self.pub_px4 = PX4Publisher(self)
        self.pub_px4.declareVehicleCommandPublisher()
        self.pub_px4.declareOffboardControlModePublisher()
        self.pub_px4.declareTrajectorySetpointPublisher()

        self.pub_heartbeat = HeartbeatPublisher(self)
        self.pub_heartbeat.declareControllerHeartbeatPublisher()
        self.pub_heartbeat.declarePathFollowingHeartbeatPublisher()
        self.pub_heartbeat.declarePathPlanningHeartbeatPublisher()

        self.pub_plotter = PlotterPublisher(self)
        self.pub_plotter.declareGlobalWaypointPublisherToPlotter()
        self.pub_plotter.declareLocalWaypointPublisherToPlotter()
        self.pub_plotter.declareHeadingPublisherToPlotter()
        self.pub_plotter.declareStatePublisherToPlotter()
        self.pub_plotter.declareMinDistancePublisherToPlotter()
        # end region
        # ----------------------------------------------------------------------------------------#
        # region SUBSCRIBERS
        self.sub_px4 = PX4Subscriber(self)
        self.sub_px4.declareVehicleLocalPositionSubscriber(self.state_var)
        self.sub_px4.declareVehicleAttitudeSubscriber(self.state_var)

        self.sub_cmd = CmdSubscriber(self)
        self.sub_cmd.declareCAVelocitySetpointSubscriber(self.veh_vel_set, self.state_var)

        self.sub_mission = MissionSubscriber(self)
        self.sub_mission.declareLidarSubscriber(self.mode_flag, self.ca_var)
        self.sub_mission.declareDepthSubscriber(self.mode_flag, self.ca_var)

        self.sub_hearbeat = HeartbeatSubscriber(self)
        self.sub_hearbeat.declareCollisionAvoidanceHeartbeatSubscriber(self.offboard_var)
        # endregion
        # ----------------------------------------------------------------------------------------#
        # region PUB FUNC
        self.pub_func_heartbeat = PubFuncHeartbeat(self)
        self.pub_func_px4       = PubFuncPX4(self)
        self.pub_func_waypoint  = PubFuncWaypoint(self)
        self.pub_func_plotter   = PubFuncPlotter(self)
        # endregion
        # ----------------------------------------------------------------------------------------#
        # region TIMER
        self.timer_offboard_control = MainTimer(self, self.offboard_var)
        self.timer_offboard_control.declareOffboardControlTimer(self.offboard_control_main)

        self.timer_cmd = CommandPubTimer(self, self.offboard_var)
        # self.timer_cmd.declareOffboardVelocityControlTimer(self.mode_flag, self.veh_vel_set, self.pub_func_px4)
        self.velocity_control_call_timer = self.create_timer(
            self.offboard_var.period_offboard_vel_ctrl,
            self.publish_vehicle_velocity_setpoint
        )

        self.timer_heartbeat = HeartbeatTimer(self, self.offboard_var, self.pub_func_heartbeat)
        self.timer_heartbeat.declareControllerHeartbeatTimer()
        self.timer_heartbeat.declarePathPlanningHeartbeatTimer()
        self.timer_heartbeat.declarePathFollowingHeartbeatTimer()
        # endregion
    # ----------------------------------------------------------------------------------------#
    # region MAIN CODE
    def offboard_control_main(self):
        if self.offboard_var.ca_heartbeat == True:
            # send offboard mode and arm mode command to px4
            if self.offboard_var.counter == self.offboard_var.flight_start_time and self.mode_flag.is_takeoff == False:
                # arm cmd to px4
                self.pub_func_px4.publish_vehicle_command(self.modes.prm_arm_mode)
                # offboard mode cmd to px4
                self.pub_func_px4.publish_vehicle_command(self.modes.prm_takeoff_mode)

            # takeoff after a certain period of time
            elif self.offboard_var.counter <= self.offboard_var.flight_start_time:
                self.offboard_var.counter += 1

            # check if the vehicle is ready to initial position
            if self.mode_flag.is_takeoff == False and self.state_var.z > self.guid_var.init_pos[2]:
                self.mode_flag.is_takeoff = True
                self.get_logger().info('Vehicle is reached to initial position')
            
            # if the vehicle was taken off send local waypoint to path following and wait in position mode
            if self.mode_flag.is_takeoff:
                publish_to_plotter(self)
                # check if the vehicle reached the waypoint
                self.check_waypoint()
                # calculate yaw angle command
                self.calculate_yaw_cmd_rad()
                # calculate ned velocity command
                self.calculate_velocity_cmd()

                self.pub_func_px4.publish_offboard_control_mode(self.offboard_mode)
                self.pub_func_px4.publish_vehicle_command(self.modes.prm_offboard_mode)
                
        
        state_logger(self)
    # endregion
    # ----------------------------------------------------------------------------------------#
    # region CALCULATION FUNC
    # calculate yaw angle command
    def calculate_yaw_cmd_rad(self):
        dx = self.guid_var.waypoint_x[self.guid_var.cur_wp] - self.state_var.x
        dy = self.guid_var.waypoint_y[self.guid_var.cur_wp] - self.state_var.y
        self.yaw_cmd_rad = np.arctan2(dy, dx)  # [rad]

    # calculate ned velocity command
    def calculate_velocity_cmd(self):
        normal_body_velocity_cmd = np.array([2, 0, 0])
        self.vel_ned_cmd_normal  = BodytoNED(normal_body_velocity_cmd, self.state_var.dcm_b2n)

    # check waypoint
    def check_waypoint(self):
        # calculate distance to waypoint
        self.guid_var.wp_distance = np.sqrt(
            (self.state_var.x - self.guid_var.waypoint_x[self.guid_var.cur_wp]) ** 2
            + (self.state_var.y - self.guid_var.waypoint_y[self.guid_var.cur_wp]) ** 2
        )
        if self.guid_var.wp_distance < 0.5:
            self.guid_var.cur_wp += 1
            if self.guid_var.cur_wp == len(self.guid_var.waypoint_x):
                self.guid_var.cur_wp = 0
    # endregion
    # ----------------------------------------------------------------------------------------#
    # region PUB FUNC
    # publish_vehicle_velocity_setpoint
    def publish_vehicle_velocity_setpoint(self):
        if self.mode_flag.is_takeoff:
            msg = TrajectorySetpoint()
            msg.timestamp = int(Clock().now().nanoseconds / 1000)  # time in microseconds
            msg.position        = self.veh_vel_set.position
            msg.acceleration    = self.veh_vel_set.acceleration
            msg.jerk            = self.veh_vel_set.jerk
            if self.mode_flag.is_ca:
                msg.velocity        = np.float32(self.vel_ned_cmd_normal)
                msg.yaw             = self.yaw_cmd_rad
                msg.yawspeed        = np.NaN
            else:
                msg.velocity        = np.float32(self.veh_vel_set.ned_velocity)
                msg.yaw             = np.NaN
                msg.yawspeed        = self.veh_vel_set.yawspeed
            self.trajectory_setpoint_publisher.publish(msg)
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
