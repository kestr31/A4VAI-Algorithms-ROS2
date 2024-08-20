# Library
# library for ros2
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from datetime import datetime

import math
import numpy as np

from px4_msgs.msg import VehicleCommand, OffboardControlMode , TrajectorySetpoint, VehicleAttitudeSetpoint
from px4_msgs.msg import VehicleLocalPosition , VehicleAttitude, VehicleAngularVelocity, VehicleStatus, VehicleGlobalPosition

from custom_msgs.msg import LocalWaypointSetpoint, ConveyLocalWaypointComplete
from custom_msgs.msg import Heartbeat
from geometry_msgs.msg import Twist 
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool

from .give_global_waypoint import GiveGlobalWaypoint
from .commonFcn import *
from .initVar import *
from sensor_msgs_py import point_cloud2

class Controller(Node):
    def __init__(self):
        super().__init__('controller')

        setInitialVariables(self)

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # region SUBSCRIBERS
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # region FROM PX4-AUTOPILOT
        # -----------------------------------------------------------------------------------------------------------------
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.vehicle_local_position_callback,
            self.qos_profile
        )
        self.vehicle_attitude_subscriber = self.create_subscription(
            VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self.vehicle_attitude_callback,
            self.qos_profile
        )
        self.vehicle_angular_velocity_subscriber = self.create_subscription(
            VehicleAngularVelocity,
            '/fmu/out/vehicle_angular_velocity',
            self.vehicle_angular_velocity_callback,
            self.qos_profile
        )
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            self.vehicle_status_callback,
            self.qos_profile
        )
        self.agl_subscriber = self.create_subscription(
            VehicleGlobalPosition,
            '/fmu/out/vehicle_global_position',
            self.agl_callback,
            self.qos_profile
        )
        # endregion
        # -----------------------------------------------------------------------------------------------------------------


        # region FROM ALGORITHM 1: PATH PLANNER
        # -----------------------------------------------------------------------------------------------------------------
        self.local_waypoint_subscriber = self.create_subscription(
            LocalWaypointSetpoint,\
            '/local_waypoint_setpoint_from_PP',\
            self.path_planning_call_back,\
            10)
        self.path_planning_heartbeat_subscriber = self.create_subscription(
            Heartbeat,
            '/path_planning_heartbeat',
            self.path_planning_heartbeat_call_back,
            10
        )
        # endregion
        # -----------------------------------------------------------------------------------------------------------------
        

        # region FROM ALGORITHM 2: PATH FOLLOWER
        # -----------------------------------------------------------------------------------------------------------------
        self.PF_attitude_setpoint_subscriber_ = self.create_subscription(
            VehicleAttitudeSetpoint,
            '/pf_att_2_control',
            self.PF_Att2Control_callback,
            10
        )
        self.convey_local_waypoint_complete_subscriber = self.create_subscription(
            ConveyLocalWaypointComplete,
            '/convey_local_waypoint_complete',
            self.convey_local_waypoint_complete_call_back,
            10
        ) 
        self.path_following_heartbeat_subscriber = self.create_subscription(
            Heartbeat,
            '/path_following_heartbeat',
            self.path_following_heartbeat_call_back,
            10
        )
        # endregion
        # -----------------------------------------------------------------------------------------------------------------


        # region FROM ALGORITHM 3: COLLISION AVOIDANCE
        # -----------------------------------------------------------------------------------------------------------------
        self.CA_velocity_setpoint_subscriber_ = self.create_subscription(
            Twist,
            '/ca_vel_2_control',
            self.CA2Control_callback,
            10
        )
        self.collision_avoidance_heartbeat_subscriber = self.create_subscription(
            Heartbeat,
            '/collision_avoidance_heartbeat',
            self.collision_avoidance_heartbeat_call_back,
            10
        )
        # endregion
        # -----------------------------------------------------------------------------------------------------------------


        # region MISCELLANEOUS SUBSRIBERS
        # -----------------------------------------------------------------------------------------------------------------

        # self.DepthSubscriber_ = self.create_subscription(
        #     Image,
        #     '/depth/raw',
        #     self.DepthCallback,
        #     1
        #     )
        
        self.LidarSubscriber_ = self.create_subscription(
            PointCloud2,
            '/airsim_node/SimpleFlight/lidar/RPLIDAR_A3',
            self.LidarCallback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        )
        # endregion
        # -----------------------------------------------------------------------------------------------------------------
        # endregion
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # region PUBLISHERS
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # region TO PX4-AUTOPILOT
        # -----------------------------------------------------------------------------------------------------------------
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            self.qos_profile
        )
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            self.qos_profile
        )
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            self.qos_profile
        )
        self.vehicle_attitude_setpoint_publisher = self.create_publisher(
            VehicleAttitudeSetpoint,
            '/fmu/in/vehicle_attitude_setpoint',
            self.qos_profile
        )
        # endregion
        # -----------------------------------------------------------------------------------------------------------------


        # region TO ALGORITHM 2: PATH FOLLOWER
        # -----------------------------------------------------------------------------------------------------------------
        self.local_waypoint_publisher = self.create_publisher(
            LocalWaypointSetpoint,
            '/local_waypoint_setpoint_to_PF',
            10
        )
        self.heartbeat_publisher = self.create_publisher(
            Heartbeat,
            '/controller_heartbeat',
            10
        )
        self.waypoint_convert_flag_publisher = self.create_publisher(
            Bool,
            '/waypoint_convert_flag',
            10
        )
        # endregion
        # -----------------------------------------------------------------------------------------------------------------
        # endregion
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # region TIMERS
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # algorithm timer
        period_heartbeat_mode  =   1        
        self.heartbeat_timer   =   self.create_timer(period_heartbeat_mode, self.publish_heartbeat)

        period_offboard_control_mode =   0.2         # required about 5Hz for attitude control (proof that the external controller is healthy
        self.offboard_main_timer     =   self.create_timer(period_offboard_control_mode, self.offboard_control_main)

        period_offboard_att_ctrl  =   0.004        # required 250Hz at least for attitude control
        self.attitude_control_call_timer =  self.create_timer(period_offboard_att_ctrl, self.publisher_vehicle_attitude_setpoint)

        period_offboard_vel_ctrl  =   0.02         # required 50Hz at least for velocity control
        self.velocity_control_call_timer =  self.create_timer(period_offboard_vel_ctrl, self.publish_vehicle_velocity_setpoint)

        timer_period = 0.1
        self.collision_avidance_timer   = self.create_timer(timer_period, self.timer_callback)
        # endregion
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # main code
    def offboard_control_main(self):
        # check another module nodes alive
        if self.path_following_heartbeat == True and self.path_planning_heartbeat == True and self.collision_avoidance_heartbeat == True:
            # send offboard mode and arm mode command to px4
            if self.offboard_setpoint_counter == self.offboard_start_flight_time :
                # offboard mode cmd to px4
                self.publish_vehicle_command(self.prm_offboard_mode)
                # arm cmd to px4
                self.publish_vehicle_command(self.prm_arm_mode)
    
            # takeoff after a certain period of time
            elif self.offboard_setpoint_counter <= self.offboard_start_flight_time:
                self.offboard_setpoint_counter += 1
    
            # send offboard heartbeat signal to px4 
            self.publish_offboard_control_mode(self.prm_off_con_mod)
    
            # check initial position
            if self.initial_position_flag == True:
                
                # stay initial position untill transmit local waypoint to path following is complete
                if self.convey_local_waypoint_is_complete == False:
                    self.takeoff_and_go_initial_position()
    
                    # check path planning complete
                    if self.path_planning_complete == False:
                    
                        # give global waypoint to path planning and path planning start
                        give_global_waypoint = GiveGlobalWaypoint()
                        give_global_waypoint.global_waypoint_publish(self.start_point, self.goal_point)
                        give_global_waypoint.destroy_node()
                    else:
                        # send local waypoint to pathfollowing
                        self.local_waypoint_publish()
                    
                # do path following if local waypoint transmit to path following is complete 
                else:
                    if self.obstacle_flag == False:

                        self.collision_avoidance_flag   =   False
                        self.prm_off_con_mod.position   =   False
                        self.prm_off_con_mod.velocity   =   False
                        self.prm_off_con_mod.attitude   =   True
                        self.publish_offboard_control_mode(self.prm_off_con_mod)
                        self.path_following_flag        =   True
                        
                        self.collision_avoidance_end_timer    =   True
                        
                    else:
                        self.collision_avoidance_flag   =   True
                        self.path_following_flag        =   False
                        self.prm_off_con_mod.position   =   False
                        self.prm_off_con_mod.velocity   =   True
                        self.prm_off_con_mod.attitude   =   False
                        self.publish_offboard_control_mode(self.prm_off_con_mod)
                        
                        self.collision_avoidance_end_timer   =   False
                        self.collision_avoidance_timer_running    =   False

                        if self.collision_avidance_elapsed_time < 6:
                            self.publish_waypoint_convert_flag()                        
                        
                        
    
            # go initial position if not in initial position 
            else:
                self.veh_trj_set.pos_NED    =   self.initial_position
                self.takeoff_and_go_initial_position()
        else:
            pass

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # region callback Functions
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # region 1. path planning callback Functions
    # -----------------------------------------------------------------------------------------------------------------

    # subscribe local waypoint and path planning complete flag
    def path_planning_call_back(self, msg):
        self.path_planning_complete = msg.path_planning_complete 
        self.waypoint_x             = msg.waypoint_x 
        self.waypoint_y             = msg.waypoint_y
        self.waypoint_z             = msg.waypoint_z
        print("                                          ")
        print("=====   Path Planning Complete!!     =====")
        print("                                          ")

    # heartbeat subscribe from path planning
    def path_planning_heartbeat_call_back(self,msg):
        self.path_planning_heartbeat = msg.heartbeat

    # endregion
    # -----------------------------------------------------------------------------------------------------------------

    # region 2. path following callback Functions
    # -----------------------------------------------------------------------------------------------------------------

    # subscribe convey local waypoint complete flag from path following
    def convey_local_waypoint_complete_call_back(self, msg):
        self.convey_local_waypoint_is_complete = msg.convey_local_waypoint_is_complete

    # heartbeat subscribe from path following
    def path_following_heartbeat_call_back(self,msg):
        self.path_following_heartbeat = msg.heartbeat

    # update attitude offboard command from path following
    def PF_Att2Control_callback(self, msg):
        self.veh_att_set.roll_body          =   msg.roll_body
        self.veh_att_set.pitch_body         =   msg.pitch_body
        self.veh_att_set.yaw_body           =   msg.yaw_body
        self.veh_att_set.yaw_sp_move_rate   =   msg.yaw_sp_move_rate
        self.veh_att_set.q_d[0]             =   msg.q_d[0]
        self.veh_att_set.q_d[1]             =   msg.q_d[1]
        self.veh_att_set.q_d[2]             =   msg.q_d[2]
        self.veh_att_set.q_d[3]             =   msg.q_d[3]
        self.veh_att_set.thrust_body[0]     =   msg.thrust_body[0]
        self.veh_att_set.thrust_body[1]     =   msg.thrust_body[1]
        self.veh_att_set.thrust_body[2]     =   msg.thrust_body[2]
    
    # endregion
    # -----------------------------------------------------------------------------------------------------------------

    # region 3. collision avoidance callback Functions
    # -----------------------------------------------------------------------------------------------------------------

    def collision_avoidance_heartbeat_call_back(self,msg):
        self.collision_avoidance_heartbeat = msg.heartbeat
    
    def CA2Control_callback(self, msg):
        # update Body velocity CMD
        # self.veh_trj_set.vel_NED[0]            =   msg.linear.x
        # self.veh_trj_set.vel_NED[1]            =   msg.linear.y
        # self.veh_trj_set.vel_NED[2]            =   msg.linear.z
        # self.veh_trj_set.yaw_vel_rad   =   msg.angular.z

        self.vel_cmd_body_x            =   msg.linear.x
        self.vel_cmd_body_y            =   msg.linear.y
        self.vel_cmd_body_z            =   msg.linear.z
        self.veh_trj_set.yaw_vel_rad   =   msg.angular.z

        self.DCM_nb = DCM(self.phi, self.theta, self.psi)
        self.DCM_bn = np.transpose(self.DCM_nb)
        BodytoNED(self)

    # endregion
    # -----------------------------------------------------------------------------------------------------------------
    
    # region 4. px4 callback Functions
    # -----------------------------------------------------------------------------------------------------------------

    # update position and velocity
    def vehicle_local_position_callback(self, msg):
        # update NED position 
        self.x      =   msg.x
        self.y      =   msg.y
        self.z      =   msg.z
        # update NED velocity
        self.v_x    =   msg.vx
        self.v_y    =   msg.vy
        self.v_z    =   msg.vz

    # update attitude
    def vehicle_attitude_callback(self, msg):
        self.psi , self.theta, self.phi     =   Quaternion2Euler(self, msg.q[0], msg.q[1], msg.q[2], msg.q[3])
     
    # update body angular velocity
    def vehicle_angular_velocity_callback(self, msg):
        self.p    =   msg.xyz[0]
        self.q    =   msg.xyz[1]
        self.r    =   msg.xyz[2]
    # update vehicle status
    def vehicle_status_callback(self, vehicle_status):
        self.vehicle_status = vehicle_status

    # ?????
    def agl_callback(self,msg):
        self.agl = msg.alt_ellipsoid

    # endregion
    # -----------------------------------------------------------------------------------------------------------------
    
    # region 5. MISCELLANEOUS callback Functions
    # -----------------------------------------------------------------------------------------------------------------
    
    def LidarCallback(self, pc_msg):
        if pc_msg.is_dense is True :
            
            input =point_cloud2.read_points(pc_msg)

            points_list = list(input)

            point = np.array(points_list, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

            x = point['x']
            y = point['y']
            dist = np.sqrt(x **2 + y ** 2)
            self.min_dist = np.min(dist)
            
            if self.z < -2.0:
                if self.min_dist < 5.0:
                    self.obstacle_check = True
                else : 
                    self.obstacle_flag = False
                    
                if self.obstacle_check == True and self.min_dist < 7.0:
                    self.obstacle_flag = True
                else :
                    self.obstacle_check = False
                    self.obstacle_flag  = False
            else :
                self.obstacle_flag = False
        else : 
            pass

    # endregion
    # -----------------------------------------------------------------------------------------------------------------
    # endregion
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # region PUBLISHERS Functions
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # region 1. PUBLISHERS Functions to path following
    # -----------------------------------------------------------------------------------------------------------------

    # publish local waypoint
    def local_waypoint_publish(self):
        msg = LocalWaypointSetpoint()
        msg.path_planning_complete = self.path_planning_complete
        msg.waypoint_x             = self.waypoint_x
        msg.waypoint_y             = self.waypoint_y
        msg.waypoint_z             = (np.add(self.waypoint_z, float(self.agl-5))).tolist()
        self.local_waypoint_publisher.publish(msg)

    # publish waypoint convert flag
    def publish_waypoint_convert_flag(self):
        msg = Bool()
        msg.data = True
        self.waypoint_convert_flag_publisher.publish(msg)
        self.get_logger().info("------------------------------------------------------------------------------")
        self.get_logger().info("                                                                              ")
        self.get_logger().info("                                                                              ")
        self.get_logger().info("                                                                              ")
        self.get_logger().info("skip Next waypoint, elapsed time =" + str(self.collision_avidance_elapsed_time) )
        self.get_logger().info("                                                                              ")
        self.get_logger().info("                                                                              ")
        self.get_logger().info("                                                                              ")
        self.get_logger().info("------------------------------------------------------------------------------")
        self.collision_avidance_elapsed_time = 10
        self.collision_avoidance_timer_running = False
    
    # endregion
    # -----------------------------------------------------------------------------------------------------------------

    # region 2. PUBLISHERS Functions to px4
    # -----------------------------------------------------------------------------------------------------------------

    # publish_vehicle_command
    def publish_vehicle_command(self, prm_veh_com):
        msg                 =   VehicleCommand()
        msg.param1          =   prm_veh_com.params[0]
        msg.param2          =   prm_veh_com.params[1]
        msg.command         =   prm_veh_com.CMD_mode
        # values below are in [3]
        msg.target_system   =   1
        msg.target_component=   1
        msg.source_system   =   1
        msg.source_component=   1
        msg.from_external   =   True
        self.vehicle_command_publisher.publish(msg)

    # publish offboard control mode
    def publish_offboard_control_mode(self, prm_off_con_mod):
        msg                 =   OffboardControlMode()
        msg.position        =   prm_off_con_mod.position
        msg.velocity        =   prm_off_con_mod.velocity
        msg.acceleration    =   prm_off_con_mod.acceleration
        msg.attitude        =   prm_off_con_mod.attitude
        msg.body_rate       =   prm_off_con_mod.body_rate
        self.offboard_control_mode_publisher.publish(msg)

    # publish position offboard command
    def publish_position_setpoint(self,veh_trj_set):
        msg                 =   TrajectorySetpoint()
        msg.timestamp = int(Clock().now().nanoseconds / 1000) # time in microseconds
        msg.position        =   veh_trj_set.pos_NED
        msg.yaw             =   veh_trj_set.yaw_rad
        self.trajectory_setpoint_publisher.publish(msg)

    # publish velocity offboard command
    def publish_vehicle_velocity_setpoint(self):
        if self.collision_avoidance_flag == True:
            msg                 =   TrajectorySetpoint()
            msg.timestamp = int(Clock().now().nanoseconds / 1000) # time in microseconds
            msg.position        =   [np.NaN, np.NaN, np.NaN]
            msg.yaw             =   np.NaN
            msg.velocity        =   np.float32(self.veh_trj_set.vel_NED)
            msg.yawspeed        =   self.veh_trj_set.yaw_vel_rad
            self.trajectory_setpoint_publisher.publish(msg)
   
    # publish attitude offboard command
    def publisher_vehicle_attitude_setpoint(self):
        if self.path_following_flag == True:
            msg                     =   VehicleAttitudeSetpoint()
            msg.roll_body           =   self.veh_att_set.roll_body
            msg.pitch_body          =   self.veh_att_set.pitch_body
            msg.yaw_body            =   self.veh_att_set.yaw_body
            msg.yaw_sp_move_rate    =   self.veh_att_set.yaw_sp_move_rate
            msg.q_d[0]              =   self.veh_att_set.q_d[0]
            msg.q_d[1]              =   self.veh_att_set.q_d[1]
            msg.q_d[2]              =   self.veh_att_set.q_d[2]
            msg.q_d[3]              =   self.veh_att_set.q_d[3]
            msg.thrust_body[0]      =   0.
            msg.thrust_body[1]      =   0.
            msg.thrust_body[2]      =   self.veh_att_set.thrust_body[2]
            self.vehicle_attitude_setpoint_publisher.publish(msg)
        else:
            pass

    # endregion
    # -----------------------------------------------------------------------------------------------------------------
    
    # region 3. MISCELLANEOUS PUBLISHERS Functions
    # -----------------------------------------------------------------------------------------------------------------

    # publish controller heartbeat signal to another module's nodes
    def publish_heartbeat(self):
        msg = Heartbeat()
        msg.heartbeat = True
        self.heartbeat_publisher.publish(msg)

    # endregion
    # -----------------------------------------------------------------------------------------------------------------
    # endregion
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # region MISCELLANEOUS Functions
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    def timer_callback(self):
        if self.collision_avoidance_end_timer is True and self.collision_avoidance_timer_running is False:
            self.current_time = datetime.now()
            self.collision_avoidance_timer_running = True
        if self.collision_avoidance_timer_running is True:
            self.collision_avidance_elapsed_time = (datetime.now() - self.current_time).total_seconds()

    # offboard control toward initial point
    def takeoff_and_go_initial_position(self):
        self.publish_position_setpoint(self.veh_trj_set)
        if abs(self.z - self.initial_position[2]) < 0.3:
            self.initial_position_flag = True

    def state_logger(self):
        # self.get_logger().info("-----------------")
        # # self.get_logger().info("sim_time =" + str(self.sim_time) )
        # self.get_logger().info("path_planning_heartbeat =" + str(self.path_planning_heartbeat) )
        # self.get_logger().info("path_following_heartbeat =" + str(self.path_following_heartbeat) )
        # self.get_logger().info("collision_avoidance_heartbeat =" + str(self.collision_avoidance_heartbeat) )
        # self.get_logger().info("obstacle_flag =" + str(self.obstacle_flag) )
        # self.get_logger().info("path_following_flag =" + str(self.path_following_flag) )
        # self.get_logger().info("collision_avoidance_flag =" + str(self.collision_avoidance_flag) )
        # self.get_logger().info("initial_position_flag =" + str(self.initial_position_flag) )
        # self.get_logger().info("NED Position:   [x]=" + str(self.x) +", [e]=" + str(self.y) +", [d]=" + str(self.z))
        # self.get_logger().info("NED Velocity:   [v_x]=" + str(self.v_x) +", [v_y]=" + str(self.v_y) +", [v_z]=" + str(self.v_z))
        # self.get_logger().info("Body Velocity:   [u]=" + str(self.u) +", [v]=" + str(self.v) +", [w]=" + str(self.w))
        # self.get_logger().info("Euler Angle:   [psi]=" + str(self.psi) +", [theta]=" + str(self.theta) +", [phi]=" + str(self.phi))
        # self.get_logger().info("Angular Velocity:   [p]=" + str(self.p) +", [q]=" + str(self.q) +", [r]=" + str(self.r))
        # self.get_logger().info("Body Velocity CMD:   [cmd_x]=" + str(self.vel_cmd_body_x) +", [cmd_y]=" + str(self.vel_cmd_body_y) +", [cmd_z]=" + str(self.vel_cmd_body_z))
        # self.get_logger().info("min_dist =" + str(self.min_dist) )
        # flightlog = "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n" %(
        #     self.obstacle_flag, self.path_following_flag, self.collision_avoidance_flag, self.x, self.y, 
        #     self.z, self.v_x, self.v_y, self.v_z, self.u, self.v, self.w, self.psi, 
        #     self.theta, self.phi, self.p, self.q, self.r, 
        #     self.vel_cmd_body_x,  self.vel_cmd_body_y,  self.vel_cmd_body_z,
        #     self.min_dist)
        # self.flightlogFile.write(flightlog)
        pass
        # self.datalogFile.close()

        # endregion
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



def main(args=None):
    print("======================================================")
    print("------------- main() in controller.py ----------------")
    print("======================================================")
    rclpy.init(args=args)
    controller = Controller()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()
    pass
if __name__ == '__main__':
    main()
