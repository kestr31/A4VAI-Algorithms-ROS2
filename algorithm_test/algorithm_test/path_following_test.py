# Librarys

# Library for common
import numpy as np
import matplotlib.pyplot as plt

# ROS libraries
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

# custom msgs libararies
from custom_msgs.msg import LocalWaypointSetpoint, ConveyLocalWaypointComplete
from custom_msgs.msg import LocalWaypointSetpoint
from custom_msgs.msg import GlobalWaypointSetpoint
from custom_msgs.msg import Heartbeat

from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import VehicleAttitudeSetpoint

class PathPlanningTest(Node):
    def __init__(self):
        super().__init__("give_global_waypoint")


        #.. parameter - offboard control mode
        class prm_msg_off_con_mod:
            def __init__(self):        
                self.position        =   False
                self.velocity        =   False
                self.acceleration    =   False
                self.attitude        =   False
                self.body_rate       =   False

        self.prm_off_con_mod            =   prm_msg_off_con_mod()
        self.prm_off_con_mod.attitude   =   True

        #.. parameter - vehicle command 
        class prm_msg_veh_com:
            def __init__(self):
                self.CMD_mode   =   np.NaN
                self.params     =   np.NaN * np.ones(2)
                # classIn.params     =   np.NaN * np.ones(8) # maximum

        # arm command in ref. [2, 3] 
        self.prm_arm_mode                 =   prm_msg_veh_com()
        self.prm_arm_mode.CMD_mode        =   VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        self.prm_arm_mode.params[0]       =   1

        # disarm command in ref. [2, 3]
        self.prm_disarm_mode              =   prm_msg_veh_com()
        self.prm_disarm_mode.CMD_mode     =   VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        self.prm_disarm_mode.params[0]    =   0

        # offboard mode command in ref. [3]
        self.prm_offboard_mode            =   prm_msg_veh_com()
        self.prm_offboard_mode.CMD_mode   =   VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        self.prm_offboard_mode.params[0]  =   1
        self.prm_offboard_mode.params[1]  =   6
        
        #.. variable - vehicle attitude setpoint
        class var_msg_veh_att_set:
            def __init__(self):
                self.roll_body  =   np.NaN      # body angle in NED frame (can be NaN for FW)
                self.pitch_body =   np.NaN      # body angle in NED frame (can be NaN for FW)
                self.yaw_body   =   np.NaN      # body angle in NED frame (can be NaN for FW)
                self.q_d        =   [np.NaN, np.NaN, np.NaN, np.NaN]
                self.yaw_sp_move_rate   =   np.NaN      # rad/s (commanded by user)

                # For clarification: For multicopters thrust_body[0] and thrust[1] are usually 0 and thrust[2] is the negative throttle demand.
                # For fixed wings thrust_x is the throttle demand and thrust_y, thrust_z will usually be zero.
                self.thrust_body    =   np.NaN * np.ones(3) # Normalized thrust command in body NED frame [-1,1]

        self.veh_att_set    =   var_msg_veh_att_set()

        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

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
        self.vehicle_attitude_setpoint_publisher = self.create_publisher(
            VehicleAttitudeSetpoint,
            '/fmu/in/vehicle_attitude_setpoint',
            self.qos_profile
        )

        self.controller_heartbeat_publisher = self.create_publisher(
            Heartbeat,
            '/controller_heartbeat',
            10
        )

        self.path_planning_heartbeat_publisher = self.create_publisher(
            Heartbeat,    
            '/path_planning_heartbeat', 
            10
        )
        
        self.collision_avoidance_heartbeat_publisher  = self.create_publisher(
            Heartbeat,    
            '/collision_avoidance_heartbeat', 
            10
        )

        self.local_waypoint_publisher = self.create_publisher(
            LocalWaypointSetpoint,
            '/local_waypoint_setpoint_to_PF',
            10
        )

        # Publisher for global waypoint setpoint
        self.global_waypoint_publisher = self.create_publisher(
            GlobalWaypointSetpoint, 
            "/global_waypoint_setpoint", 
            10
        )
        self.local_waypoint_publisher2 = self.create_publisher(
            LocalWaypointSetpoint,
            '/local_waypoint_setpoint_from_PP',
            10
        )
        # set initial value
        self.offboard_setpoint_counter = 0
        self.offboard_start_flight_time = 10

        # set waypoint
        self.waypoint_x = [0.0, 72.28235294117647, 216.8470588235294, 349.36470588235295, 313.22352941176473, 445.74117647058824, 578.2588235294118, 710.7764705882353, 843.2941176470588, 939.6705882352941]
        self.waypoint_y = [0.0, 0.0, 48.188235294117646, 180.70588235294116, 481.88235294117646, 614.4, 746.9176470588235, 879.435294117647, 1011.9529411764706, 939.6705882352941]
        self.waypoint_z = [12.0, 20.0, 38.0, 103.0, 109.0, 68.0, 151.0, 84.0, 20.0, 35.0]

        self.convey_local_waypoint_is_complete = False

        # create timer for global waypoint publish
        
        period_heartbeat_mode =   1        
        self.heartbeat_timer  =   self.create_timer(period_heartbeat_mode, self.publish_collision_avoidance_heartbeat)
        self.heartbeat_timer  =   self.create_timer(period_heartbeat_mode, self.publish_path_planning_heartbeat)
        self.heartbeat_timer  =   self.create_timer(period_heartbeat_mode, self.publish_controller_heartbeat)


        period_offboard_control_mode =   0.2         # required about 5Hz for attitude control (proof that the external controller is healthy
        self.offboard_main_timer     =   self.create_timer(period_offboard_control_mode, self.offboard_control_main)

        period_offboard_att_ctrl  =   0.004        # required 250Hz at least for attitude control
        self.attitude_control_call_timer =  self.create_timer(period_offboard_att_ctrl, self.publisher_vehicle_attitude_setpoint)
    
    # main code
    def offboard_control_main(self):

        if self.convey_local_waypoint_is_complete == True:
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
        else:  
            self.timer = self.create_timer(1, self.local_waypoint_publish)


    # publish local waypoint
    def local_waypoint_publish(self):
        if self.convey_local_waypoint_is_complete == False:
            msg = LocalWaypointSetpoint()
            msg.path_planning_complete = True
            msg.waypoint_x             = self.waypoint_x
            msg.waypoint_y             = self.waypoint_y
            msg.waypoint_z             = self.waypoint_z
            self.local_waypoint_publisher2.publish(msg)
            self.local_waypoint_publisher.publish(msg)
            

            self.get_logger().info( "======================================================")
            self.get_logger().info( "local waypoint publish")
            self.get_logger().info( "======================================================")

    # publish attitude offboard command
    def publisher_vehicle_attitude_setpoint(self):
        if self.convey_local_waypoint_is_complete == True:
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

    # publish offboard control mode
    def publish_offboard_control_mode(self, prm_off_con_mod):
        msg                 =   OffboardControlMode()
        msg.position        =   prm_off_con_mod.position
        msg.velocity        =   prm_off_con_mod.velocity
        msg.acceleration    =   prm_off_con_mod.acceleration
        msg.attitude        =   prm_off_con_mod.attitude
        msg.body_rate       =   prm_off_con_mod.body_rate
        self.offboard_control_mode_publisher.publish(msg)

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

    # heartbeat publish
    def publish_collision_avoidance_heartbeat(self):
        msg = Heartbeat()
        msg.heartbeat = True
        self.collision_avoidance_heartbeat_publisher.publish(msg)

    def publish_path_planning_heartbeat(self):
        msg = Heartbeat()
        msg.heartbeat = True
        self.path_planning_heartbeat_publisher.publish(msg)

    def publish_controller_heartbeat(self):
        msg = Heartbeat()
        msg.heartbeat = True
        self.controller_heartbeat_publisher.publish(msg)

    def global_waypoint_publish(self):
        msg = GlobalWaypointSetpoint()
        msg.start_point = [self.waypoint_x[0], self.waypoint_z[0], self.waypoint_y[0]]
        msg.goal_point = [self.waypoint_x[-1], self.waypoint_z[-1], self.waypoint_y[-1]]
        self.global_waypoint_publisher.publish(msg)

    # subscribe convey local waypoint complete flag from path following
    def convey_local_waypoint_complete_call_back(self, msg):
        self.convey_local_waypoint_is_complete = msg.convey_local_waypoint_is_complete
        self.get_logger().info( "======================================================")
        self.get_logger().info( "local waypoint conveying complete")
        self.get_logger().info( "======================================================")

        self.global_waypoint_publish()

def main(args=None):
    rclpy.init(args=args)
    path_planning_test = PathPlanningTest()
    rclpy.spin(path_planning_test)
    path_planning_test.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
