#.. public libaries
import numpy as np
import math as m

#.. ROS libraries
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import qos_profile_sensor_data


#.. PX4 libararies - sub.
from px4_msgs.msg import EstimatorStates
#.. PX4 libararies - pub.
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint

class TestPositionControlNode(Node):
    
    def __init__(self):
        super().__init__('test_position_control_node')
        #.. variable name convention
        #   prm_ : parameter which can be set by user
        
        #.. Reference        
        #   [1]: https://docs.px4.io/main/en/msg_docs/vehicle_command.html
        #   [2]: https://mavlink.io/en/messages/common.html#MAV_CMD_COMPONENT_ARM_DISARM
        #   [3]: https://github.com/PX4/px4_ros_com/blob/release/1.13/src/examples/offboard/offboard_control.cpp
        
        #.. mapping of ros2-px4 message name using in this code
        #   from ' [basedir]/ws_sensor_combined/src/px4_ros_com/templates/urtps_bridge_topics.yaml '
        #   to ' [basedir]/PX4-Autopilot/msg/tools/urtps_bridge_topics.yaml '
        class msg_mapping_ros2_to_px4:
            VehicleCommand          =   '/fmu/in/vehicle_command'
            OffboardControlMode     =   '/fmu/in/offboard_control_mode'
            TrajectorySetpoint      =   '/fmu/in/trajectory_setpoint'
            EstimatorStates         =   '/fmu/out/estimator_states'
                    
        #.. publishers - from ROS2 msgs to px4 msgs
        self.vehicle_command_publisher_             =   self.create_publisher(VehicleCommand, msg_mapping_ros2_to_px4.VehicleCommand, 10)
        self.offboard_control_mode_publisher_       =   self.create_publisher(OffboardControlMode, msg_mapping_ros2_to_px4.OffboardControlMode , 10)
        self.trajectory_setpoint_publisher_         =   self.create_publisher(TrajectorySetpoint, msg_mapping_ros2_to_px4.TrajectorySetpoint, 10)
        #.. subscriptions - from px4 msgs to ROS2 msgs
        self.estimator_states_subscription          =   self.create_subscription(EstimatorStates, msg_mapping_ros2_to_px4.EstimatorStates, self.subscript_estimator_states, qos_profile_sensor_data)
        #.. waypoint array
        self.prm_WP_array       =   np.array([
            [0., 0., -10.], 
            [50., 0., -10.], 
            [50., 50., -10.], 
            [0., 50., -10.], 
            [0., 0., -10.]
            ])
        self.prm_stn_WP_chn     =   10.0     # standard to change the WP [m]
        self.ind_head_WP        =   0       # index current WP          [-]
        
        #.. parameter - path following algorithm
        self.prm_dist_vir_tg    =   10.0     # distance to the virtual target from the vehicle
        #.. parameter - vehicle command 
        class prm_msg_veh_com:
            def __init__(self):
                self.CMD_mode   =   np.NaN
                self.params     =   np.NaN * np.ones(2)
                # self.params     =   np.NaN * np.ones(8) # maximum
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
        
        #.. parameter - offboard control mode
        class prm_msg_off_con_mod:
            def __init__(self):        
                self.position        =   False
                self.velocity        =   False
                self.acceleration    =   False
                self.attitude        =   False
                self.body_rate       =   False
                
        self.prm_off_con_mod              =   prm_msg_off_con_mod()
        self.prm_off_con_mod.position     =   True
        
        #.. variable - trajectory setpoint 
        class msg_trj_set:
            def __init__(self):
                self.pos_NED    =   np.NaN * np.ones(3)
                self.yaw_rad    =   np.NaN
                # self.vel_NED    =   np.NaN * np.ones(3)
                # self.yawspeed_rad   =   np.NaN
                # self.acc_NED    =   np.NaN * np.ones(3)
                # self.jerk_NED   =   np.NaN * np.ones(3)
                # self.thrust_NED =   np.NaN * np.ones(3)
                
        self.trj_set    =   msg_trj_set()
        
        #.. other parameter & variable
        # timestamp
        self.timestamp  =   0
        
        # offboard counter in [3]
        self.prm_offboard_setpoint_counter_start_flight  =   10    
        self.offboard_setpoint_counter_     =   0
        
        # vehicle state variable
        self.pos_NED    =   np.zeros(3)
        self.vel_NED    =   np.zeros(3)
        self.eul_ang_rad    =   np.zeros(3)
        self.windvel_NE     =   np.zeros(2)
                
        # callback main function
        freq_offboard_pos_ctrl      =   50
        pperiod_offboard_pos_ctrl   =   1/freq_offboard_pos_ctrl
        self.timer  =   self.create_timer(pperiod_offboard_pos_ctrl, self.test_position_control)
        
    ### main function
    #.. test_position_control 
    def test_position_control(self):
        
        print("----- debug point [1] -----")
        if self.offboard_setpoint_counter_ == self.prm_offboard_setpoint_counter_start_flight:
            # offboard mode cmd
            self.publish_vehicle_command(self.prm_offboard_mode)
            # arm cmd
            self.publish_vehicle_command(self.prm_arm_mode)
            print("----- debug point [111111111111111] -----")
        # set offboard cntrol mode 
        self.publish_offboard_control_mode(self.prm_off_con_mod)
        
        # # get the current heading WP
        # self.ind_head_WP = self.upd_heading_WP(self.pos_NED, self.prm_WP_array, self.ind_head_WP, self.prm_stn_WP_chn)
        # head_WP =   self.prm_WP_array[self.ind_head_WP]
        # # offboard control algorithm - position setpoint & virtual target position
        # self.trj_set.pos_NED, self.trj_set.yaw_rad = self.get_trajectory_setpoint_for_path_following(head_WP, self.pos_NED, self.prm_dist_vir_tg)
        # self.publish_trajectory_setpoint(self.trj_set)
        
        # # check - print
        # print("pos_NED = [%05.2f, %05.2f, %05.2f], \
        #     head_WP = [%05.2f, %05.2f, %05.2f], \
        #     trj_set_pos_NED = [%05.2f, %05.2f, %05.2f]" \
        #     %(self.pos_NED[0], self.pos_NED[1], self.pos_NED[2], 
        #       head_WP[0], head_WP[1], head_WP[2], 
        #       self.trj_set.pos_NED[0], self.trj_set.pos_NED[1], self.trj_set.pos_NED[2]))
        
        # count offboard_setpoint_counter_
        if self.offboard_setpoint_counter_ < self.prm_offboard_setpoint_counter_start_flight:
            self.offboard_setpoint_counter_ = self.offboard_setpoint_counter_ + 1
    
    ### algorithms
    #.. upd_heading_WP
    def upd_heading_WP(self, myPos_NED, prm_WP_array, ind_head_WP, prm_stn_WP_chn):
        head_WP     =   prm_WP_array[ind_head_WP]
        rel_pos     =   head_WP - myPos_NED
        rel_dist    =   np.linalg.norm(rel_pos)
        if rel_dist < prm_stn_WP_chn:
            ind_next_WP    =   ind_head_WP + 1
        else:
            ind_next_WP    =   ind_head_WP
        ind_next_WP =   min(ind_next_WP, prm_WP_array.shape[0] - 1)
        return ind_next_WP
                    
    #.. get_trajectory_setpoint_for_path_following
    def get_trajectory_setpoint_for_path_following(self, head_WP, myPos_NED, prm_dist_vir_tg):
        relvec          =   head_WP - myPos_NED
        unitvec_to_WP   =   relvec / np.linalg.norm(relvec)
        set_yaw_rad     =   m.atan2(unitvec_to_WP[1], unitvec_to_WP[0])
        # set_yaw_rad     =   0.
        set_pos_NED     =   myPos_NED + prm_dist_vir_tg * unitvec_to_WP
        return set_pos_NED, set_yaw_rad
                    
    ### publushers
    #.. publish_vehicle_command
    def publish_vehicle_command(self, prm_veh_com):
        msg                 =   VehicleCommand()
        msg.timestamp = int(Clock().now().nanoseconds / 1000) # time in microseconds
        msg.param1          =   prm_veh_com.params[0]
        msg.param2          =   prm_veh_com.params[1]
        msg.command         =   prm_veh_com.CMD_mode
        # values below are in [3]
        msg.target_system   =   1
        msg.target_component=   1
        msg.source_system   =   1
        msg.source_component=   1
        msg.from_external   =   True
        self.vehicle_command_publisher_.publish(msg)
        
        print("----- debug point [2] -----")
        pass
        
    #.. publish_offboard_control_mode
    def publish_offboard_control_mode(self, prm_off_con_mod):
        msg                 =   OffboardControlMode()
        msg.timestamp = int(Clock().now().nanoseconds / 1000) # time in microseconds
        msg.position        =   prm_off_con_mod.position
        msg.velocity        =   prm_off_con_mod.velocity
        msg.acceleration    =   prm_off_con_mod.acceleration
        msg.attitude        =   prm_off_con_mod.attitude
        msg.body_rate       =   prm_off_con_mod.body_rate
        self.offboard_control_mode_publisher_.publish(msg)
        
        print("----- debug point [3] -----")
        pass
        
    #.. publish_trajectory_setpoint
    def publish_trajectory_setpoint(self, trj_set):
        msg                 =   TrajectorySetpoint()
        msg.timestamp = int(Clock().now().nanoseconds / 1000) # time in microseconds
        # msg.x               =   trj_set.pos_NED[0]
        # msg.y               =   trj_set.pos_NED[1]
        # msg.z               =   trj_set.pos_NED[2]
        msg.position        =   trj_set.pos_NED.tolist()
        msg.yaw             =   trj_set.yaw_rad
        self.trajectory_setpoint_publisher_.publish(msg)
        
        print("----- debug point [4] -----")
        pass
        
    ### subscriptions        
    #.. subscript subscript_estimator_states
    def subscript_estimator_states(self, msg):
        self.pos_NED[0]     =   msg.states[7]
        self.pos_NED[1]     =   msg.states[8]
        self.pos_NED[2]     =   msg.states[9]
        self.vel_NED[0]     =   msg.states[4]
        self.vel_NED[1]     =   msg.states[5]
        self.vel_NED[2]     =   msg.states[6]
        # Attitude
        self.eul_ang_rad[0], self.eul_ang_rad[1], self.eul_ang_rad[2] = \
            self.Quaternion2Euler(msg.states[0], msg.states[1], msg.states[2], msg.states[3])
        
        # Wind Velocity NE
        self.windvel_NE[0]  =   msg.states[22]
        self.windvel_NE[1]  =   msg.states[23]
        pass
            
    ### Mathmatics Functions 
    #.. Quaternion to Euler
    def Quaternion2Euler(self, w, x, y, z):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        Roll = m.atan2(t0, t1) * 57.2958
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        Pitch = m.asin(t2) * 57.2958
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        Yaw = m.atan2(t3, t4) * 57.2958
        return Roll, Pitch, Yaw
    
    #.. Euler to Quaternion
    def Euler2Quaternion(self, Roll, Pitch, Yaw):
        CosYaw = m.cos(Yaw * 0.5)
        SinYaw = m.sin(Yaw * 0.5)
        CosPitch = m.cos(Pitch * 0.5)
        SinPitch = m.sin(Pitch * 0.5)
        CosRoll = m.cos(Roll * 0.5)
        SinRoll= m.sin(Roll * 0.5)
        
        w = CosRoll * CosPitch * CosYaw + SinRoll * SinPitch * SinYaw
        x = SinRoll * CosPitch * CosYaw - CosRoll * SinPitch * SinYaw
        y = CosRoll * SinPitch * CosYaw + SinRoll * CosPitch * SinYaw
        z = CosRoll * CosPitch * SinYaw - SinRoll * CosPitch * CosYaw
        
        return w, x, y, z
        
def main(args=None):
    print("======================================================")
    print("------------- main() in test_pos_ctrl.py -------------")
    print("======================================================")
    rclpy.init(args=args)
    TestPositionControl = TestPositionControlNode()
    rclpy.spin(TestPositionControl)
    TestPositionControl.destroy_node()
    rclpy.shutdown()
    pass
if __name__ == '__main__':
    main()


