#.. public libaries
import numpy as np
import math as m

#.. ROS libraries
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import qos_profile_sensor_data
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from std_msgs.msg import Float64MultiArray

#.. PX4 libararies - sub.
from px4_msgs.msg import EstimatorStates
from px4_msgs.msg import HoverThrustEstimate 
#.. PX4 libararies - pub.
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleAttitudeSetpoint

#.. PF algorithms libararies
# from .testpy1 import testpypy1
from .quadrotor_6dof import Quadrotor_6DOF
from .virtual_target import Virtual_Target
from .set_parameter import Simulation_Parameter, MPPI_Guidance_Parameter, Way_Point
from .MPPI_guidance import MPPI_Guidance_Modules
from .Funcs_PF_Base import kinematics, distance_from_Q6_to_path, check_waypoint, virtual_target_position, \
    guidance_modules, compensate_Aqi_cmd, thrust_cmd, att_cmd, NDO_Aqi
from .utility_functions import DCM_from_euler_angle



class TestAttitudeControlNode(Node):
    
    def __init__(self):
        super().__init__('test_attitude_control_node')
        
        # self.OffboardGroup = MutuallyExclusiveCallbackGroup()
        self.MPPIGroup = MutuallyExclusiveCallbackGroup()
        
        #.. Reference        
        #   [1]: https://docs.px4.io/main/en/msg_docs/vehicle_command.html
        #   [2]: https://mavlink.io/en/messages/common.html#MAV_CMD_COMPONENT_ARM_DISARM
        #   [3]: https://github.com/PX4/px4_ros_com/blob/release/1.13/src/examples/offboard/offboard_control.cpp
        #   [4]: https://docs.px4.io/main/ko/advanced_config/tuning_the_ecl_ekf.html
        
        #.. mapping of ros2-px4 message name using in this code
        #   from ' [basedir]/ws_sensor_combined/src/px4_ros_com/templates/urtps_bridge_topics.yaml '
        #   to ' [basedir]/PX4-Autopilot/msg/tools/urtps_bridge_topics.yaml '
        class msg_mapping_ros2_to_px4:
            VehicleCommand          =   '/fmu/in/vehicle_command'
            OffboardControlMode     =   '/fmu/in/offboard_control_mode'
            TrajectorySetpoint      =   '/fmu/in/trajectory_setpoint'
            VehicleAttitudeSetpoint =   '/fmu/in/vehicle_attitude_setpoint'
            EstimatorStates         =   '/fmu/out/estimator_states'
            HoverThrustEstimate     =   '/fmu/out/hover_thrust_estimate'
                    
        #.. publishers - from ROS2 msgs to px4 msgs
        self.vehicle_command_publisher_             =   self.create_publisher(VehicleCommand, msg_mapping_ros2_to_px4.VehicleCommand, 10)
        self.offboard_control_mode_publisher_       =   self.create_publisher(OffboardControlMode, msg_mapping_ros2_to_px4.OffboardControlMode , 10)
        self.trajectory_setpoint_publisher_         =   self.create_publisher(TrajectorySetpoint, msg_mapping_ros2_to_px4.TrajectorySetpoint, 10)
        self.vehicle_attitude_setpoint_publisher_   =   self.create_publisher(VehicleAttitudeSetpoint, msg_mapping_ros2_to_px4.VehicleAttitudeSetpoint, 10)        
        #.. subscriptions - from px4 msgs to ROS2 msgs
        self.estimator_states_subscription          =   self.create_subscription(EstimatorStates, msg_mapping_ros2_to_px4.EstimatorStates, self.subscript_estimator_states, qos_profile_sensor_data)
        self.hover_thrust_estimate_subscription     =   self.create_subscription(HoverThrustEstimate, msg_mapping_ros2_to_px4.HoverThrustEstimate, self.subscript_hover_thrust_estimate, qos_profile_sensor_data)

        #.. publishers - from ROS2 msgs to ROS2 msgs
        self.MPPI_output_publisher_     =   self.create_publisher(Float64MultiArray, 'topic_name_MPPI_output', qos_profile_sensor_data)

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
                
        self.prm_off_con_mod            =   prm_msg_off_con_mod()
        # self.prm_off_con_mod.position   =   True
        self.prm_off_con_mod.attitude   =   True
        # True
        
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
        
        
        #.. variable - vehicle attitude setpoint
        class msg_veh_att_set:
            def __init__(self):
                self.roll_body  =   np.NaN      # body angle in NED frame (can be NaN for FW)
                self.pitch_body =   np.NaN      # body angle in NED frame (can be NaN for FW)
                self.yaw_body   =   np.NaN      # body angle in NED frame (can be NaN for FW)
                self.q_d        =   [np.NaN, np.NaN, np.NaN, np.NaN]
                self.yaw_sp_move_rate   =   np.NaN      # rad/s (commanded by user)
                
                # For clarification: For multicopters thrust_body[0] and thrust[1] are usually 0 and thrust[2] is the negative throttle demand.
                # For fixed wings thrust_x is the throttle demand and thrust_y, thrust_z will usually be zero.
                self.thrust_body    =   np.NaN * np.ones(3) # Normalized thrust command in body NED frame [-1,1]
                
        self.veh_att_set    =   msg_veh_att_set()
                
        
        #.. other parameter & variable
        # timestamp
        self.timestamp  =   0
        
        # vehicle state variable
        self.pos_NED    =   np.zeros(3)
        self.vel_NED    =   np.zeros(3)
        self.eul_ang_deg    =   np.zeros(3)
        self.windvel_NE     =   np.zeros(2)
                
        # callback test_attitude_control
        period_offboard_att_ctrl    =   0.004
        # self.timer  =   self.create_timer(period_offboard_att_ctrl, self.test_attitude_control)
        # self.timer  =   self.create_timer(period_offboard_att_ctrl, self.test_attitude_control, callback_group = self.OffboardGroup)
        
        # callback offboard_control_mode
        # offboard counter in [3]
        period_offboard_control_mode    =   0.2
        # self.timer  =   self.create_timer(period_offboard_control_mode, self.offboard_control_mode)
        # self.timer  =   self.create_timer(period_offboard_control_mode, self.offboard_control_mode, callback_group = self.OffboardGroup)
        self.prm_offboard_setpoint_counter_start_flight  =   m.ceil(2/period_offboard_control_mode)
        self.offboard_setpoint_counter_     =   0
        
        # callback test_attitude_control
        period_MPPI_param       =   0.05
        self.timer  =   self.create_timer(period_MPPI_param, self.PF_MPPI_param, callback_group = self.MPPIGroup)
        
        ###### - start - Vars. for PF algorithm ######
        #.. declare variables/instances
        self.Q6     =   Quadrotor_6DOF()
        self.Q6.dt_GCU  =   period_offboard_att_ctrl
        self.Guid_type  =   3           # | 0: PD control | 1: guidance law | 2: MPPI direct accel cmd | 3: MPPI guidance-based |
        
        self.VT     =   Virtual_Target()
        
        #.. set waypoint
        wp_type_selection   =   1       # | 0: straight line | 1: ractangle | 2: circle | 3: designed
        self.WP     =   Way_Point(wp_type_selection)

        #.. MPPI setting
        # parameter
        type_MPPI   =   self.Q6.Guid_type   # 0~1: no use MPPI | 2: direct accel cmd | 3: guidance-based |
        self.MP     =   MPPI_Guidance_Parameter(type_MPPI)
        self.MP.dt  =   period_MPPI_param
        if type_MPPI == 2:
            self.Q6.desired_speed = 2.
            self.Q6.look_ahead_distance = 4.
            self.Q6.guid_eta = 3.
            self.MP.u1_init = 0.
            self.MP.u2_init = 0.
            self.MP.u3_init = 0.
        elif type_MPPI == 3:
            self.Q6.desired_speed = 1.
            self.Q6.look_ahead_distance = 1.
            self.Q6.guid_eta = 3.
            self.MP.u1_init = self.Q6.look_ahead_distance
            self.MP.u2_init = self.Q6.desired_speed
            self.MP.u3_init = self.Q6.guid_eta
        # module
        self.MG      =   MPPI_Guidance_Modules(self.MP)
        # initialization
        self.MG.set_total_MPPI_code(self.WP.WPs.shape[0])
        self.MG.set_MPPI_entropy_calc_code()
        
        self.MPPI_ctrl_input = np.array([self.MP.u1_init, self.MP.u2_init, self.MP.u3_init])
                
        #.. hover_thrust
        self.hover_thrust = 0.75
        ###### -  end  - Vars. for PF algorithm ######
        
        
    ### main function
    def offboard_control_mode(self):
        if self.offboard_setpoint_counter_ == self.prm_offboard_setpoint_counter_start_flight:
            # print("----- debug point [1] -----")
            
            # offboard mode cmd
            self.publish_vehicle_command(self.prm_offboard_mode)
            # arm cmd
            self.publish_vehicle_command(self.prm_arm_mode)
            
        # set offboard cntrol mode 
        self.publish_offboard_control_mode(self.prm_off_con_mod)
            
        # count offboard_setpoint_counter_
        if self.offboard_setpoint_counter_ < self.prm_offboard_setpoint_counter_start_flight:
            self.offboard_setpoint_counter_ = self.offboard_setpoint_counter_ + 1
            #.. variable setting
            self.Q6.Ri  =   self.pos_NED
            self.Q6.Vi  =   self.vel_NED
            self.Q6.att_ang =   self.eul_ang_deg * m.pi / 180.
            self.Q6.cI_B    =   DCM_from_euler_angle(self.Q6.att_ang)
            #.. initialization
            self.WP.init_WP(self.Q6.Ri)
            self.VT.init_VT_Ri(self.WP.WPs, self.Q6.Ri, self.Q6.look_ahead_distance)
            
        pass
    
    
    #.. test_attitude_control 
    def test_attitude_control(self):
        
        #.. variable setting
        self.Q6.Ri  =   self.pos_NED
        self.Q6.Vi  =   self.vel_NED
        self.Q6.att_ang =   self.eul_ang_deg * m.pi / 180.
        self.Q6.cI_B    =   DCM_from_euler_angle(self.Q6.att_ang)
        self.Q6.throttle_hover = self.hover_thrust
        
        ###### - start - PF algorithm ######
        #.. kinematics
        mag_Vqi, LOS_azim, LOS_elev, tgo, FPA_azim, FPA_elev, self.Q6.cI_W = kinematics(self.VT.Ri, self.Q6.Ri, self.Q6.Vi)
        LA_azim     =   FPA_azim - LOS_azim
        LA_elev     =   FPA_elev - LOS_elev
        
        #.. distance from quadrotor to ref. path  
        dist_to_path, self.Q6.p_closest_on_path, self.Q6.WP_idx_passed, unit_Rw1w2 = \
            distance_from_Q6_to_path(self.WP.WPs, self.Q6.WP_idx_heading, self.Q6.Ri, self.Q6.p_closest_on_path, self.Q6.WP_idx_passed)
            
        
        #.. virtual target modules
        #.. directly decide a position of the virtual target 
        # check waypoint - quadrotor
        self.Q6.WP_idx_heading = check_waypoint(self.WP.WPs, self.Q6.WP_idx_heading, self.Q6.Ri, self.Q6.distance_change_WP)
        # virtual target position
        self.VT.Ri = virtual_target_position(dist_to_path, self.Q6.look_ahead_distance, self.Q6.p_closest_on_path, self.Q6.WP_idx_passed, self.WP.WPs)
        
        #.. guidance modules
        self.Q6.Ai = (self.Q6.Vi - self.Q6.Vi_prev) / self.Q6.dt_GCU
        self.Q6.Vi_prev = self.Q6.Vi
        Aqi_cmd, self.Q6.flag_guid_trans = guidance_modules(self.Q6.Guid_type, mag_Vqi, self.Q6.WP_idx_passed, self.Q6.flag_guid_trans, self.Q6.WP_idx_heading, self.WP.WPs.shape[0],
                    self.VT.Ri, self.Q6.Ri, self.Q6.Vi, self.Q6.Ai, self.Q6.desired_speed, self.Q6.Kp_vel, self.Q6.Kd_vel, self.Q6.Kp_speed, self.Q6.Kd_speed, self.Q6.guid_eta, self.Q6.cI_W, tgo, self.MPPI_ctrl_input)                
            
            
        #.. compensate Aqi_cmd
        self.Q6.Ai_est_dstb  =   self.Q6.out_NDO.copy()
        Aqi_grav    =   np.array([0., 0., 9.81])
        Aqi_cmd = compensate_Aqi_cmd(Aqi_cmd, self.Q6.Ai_est_dstb, Aqi_grav)
        
        #.. thrust command
        MP_a_lim = 1.0
        self.Q6.mag_Aqi_thru, mag_thrust_cmd, norm_thrust_cmd = thrust_cmd(Aqi_cmd, self.Q6.throttle_hover, MP_a_lim, self.Q6.mass)
        
        # .. att_cmd
        att_ang_cmd = att_cmd(Aqi_cmd, self.Q6.att_ang[2], LOS_azim)
        
        #.. NDO_Aqi
        self.Q6.thr_unitvec, self.Q6.out_NDO, self.Q6.z_NDO = NDO_Aqi(
            Aqi_grav, self.Q6.mag_Aqi_thru, self.Q6.cI_B, self.Q6.thr_unitvec, self.Q6.gain_NDO, self.Q6.z_NDO, self.Q6.Vi, self.Q6.dt_GCU)
        
        ###### -  end  - PF algorithm ######
        
        w, x, y, z = self.Euler2Quaternion(att_ang_cmd[0], att_ang_cmd[1], att_ang_cmd[2])
        self.veh_att_set.thrust_body    =   [0., 0., -norm_thrust_cmd]
        
        self.veh_att_set.q_d            =   [w, x, y, z]
        self.publisher_vehicle_attitude_setpoint(self.veh_att_set)
        
        pass
    
    #.. PF_MPPI_param 
    def PF_MPPI_param(self):
        #.. MPPI algorithm
        MPPI_ctrl_input1, MPPI_ctrl_input2    =   self.MG.run_MPPI_Guidance(self.Q6, self.WP.WPs, self.VT)
        self.MPPI_ctrl_input    =   MPPI_ctrl_input1.copy()
        print("MPPI: [0]=" + str(self.MPPI_ctrl_input[0]) + ", [1]=" + str(self.MPPI_ctrl_input[1]) + ", [2]=" + str(self.MPPI_ctrl_input[2]))
        pass
    
    
                    
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
        
        pass
    
    #.. publisher_vehicle_attitude_setpoint 
    def publisher_vehicle_attitude_setpoint(self, veh_att_set):
        msg                 =   VehicleAttitudeSetpoint()
        msg.timestamp = int(Clock().now().nanoseconds / 1000) # time in microseconds
        # msg.roll_body       =   veh_att_set.roll_body
        # msg.pitch_body      =   veh_att_set.pitch_body
        # msg.yaw_body        =   veh_att_set.yaw_body
        msg.q_d[0]          =   veh_att_set.q_d[0]
        msg.q_d[1]          =   veh_att_set.q_d[1]
        msg.q_d[2]          =   veh_att_set.q_d[2]
        msg.q_d[3]          =   veh_att_set.q_d[3]
        msg.thrust_body[0]  =   0.
        msg.thrust_body[1]  =   0.
        msg.thrust_body[2]  =   veh_att_set.thrust_body[2]
        self.vehicle_attitude_setpoint_publisher_.publish(msg)
        
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
        self.eul_ang_deg[0], self.eul_ang_deg[1], self.eul_ang_deg[2] = \
            self.Quaternion2Euler(msg.states[0], msg.states[1], msg.states[2], msg.states[3])
        
        # Wind Velocity NE
        self.windvel_NE[0]  =   msg.states[22]
        self.windvel_NE[1]  =   msg.states[23]
        pass
    
    def subscript_hover_thrust_estimate(self, msg):
        print("------ Debug [6] ------")
        self.hover_thrust   =   msg.hover_thrust
        print("----- self.hover_thrust = " + str(self.hover_thrust))
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
    print("------------- main() in test_att_ctrl_dummy_MPPI.py -------------")
    print("======================================================")
    rclpy.init(args=args)
    TestAttitudeControl = TestAttitudeControlNode()
    rclpy.spin(TestAttitudeControl)
    TestAttitudeControl.destroy_node()
    rclpy.shutdown()
    pass
if __name__ == '__main__':
    main()


