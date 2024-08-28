from datetime import datetime
import numpy as np

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import VehicleCommand
from cv_bridge import CvBridge

def setInitialVariables(classIn):
    ## initialize flag
    # flag of start
    classIn.take_off_flag                      =   False
    classIn.initial_position_flag              =   False

    # flag of conveying local waypoint to another node
    classIn.path_planning_complete             =   False       # flag whether path planning is complete 
    classIn.convey_local_waypoint_to_PF_start  =   False
    classIn.convey_local_waypoint_is_complete  =   False       # flag whether path planning convey to path following
    classIn.path_following_flag                =   False
    classIn.collision_avoidance_flag           =   False
    classIn.obstacle_flag                      =   False
    classIn.obstacle_check 			        =   False

    classIn.collision_avoidance_end_timer      = False
    classIn.collision_avoidance_timer_running  = False
    classIn.collision_avoidance_start          = False
    # heartbeat signal of another module node
    classIn.path_planning_heartbeat            =   False
    classIn.path_following_heartbeat           =   False
    classIn.collision_avoidance_heartbeat      =   False

    classIn.bridge = CvBridge()

    classIn.collision_avidance_time = 100
    classIn.collision_avidance_elapsed_time = 0
    classIn.current_time = datetime.now()
    ## initialize State Variable
    # NED Position 
    classIn.x      =   0       # [m]
    classIn.y      =   0       # [m]
    classIn.z      =   0       # [m]

    # NED Velocity
    classIn.v_x    =   0       # [m/s]
    classIn.v_y    =   0       # [m/s]
    classIn.v_z    =   0       # [m/s]
    classIn.u      =   0   
    classIn.v      =   0   
    classIn.w      =   0  
    # Body Velocity CMD
    classIn.vel_cmd_body = [0.0, 0.0, 0.0]     # [m/s]
    classIn.vel_cmd_ned  = [0.0, 0.0, 0.0]     # [m/s]
    classIn.DCM_bn = np.zeros((3,3))
    classIn.DCM_nb = np.zeros((3,3))
    classIn.vel_cmd_body_x = 0.0
    classIn.vel_cmd_body_y = 0.0
    classIn.vel_cmd_body_z = 0.0
    classIn.min_dist = 0
    # Euler Angle
    classIn.psi    =   0       # [rad]
    classIn.theta  =   0       # [rad]
    classIn.phi    =   0       # [rad]

    classIn.vehicle_heading = 0
    # Body frame Angular Velocity
    classIn.p      =   0       # [rad/s]
    classIn.q      =   0       # [rad/s]
    classIn.r      =   0       # [rad/s]

    classIn.min_distance = 0

    # initial position
    classIn.initial_position = [0.0, 0.0, -10.0]

    #.. callback state_logger
    # classIn.period_state_logger = 0.1
    # classIn.timer  =   classIn.create_timer(classIn.period_state_logger, classIn.state_logger)
    # classIn.flightlogFile = open("/home/user/workspace/ros2/ros2_ws/log/flight_log.txt",'w')
    # classIn.datalogFile = open("/home/user/workspace/ros2/ros2_ws/log/data_log.txt",'w')

    ## initialize path planning parameter
    # path planning global waypoint [x, z, y]
    classIn.start_point        =   [0.0, 10.0, 0.0]
    classIn.goal_point         =   [950.0, 5.0, 950.0]

    # path planning waypoint list
    classIn.waypoint_x         =   []
    classIn.waypoint_y         =   []
    classIn.waypoint_z         =   []

    #.. parameter - offboard control mode
    class prm_msg_off_con_mod:
        def __init__(self):        
            classIn.position        =   False
            classIn.velocity        =   False
            classIn.acceleration    =   False
            classIn.attitude        =   False
            classIn.body_rate       =   False
            
    classIn.prm_off_con_mod            =   prm_msg_off_con_mod()
    classIn.prm_off_con_mod.position   =   True

    # offboard times
    classIn.offboard_setpoint_counter = 0
    classIn.offboard_start_flight_time = 10

    classIn.qos_profile = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=1
    )
    classIn.agl = 0
    ###.. - Start - set variable of publisher msg for PX4 - ROS2  ..###
    #
    #.. parameter - vehicle command 
    class prm_msg_veh_com:
        def __init__(self):
            self.CMD_mode   =   np.NaN
            self.params     =   np.NaN * np.ones(2)
            # classIn.params     =   np.NaN * np.ones(8) # maximum
            
    # arm command in ref. [2, 3] 
    classIn.prm_arm_mode                 =   prm_msg_veh_com()
    classIn.prm_arm_mode.CMD_mode        =   VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
    classIn.prm_arm_mode.params[0]       =   1
            
    # disarm command in ref. [2, 3]
    classIn.prm_disarm_mode              =   prm_msg_veh_com()
    classIn.prm_disarm_mode.CMD_mode     =   VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
    classIn.prm_disarm_mode.params[0]    =   0
    
    # offboard mode command in ref. [3]
    classIn.prm_offboard_mode            =   prm_msg_veh_com()
    classIn.prm_offboard_mode.CMD_mode   =   VehicleCommand.VEHICLE_CMD_DO_SET_MODE
    classIn.prm_offboard_mode.params[0]  =   1
    classIn.prm_offboard_mode.params[1]  =   6
    
    #.. parameter - offboard control mode
    class prm_msg_off_con_mod:
        def __init__(self):        
            self.position        =   False
            self.velocity        =   False
            self.acceleration    =   False
            self.attitude        =   False
            self.body_rate       =   False
            
    classIn.prm_off_con_mod            =   prm_msg_off_con_mod()
    classIn.prm_off_con_mod.position   =   True

    class msg_veh_trj_set:
        def __init__(self):
            self.pos_NED        =   np.zeros(3)     # meters
            self.vel_NED        =   np.zeros(3)     # meters/second
            self.yaw_rad        =   0.
            self.yaw_vel_rad    =   0.                    # radians/second
    
    classIn.veh_trj_set    =   msg_veh_trj_set()

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
            
    classIn.veh_att_set    =   var_msg_veh_att_set()
    #
    ###.. -  End  - set variable of publisher msg for PX4 - ROS2  ..###