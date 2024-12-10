from datetime import datetime
import numpy as np

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import VehicleCommand
from cv_bridge import CvBridge


def set_initial_variables(classIn):

    ## initialize State Variable
    # NED Position
    classIn.x = 0.0  # [m]
    classIn.y = 0.0  # [m]
    classIn.z = 0.0  # [m]

    # Euler Angle
    classIn.psi = 0.0  # [rad]
    classIn.theta = 0.0  # [rad]
    classIn.phi = 0.0  # [rad]

    classIn.heading = 0

    # DCM
    classIn.DCM_nb = np.zeros((3, 3))
    classIn.DCM_bn = np.zeros((3, 3))

    # waypoint
    classIn.wp_x = [-27.50*2, 30.0, 30.0, 0.0]
    classIn.wp_y = [27.50*2, 30.0, 0.0, 0.0]

    classIn.cur_wp = 0
    classIn.wp_distance = 0


    classIn.bridge = CvBridge()

    classIn.yaw_cmd_rad = 0.0

    classIn.obstacle_check = False
    classIn.obstacle_flag = False

    classIn.vel_body_cmd_normal = 2.0
    classIn.vel_body_cmd = np.zeros(3)

    classIn.vel_ned_cmd_normal = np.zeros(3)
    classIn.vel_ned_cmd_ca = np.zeros(3)

    classIn.vel_cmd_body_x = 0.0
    classIn.vel_cmd_body_y = 0.0
    classIn.vel_cmd_body_z = 0.0
    classIn.collision_avoidance_yaw_vel_rad = 0.0

    classIn.offboard_initial_time = 0
    classIn.takeoff_start_time = 10

    classIn.initial_position_flag = False
    classIn.initial_position = [0.0, 0.0, -6.0]

    classIn.collision_avoidance_heartbeat = False
    
    classIn.qos_profile = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )

    # .. parameter - vehicle command
    class prm_msg_veh_com:
        def __init__(self):
            self.CMD_mode = np.NaN
            self.params = np.NaN * np.ones(2)
            # classIn.params     =   np.NaN * np.ones(8) # maximum

    # arm command in ref. [2, 3]
    classIn.prm_arm_mode = prm_msg_veh_com()
    classIn.prm_arm_mode.CMD_mode = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
    classIn.prm_arm_mode.params[0] = 1

    # disarm command in ref. [2, 3]
    classIn.prm_disarm_mode = prm_msg_veh_com()
    classIn.prm_disarm_mode.CMD_mode = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
    classIn.prm_disarm_mode.params[0] = 0

    # offboard mode command in ref. [3]
    classIn.prm_offboard_mode = prm_msg_veh_com()
    classIn.prm_offboard_mode.CMD_mode = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
    classIn.prm_offboard_mode.params[0] = 1
    classIn.prm_offboard_mode.params[1] = 6

    # .. parameter - offboard control mode
    class prm_msg_off_con_mod:
        def __init__(self):
            self.position = False
            self.velocity = False
            self.acceleration = False
            self.attitude = False
            self.body_rate = False

    classIn.prm_off_con_mod = prm_msg_off_con_mod()
    classIn.prm_off_con_mod.position = True

    class msg_veh_trj_set:
        def __init__(self):
            self.pos_NED = np.zeros(3)  # meters
            self.vel_NED = np.zeros(3)  # meters/second
            self.yaw_rad = 0.0
            self.yaw_vel_rad = 6.0  # radians/second

    classIn.veh_trj_set = msg_veh_trj_set()


    classIn.flightlogFile = open("/home/user/workspace/ros2/ros2_ws/src/algorithm_test/algorithm_test/collosion_avoidance_unit_test/log.txt",'w')

    #
    ###.. -  End  - set variable of publisher msg for PX4 - ROS2  ..###
