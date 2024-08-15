# Library
# Library for common
import numpy as np
import matplotlib.pyplot as plt

# Library for ros2
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

def setInitialVariables(classIn):

    # initialize figure
    classIn.fig = plt.figure()
    classIn.ax = classIn.fig.add_subplot(111, projection='3d')

    # initialize flag
    classIn.global_waypoint_set = False  # flag whether global waypoint is subscribed
    classIn.local_waypoint_set  = False  # flag whether local waypoint is subscribed

    # initialize global waypoint
    classIn.start_global_waypoint        =   []       # start global waypoint
    classIn.goal_global_waypoint         =   []       # goal global waypoint

    # initialize local waypoint
    classIn.waypoint_x         =   []
    classIn.waypoint_y         =   []
    classIn.waypoint_z         =   []

    # initialize vehicle position
    classIn.vehicle_x          =    np.array([])       # [m]
    classIn.vehicle_y          =    np.array([])       # [m]
    classIn.vehicle_z          =    np.array([])       # [m]

    # set qos profile
    classIn.qos_profile = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=1
    )