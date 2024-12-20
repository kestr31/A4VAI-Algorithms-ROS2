from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import VehicleAttitudeSetpoint
from px4_msgs.msg import TrajectorySetpoint

from custom_msgs.msg import Heartbeat
from custom_msgs.msg import LocalWaypointSetpoint
from custom_msgs.msg import GlobalWaypointSetpoint

from std_msgs.msg import Float32
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Bool

class PX4Publisher:
    def __init__(self, node):
        self.node = node

    def declareVehicleCommandPublisher(self):
        self.node.vehicle_command_publisher = self.node.create_publisher(
            VehicleCommand, 
            "/fmu/in/vehicle_command",
            self.node.qos_profile_px4
        )

    def declareOffboardControlModePublisher(self):
        self.node.offboard_control_mode_publisher = self.node.create_publisher(
            OffboardControlMode,
            "/fmu/in/offboard_control_mode",
            self.node.qos_profile_px4
        )
    # publisher for vehicle attitude setpoint
    def declareVehicleAttitudeSetpointPublisher(self):
        self.node.vehicle_attitude_setpoint_publisher = self.node.create_publisher(
            VehicleAttitudeSetpoint,
            "/fmu/in/vehicle_attitude_setpoint",
            self.node.qos_profile_px4,
        )
    
    # publisher for vehicle velocity setpoint
    def declareTrajectorySetpointPublisher(self):
        self.node.trajectory_setpoint_publisher = self.node.create_publisher(
            TrajectorySetpoint,
            "/fmu/in/trajectory_setpoint",
            self.node.qos_profile_px4
        )

class WaypointPublisher:
    def __init__(self, node):
        self.node = node

    def declareLocalWaypointPublisherToPF(self):
        self.node.local_waypoint_publisher = self.node.create_publisher(
            LocalWaypointSetpoint,
            "/local_waypoint_setpoint_to_PF",
            1
        )

class HeartbeatPublisher:
    def __init__(self, node):
        self.node = node

    def declareControllerHeartbeatPublisher(self):
        self.node.controller_heartbeat_publisher = self.node.create_publisher(
            Bool,
            "/controller_heartbeat",
            1
        )

    def declarePathPlanningHeartbeatPublisher(self):
        self.node.path_planning_heartbeat_publisher = self.node.create_publisher(
            Bool,
            "/path_planning_heartbeat",
            1
        )

    def declareCollisionAvoidanceHeartbeatPublisher(self):
        self.node.collision_avoidance_heartbeat_publisher = self.node.create_publisher(
            Bool,
            "/collision_avoidance_heartbeat",
            1
        )

    def declarePathFollowingHeartbeatPublisher(self):
        self.node.path_following_heartbeat_publisher = self.node.create_publisher(
            Bool,
            "/path_following_heartbeat",
            1
        )

class PlotterPublisher:
    def __init__(self, node):
        self.node = node
    
    def declareGlobalWaypointPublisherToPlotter(self):
        self.node.global_waypoint_publisher_to_plotter = self.node.create_publisher(
            GlobalWaypointSetpoint,
            "/global_waypoint_setpoint_to_plotter",
            1
        )
    
    def declareLocalWaypointPublisherToPlotter(self):
        self.node.local_waypoint_publisher_to_plotter = self.node.create_publisher(
            LocalWaypointSetpoint,
            "/local_waypoint_setpoint_to_plotter",
            1
        )

    def declareHeadingPublisherToPlotter(self):
        self.node.heading_publisher_to_plotter = self.node.create_publisher(
            Float32,
            "/heading",
            1
        )

    def declareStatePublisherToPlotter(self):
        self.node.control_mode_publisher_to_plotter = self.node.create_publisher(
            Bool,
            "/controller_state",
            1
        )

    def declareMinDistancePublisherToPlotter(self):
        self.node.min_distance_publisher_to_plotter = self.node.create_publisher(
            Float64MultiArray,
            "/min_distance",
            1
        )


