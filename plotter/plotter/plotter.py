# Library

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Library for ros2
import rclpy
from rclpy.node import Node

# Library for custom message
from custom_msgs.msg import GlobalWaypointSetpoint, LocalWaypointSetpoint

# Library for px4 message
from px4_msgs.msg import VehicleLocalPosition

# submodule for initial variables
from .initVar import *

class Plotter(Node):
    def __init__(self):
        super().__init__('plotter')

        setInitialVariables(self)

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # region Subscriber

        # Subscriber for global waypoint setpoint from controller
        self.global_waypoint_subscriber     =   self.create_subscription(
            GlobalWaypointSetpoint, 
            '/global_waypoint_setpoint', 
            self.global_waypoint_callback, 
            10
        )
        
        # Subscriber for local waypoint setpoint from path planning
        self.local_waypoint_subscriber      =   self.create_subscription(
            LocalWaypointSetpoint, 
            '/local_waypoint_setpoint_from_PP', 
            self.local_waypoint_callback, 
            10
        )

        # Subscriber for vehicle local position from px4
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.vehicle_local_position_callback,
            self.qos_profile
        )
        # endregion
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # region timer
        
        # update plot timer
        period_update_plot = 0.1
        self.timer = self.create_timer(period_update_plot, self.update_plot)

        # endregion
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # region Callback functions

    # global waypoint callback function
    def global_waypoint_callback(self, msg):
        self.start_global_waypoint    =   msg.start_point
        self.goal_global_waypoint     =   msg.goal_point
        self.global_waypoint_set = True

    # local waypoint callback function
    def local_waypoint_callback(self, msg):
        self.waypoint_x = msg.waypoint_x
        self.waypoint_y = msg.waypoint_y
        self.waypoint_z = msg.waypoint_z
        self.local_waypoint_set = msg.path_planning_complete
        
    # vehicle local position callback function
    def vehicle_local_position_callback(self, msg):
        # convert data to list
        vehicle_x = np.array(msg.x)
        vehicle_y = np.array(msg.y)
        vehicle_z = np.array(-msg.z)

        # append to list
        self.vehicle_x = np.append(self.vehicle_x, vehicle_x).flatten()
        self.vehicle_y = np.append(self.vehicle_y, vehicle_y).flatten()
        self.vehicle_z = np.append(self.vehicle_z, vehicle_z).flatten()

    # endregion
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    def update_plot(self):
        # Clear the previous plot
        self.ax.clear()

        # Plot global waypoints with blue color
        if self.global_waypoint_set == True:
            self.ax.scatter([self.start_global_waypoint[0], self.goal_global_waypoint[0]], 
                       [self.start_global_waypoint[2], self.goal_global_waypoint[2]], 
                       [self.start_global_waypoint[1], self.goal_global_waypoint[1]], 
                       color='blue', label='Start and Goal', s=100)

        # Plot local waypoints with red color
        if self.local_waypoint_set == True:
            self.ax.scatter(self.waypoint_x, self.waypoint_y, self.waypoint_z, color='red', label='Local Waypoints', s=2)  
            self.ax.plot(self.waypoint_x, self.waypoint_y, self.waypoint_z, color='red', label='Local Waypoints', linewidth=1)

        # Plot vehicle positions
        if len(self.vehicle_x) > 0:
            self.ax.plot(self.vehicle_x, self.vehicle_y, self.vehicle_z, color='green', label='Vehicle Position', linewidth=1)

        self.ax.set_title("3D Start and Goal Waypoints")
        self.ax.set_xlabel("X Coordinate")
        self.ax.set_ylabel("Y Coordinate")
        self.ax.set_zlabel("Z Coordinate")

        # Draw the updated plot
        if self.global_waypoint_set == True and self.local_waypoint_set == True:
            plt.draw()
            plt.pause(0.001)



def main(args=None):
    rclpy.init(args=args)
    node = Plotter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()