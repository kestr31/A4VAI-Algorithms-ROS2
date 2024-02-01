import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import math

#   ROS2 python 
import rclpy
from rclpy.node import Node


from custom_msgs.srv import PathFollowingSetpoint

class PathFollowingService(Node):
    def __init__(self):
        super().__init__('following_service')

        self.local_waypoint_service_client = self.create_client(PathFollowingSetpoint, '/path_following_att_cmd')
        while not self.local_waypoint_service_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Path Following Setpoint Service not available, waiting again...') 
        
    def request_path_following(self, waypoint_x, waypoint_y, waypoint_z):
        self.path_following_request = PathFollowingSetpoint.Request()
        self.path_following_request.request_pathfollowing = True
        self.path_following_request.waypoint_x = waypoint_x
        self.path_following_request.waypoint_y = waypoint_y
        self.path_following_request.waypoint_z = waypoint_z
        
        self.future_setpoint = self.local_waypoint_service_client.call_async(self.path_following_request)