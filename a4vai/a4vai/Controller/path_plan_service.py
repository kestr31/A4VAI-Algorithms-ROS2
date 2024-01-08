#.. ROS libraries
import rclpy
from rclpy.node import Node

#.. custom msgs libararies - srv.
from custom_msgs.srv  import WaypointSetpoint

class PathPlanningService(Node):
    def __init__(self):
        super().__init__('planning_service')

        # Create waypoint service client
        self.waypoint_service_client = self.create_client(WaypointSetpoint, '/waypoint_setpoint')
        while not self.waypoint_service_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Path Planning not available, waiting a moment...') 
 
        # set service contents and request
    def request_path_planning(self, start_point, goal_point):
        self.request                            =   WaypointSetpoint.Request()
        self.request.request_path_planning      =   True
        self.request.start_point                =   start_point
        self.request.goal_point                 =   goal_point
        
        self.future = self.waypoint_service_client.call_async(self.request)

        #####
        # QOS setting is needed
        # It will be added after further study
        #####