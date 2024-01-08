import numpy as np

import rclpy
from rclpy.node import Node

from .path_plan_service import PathPlanningService

class Controller(Node):

    def __init__(self):
        super().__init__("controller")
        
        # initialize waypoint parameter
        self.waypoint_index     =   0
        self.waypoint_x         =   []
        self.waypoint_y         =   []
        self.waypoint_z         =   []

        # path planning position
        # [x, z, y]
        self.start_point        =   [1.0, 5.0, 1.0]
        self.goal_point         =   [950.0, 150.0, 950.0]

        # NED position status
        self.x                  =   0
        self.y                  =   0
        self.z                  =   0

        # path planning start
        path_planning_service = PathPlanningService()
        path_planning_service.request_path_planning(self.start_point, self.goal_point)
        rclpy.spin_until_future_complete(path_planning_service, path_planning_service.future)
        if path_planning_service.future.done():
            try : 
                path_planning_service.result = path_planning_service.future.result()
            except Exception as e:
                path_planning_service.get_logger().info(
                    'Path Planning Service call failed %r' % (e,))
            else :
                path_planning_service.get_logger().info( "Path Planning Complete!! ")
                if path_planning_service.result.response_path_planning is True :
                    self.waypoint_x                 =   path_planning_service.result.waypoint_x
                    self.waypoint_y                 =   path_planning_service.result.waypoint_y
                    self.waypoint_z                 =   path_planning_service.result.waypoint_z
                    self.path_planning_complete     = True
                else :
                    pass
            finally : 
                path_planning_service.destroy_node()
        else : 
            self.get_logger().warn("===== Path Planning Module Can't Response =====")

       

        
    #     # period of publishing waypoint from Controller to PathPlanning
    #     ##### can change period 
    #     self.waypoint_index_publish_period = 0.1

    #     self.waypoint_index_publisher = self.create_publisher(WaypointIndex,'/waypoint_index',10)

    #     self.waypoint_index_publish_timer = self.create_timer(self.waypoint_publish_period, self.publish_waypoint_index)


    # def publish_waypoint_index(self):
    #     msg = WaypointIndex()
    #     msg.waypoint_index = self.waypoint_index
    #     self.waypoint_index_publisher.publish(msg)

    # def check_waypoint_index(self):
    #     remaining_distance = np.sqrt((self.x - self.waypoint_x[self.waypoint_index])**2 - (self.y - self.waypoint_y[self.waypoint_index])**2)
        
    #     if remaining_distance <= 3.0 :
    #         self.waypoint_index += 1
    #     else:
    #         pass


def main(args=None):
    print("======================================================")
    print("------------- main() in get_waypoint.py -------------")
    print("======================================================")
    rclpy.init(args=args)
    controller = Controller()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()