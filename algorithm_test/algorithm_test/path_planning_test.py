# Librarys

# Library for common
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ROS libraries
import rclpy
from rclpy.node import Node

# custom msgs libararies
from custom_msgs.msg import GlobalWaypointSetpoint
from custom_msgs.msg import LocalWaypointSetpoint
from custom_msgs.msg import Heartbeat


class PathPlanningTest(Node):
    def __init__(self):
        super().__init__("give_global_waypoint")

        # Publisher for global waypoint setpoint
        self.global_waypoint_publisher = self.create_publisher(
            GlobalWaypointSetpoint, 
            "/global_waypoint_setpoint", 
            10
        )

        # Subscriber for local waypoint setpoint from path planning
        self.local_waypoint_subscriber = self.create_subscription(
            LocalWaypointSetpoint,
            "/local_waypoint_setpoint_from_PP",
            self.local_waypoint_callback,
            10,
        )

        # set initial value

        # set start and goal point
        self.start_point = [0.0, 5.0, 0.0]
        self.goal_point = [950.0, 15.0, 950.0]

        # set waypoint
        self.waypoint_x = []
        self.waypoint_y = []
        self.waypoint_z = []

        # set flag
        self.path_planning_complete = False
        self.local_waypoint_set = False

        # initialize figure
        self.fig = plt.figure(figsize=(10, 10))
        self.ax1 = self.fig.add_subplot(121)  
        self.ax2 = self.fig.add_subplot(122, projection='3d') 

        self.controller_heartbeat_publisher = self.create_publisher(
            Heartbeat,
            '/controller_heartbeat',
            10
        )

        self.path_following_heartbeat_publisher = self.create_publisher(
            Heartbeat,    
            '/path_following_heartbeat', 
            10
        )
        
        self.collision_avoidance_heartbeat_publisher  = self.create_publisher(
            Heartbeat,    
            '/collision_avoidance_heartbeat', 
            10
        )

        # create timer for global waypoint publish
        self.timer = self.create_timer(1, self.global_waypoint_publish)

        period_heartbeat_mode =   1        
        self.heartbeat_timer  =   self.create_timer(period_heartbeat_mode, self.publish_collision_avoidance_heartbeat)
        self.heartbeat_timer  =   self.create_timer(period_heartbeat_mode, self.publish_path_following_heartbeat)
        self.heartbeat_timer  =   self.create_timer(period_heartbeat_mode, self.publish_controller_heartbeat)

    # publish global waypoint
    def global_waypoint_publish(self):

        if self.path_planning_complete == False:

            # input : global start waypoint [x, z, y]
            #         global goal waypoint  [x, z, y]

            msg = GlobalWaypointSetpoint()
            msg.start_point = self.start_point
            msg.goal_point = self.goal_point

            self.global_waypoint_publisher.publish(msg)

            self.get_logger().info(
                "======================================================"
            )
            self.get_logger().info("Global Waypoint Publish")
            self.get_logger().info("start_point :" + str(self.start_point))
            self.get_logger().info("goal_point  :" + str(self.goal_point))
            self.get_logger().info(
                "======================================================"
            )

            #####
            # QOS setting is needed
            # It will be added after further study
            #####

    # local waypoint callback function
    def local_waypoint_callback(self, msg):
        self.waypoint_x = msg.waypoint_x
        self.waypoint_y = msg.waypoint_y
        self.waypoint_z = msg.waypoint_z
    
        self.path_planning_complete = msg.path_planning_complete

        with open('/home/user/workspace/ros2/logs/pathplanning.log', 'a') as log_file:  # 'a' 모드는 append 모드로, 기존 내용에 추가합니다.
            log_file.write("\n")  
            log_file.write(f"Waypoint X: {self.waypoint_x}\n")
            log_file.write(f"Waypoint Y: {self.waypoint_y}\n")
            log_file.write(f"Waypoint Z: {self.waypoint_z}\n")
            log_file.write("\n")  
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # region Plot 1 2D full trajectory

        # Plot global waypoints with blue color
        self.ax1.scatter(
            [self.start_point[0], self.goal_point[0]],
            [self.start_point[2], self.goal_point[2]],
            color="blue",
            label="Global Waypoint",
            s=70,
        )

        # Plot local waypoints with red color
        self.ax1.scatter(
            self.waypoint_x,
            self.waypoint_y,
            color="red",
            label="Local Waypoints",
            s=6,
        )
        self.ax1.plot(
            self.waypoint_x,
            self.waypoint_y,
            color="red",
            linewidth=1,
        )

        # set the title, x and y labels
        self.ax1.set_title("full trajectory")
        self.ax1.set_xlabel("X Coordinate")
        self.ax1.set_ylabel("Y Coordinate")
        self.ax1.legend()

        # endregion
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # region Plot 2 altutude

        # Plot global waypoints with blue color


        # Plot local waypoints with red color
        self.ax2.scatter(
            [self.start_point[0], self.goal_point[0]],
            [self.start_point[2], self.goal_point[2]],
            [self.start_point[1], self.goal_point[1]],
            color="blue",
            label="Local Waypoints",
            s=70,
        )
        self.ax2.plot(
            self.waypoint_x,
            self.waypoint_y,
            self.waypoint_z,
            color="red",
            linewidth=4,
        )

        # set the title, x and y labels
        self.ax2.set_title("Vihicle Position")
        self.ax2.set_xlabel("X Coordinate")
        self.ax2.set_ylabel("Y Coordinate")
        self.ax2.set_zlabel("Z Coordinate")
        self.ax2.legend()

        self.get_logger().info("======================================================")
        self.get_logger().info("plot")
        self.get_logger().info("======================================================")

        plt.tight_layout()
        plt.draw()
        plt.pause(10000)

        # endregion
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # heartbeat publish
    def publish_collision_avoidance_heartbeat(self):
        msg = Heartbeat()
        msg.heartbeat = True
        self.collision_avoidance_heartbeat_publisher.publish(msg)

    def publish_path_following_heartbeat(self):
        msg = Heartbeat()
        msg.heartbeat = True
        self.path_following_heartbeat_publisher.publish(msg)

    def publish_controller_heartbeat(self):
        msg = Heartbeat()
        msg.heartbeat = True
        self.controller_heartbeat_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    path_planning_test = PathPlanningTest()
    rclpy.spin(path_planning_test)
    path_planning_test.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
