import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from rclpy.clock import Clock
from .data_class import StateVariable, ModeFlag

import numpy as np
from px4_msgs.msg import VehicleLocalPosition
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Point
from std_msgs.msg import Byte


from .common_fuctions import set_initial_variables, state_logger, publish_to_plotter, set_wp
from custom_msgs.msg import StateFlag

class CollisionAvoidanceNode(Node):
    def __init__(self):
        super().__init__('collision_avoidance_node')

        self.mode_flag = ModeFlag()

        # obstacle info subscriber
        self.obstacle_subscription = self.create_subscription(
            Float32MultiArray,
            '/obstacle_info',
            self.obstacle_callback,
            1
        )

        self.flag_subscription = self.create_subscription(
            StateFlag,
            '/mode_flag2collision',
            self.flag_callback,
            1
        )
        # flag publisher
        self.flag_publisher = self.create_publisher(StateFlag, '/mode_flag2control', 1)
        self.obstacle_publisher_ = self.create_publisher(Float32MultiArray, '/obstacle', 1)


        # # temp waypoint publisher
        # self.temp_waypoint_publisher = self.create_publisher(Point, '/temp_waypoint', 1)

    def obstacle_callback(self, msg):
        # self.get_logger().info(f"is_pf: {self.mode_flag.is_pf}, is_ca: {self.mode_flag.is_ca}")
        obstacle_info = np.array(msg.data).reshape(-1, 6)  # [distance, azimuth, elevation]
        if self.mode_flag.is_offboard:
            if self.mode_flag.is_pf:
                for obstacle in obstacle_info:
                    azimuth, elevation, distance, x,y,z = obstacle
                    if distance < 8.0 and np.deg2rad(-18) <= azimuth <= np.deg2rad(18):
                        
                        self.mode_flag.is_ca = True
                        self.mode_flag.is_pf = False
                        self.publish_flags()
                        # self.publish_obstacle_info(x,y,z)
                        # self.get_logger().info(f"is_ca: distance={distance}, azimuth={np.degrees(azimuth)}")
                        break

            if self.mode_flag.is_ca:
                obstacle_detected = False
                for obstacle in obstacle_info:
                    azimuth, elevation, distance, x,y,z = obstacle
                    if distance < 4.:
                        obstacle_detected = True
                        break
                    elif np.deg2rad(-30) <= azimuth <= np.deg2rad(30):  
                        if distance < 10:  
                            obstacle_detected = True
                            break
                if not obstacle_detected:
                    self.mode_flag.is_ca = False
                    self.mode_flag.is_pf = True
                    self.publish_flags()
                    # self.publish_obstacle_info(x,y,z)
                    # self.get_logger().info(f"is_pf: distance={distance}, azimuth={np.degrees(azimuth)}")
 
    def publish_obstacle_info(self, obstacle_x, obstacle_y, obstacle_z):
        msg = Float32MultiArray()
        msg.data = [
                float(obstacle_x),
                float(obstacle_y),
                float(obstacle_z)
            ]
        self.obstacle_publisher_.publish(msg)
        
    def flag_callback(self, msg):
        self.mode_flag.is_offboard = msg.is_offboard
        self.mode_flag.is_ca = msg.is_ca
        self.mode_flag.is_pf = msg.is_pf
    
    def publish_flags(self):
        msg = StateFlag()
        msg.is_pf = self.mode_flag.is_pf
        msg.is_ca = self.mode_flag.is_ca
        print(f"CA: is_pf: {msg.is_pf}, is_ca: {msg.is_ca}")
        self.flag_publisher.publish(msg)
    


    # def generate_waypoint(self):
    #     # 현재 위치 및 yaw 가져오기
    #     x0, y0, z0 = self.state_var.x, self.state_var.y, self.state_var.z
    #     yaw = self.heading

    #     # 7미터 전방 웨이포인트 계산
    #     wx = x0 + 10 * np.cos(yaw)
    #     wy = y0 + 10 * np.sin(yaw)
    #     wz = z0  # 고도는 유지

    #     # 웨이포인트 퍼블리싱
    #     waypoint = Point()
    #     waypoint.x = wx
    #     waypoint.y = wy
    #     waypoint.z = wz
    #     self.temp_waypoint_publisher.publish(waypoint)
    #     self.get_logger().info(f"Waypoint published: N={wx}, E={wy}, D={wz}")

def main(args=None):
    rclpy.init(args=args)
    collision_avoidance_node = CollisionAvoidanceNode()
    rclpy.spin(collision_avoidance_node)
    collision_avoidance_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
