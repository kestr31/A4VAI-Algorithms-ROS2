# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node  # rclpy의 node class 사용

  # topic을 지나가는 data를 구조화하기 위해 built-in string message type import
from custom_msgs.msg import CustomMsg  # topic을 지나가는 data를 구조화하기 위해
import cv2
import numpy as np
from cv_bridge import CvBridge

# 위 줄은 node의 dependencies를 나타내며, package.xml에 포함되어야 함

class MinimalPublisher(Node):  # MinimalPublisher는 Node로부터 상속됨 (Node의 subclass)

    def __init__(self):
        super().__init__('minimal_publisher')  # Node class의 custructor를 부르고 minimal_publisher라는 node 이름 줌
        
        self.publisher_ = self.create_publisher(CustomMsg, 'topic', 10)  # node가 topic으로 String 타입의 message를 publish하고 que size는 10이라고 선언 
        
        timer_period = 0.5  # seconds  # 매 0.5초마다 실행하는 callback과 함께 timer 생성
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0  # callback에서 사용되는 counter
        # self.cv_image = cv2.imread('256-001.png')  # ws_ros2 폴더 (workspace)에 있어야 함!!!
        self.bridge = CvBridge()


    def timer_callback(self):  # 더해지는 counter 값과 함께 message를 생성하고, 이를 get_logger().info와 함께 console창에 publish한다.
        
        # 1: 이미지 경로, 2: 시작점, 3: 도착점, 4: Mode
        msg = CustomMsg()
        msg.image_path = '/home/user/px4_ros_ws/src/PathPlaning_data/1000-003.png'  # 1. 이미지 경로 설정
        msg.init_custom = [100.0, 2.0, 100.0]  # 2. 시작점   // float
        msg.target_custom = [950.0, 2.0, 950.0]  # 3. 도착점  // float
        msg.mode = 3  # 4. Mode 설정

        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: {}'.format(msg))
        self.i += 1


def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin_once(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
