import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import math

#   ROS2 python 
import rclpy
from rclpy.node import Node


from px4_msgs.msg import Timesync
from msg_srv_act_interface.srv import PathFollowingSetpoint
from msg_srv_act_interface.srv import PathFollowingGPR
from msg_srv_act_interface.srv import PathFollowingGuid

class PathFollowingService(Node):
    def __init__(self):
        super().__init__('following_service')

        self.PathFollowingServiceClient_ = self.create_client(PathFollowingSetpoint, 'path_following_att_cmd')
        while not self.PathFollowingServiceClient_.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Path Following Setpoint Service not available, waiting again...') 
        
    def RequestPathFollowing(self, waypoint_x, waypoint_y, waypoint_z):
        self.path_following_request = PathFollowingSetpoint.Request()
        self.path_following_request.request_timestamp = self.timestamp
        self.path_following_request.request_pathfollowing = True
        self.path_following_request.waypoint_x = waypoint_x
        self.path_following_request.waypoint_y = waypoint_y
        self.path_following_request.waypoint_z = waypoint_z
        self.future_setpoint = self.PathFollowingServiceClient_.call_async(self.path_following_request)
        
    def TimesyncCallback(self, msg):
        self.timestamp = msg.timestamp
        

class PathFollowingGPRService(Node):
    def __init__(self):
        super().__init__('following_gpr_service')
        self.qosProfileGen()
        self.TimesyncSubscriber_ = self.create_subscription(Timesync, '/fmu/time_sync/out', self.TimesyncCallback, self.QOS_Sub_Sensor)
        self.declare_service_client_custom()
        self.timestamp = 0
        
    def declare_service_client_custom(self): 
        self.PathFollowingGPRServiceClient_ = self.create_client(PathFollowingGPR, 'path_following_gpr')
        while not self.PathFollowingGPRServiceClient_.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Path Following GPR Service not available, waiting again...') 

        
    def qosProfileGen(self):
        #   Reliability : 데이터 전송에 있어 속도를 우선시 하는지 신뢰성을 우선시 하는지를 결정하는 QoS 옵션
        #   History : 데이터를 몇 개나 보관할지를 결정하는 QoS 옵션
        #   Durability : 데이터를 수신하는 서브스크라이버가 생성되기 전의 데이터를 사용할지 폐기할지에 대한 QoS 옵션
        self.QOS_Sub_Sensor = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            durability=QoSDurabilityPolicy.VOLATILE)
        
        self.QOS_Service = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE)

    def RequestPathFollowingGPR(self):
        self.path_following_gpr_request = PathFollowingGPR.Request()
        self.path_following_gpr_request.request_timestamp = self.timestamp
        self.path_following_gpr_request.request_gpr = True
        self.future_gpr = self.PathFollowingGPRServiceClient_.call_async(self.path_following_gpr_request)
        
    def TimesyncCallback(self, msg):
        self.timestamp = msg.timestamp




class PathFollowingGuidService(Node):
    def __init__(self):
        super().__init__('following_guid_service')
        self.qosProfileGen()
        self.TimesyncSubscriber_ = self.create_subscription(Timesync, '/fmu/time_sync/out', self.TimesyncCallback, self.QOS_Sub_Sensor)
        self.declare_service_client_custom()
        self.timestamp = 0
        
    def declare_service_client_custom(self): 
        self.PathFollowingGuidServiceClient_ = self.create_client(PathFollowingGuid, 'path_following_guid')
        while not self.PathFollowingGuidServiceClient_.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Path Following Guid Service not available, waiting again...') 
    
    def qosProfileGen(self):
        #   Reliability : 데이터 전송에 있어 속도를 우선시 하는지 신뢰성을 우선시 하는지를 결정하는 QoS 옵션
        #   History : 데이터를 몇 개나 보관할지를 결정하는 QoS 옵션
        #   Durability : 데이터를 수신하는 서브스크라이버가 생성되기 전의 데이터를 사용할지 폐기할지에 대한 QoS 옵션
        self.QOS_Sub_Sensor = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            durability=QoSDurabilityPolicy.VOLATILE)
        
        self.QOS_Service = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE)
    
    def RequestPathFollowingGuid(self, waypoint_x, waypoint_y, waypoint_z, flag_guid_param):
        self.path_following_guid_request = PathFollowingGuid.Request()
        self.path_following_guid_request.request_timestamp = self.timestamp
        self.path_following_guid_request.request_guid = True
        self.path_following_guid_request.waypoint_x = waypoint_x
        self.path_following_guid_request.waypoint_y = waypoint_y
        self.path_following_guid_request.waypoint_z = waypoint_z
        self.path_following_guid_request.flag_guid_param = flag_guid_param
        self.future_guid = self.PathFollowingGuidServiceClient_.call_async(self.path_following_guid_request)



        #####
        # QOS setting is needed
        # It will be added after further study
        #####