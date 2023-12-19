#.. public libaries
import numpy as np
import math as m

#.. ROS libraries
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Int32MultiArray

#.. PF algorithms libararies
# from .testpy1 import testpypy1
from .quadrotor_6dof import Quadrotor_6DOF
from .virtual_target import Virtual_Target
from .set_parameter import MPPI_Guidance_Parameter, Way_Point
from .MPPI_guidance import MPPI_Guidance_Modules



class Node_MPPI_Output(Node):
    
    def __init__(self):
        super().__init__('node_MPPI_output')
        
        #.. Reference        
        #   [1]: https://hostramus.tistory.com/category/ROS2
        
        #.. publishers - from ROS2 msgs to ROS2 msgs
        self.MPPI_output_publisher_     =   self.create_publisher(Float64MultiArray, 'MPPI/out/dbl_MPPI', 10)

        #.. subscriptions - from ROS2 msgs to ROS2 msgs
        self.MPPI_input_int_Q6_subscription =   self.create_subscription(Int32MultiArray, 'MPPI/in/int_Q6', self.subscript_MPPI_input_int_Q6, qos_profile_sensor_data)
        self.MPPI_input_dbl_Q6_subscription =   self.create_subscription(Float64MultiArray, 'MPPI/in/dbl_Q6', self.subscript_MPPI_input_dbl_Q6, qos_profile_sensor_data)
        self.MPPI_input_dbl_VT_subscription =   self.create_subscription(Float64MultiArray, 'MPPI/in_dbl_VT', self.subscript_MPPI_input_dbl_VT, qos_profile_sensor_data)
                
        ###### - start - Vars. for PF algorithm ######
        #.. declare variables/instances
        self.Q6     =   Quadrotor_6DOF()
        self.VT     =   Virtual_Target()
        
        #.. set waypoint
        wp_type_selection   =   1       # | 0: straight line | 1: ractangle | 2: circle | 3: designed
        self.WP     =   Way_Point(wp_type_selection)

        #.. MPPI setting
        # parameter
        type_MPPI   =   self.Q6.Guid_type   # 0~1: no use MPPI | 2: direct accel cmd | 3: guidance-based |
        self.MP     =   MPPI_Guidance_Parameter(type_MPPI)
        
        # callback PF_MPPI_param
        period_MPPI_param       =   self.MP.dt
        self.timer  =   self.create_timer(period_MPPI_param, self.PF_MPPI_param)
        
        # module
        self.MG      =   MPPI_Guidance_Modules(self.MP)
        # initialization
        self.MG.set_total_MPPI_code(self.WP.WPs.shape[0])
        self.MG.set_MPPI_entropy_calc_code()
        
        self.MPPI_ctrl_input = np.array([self.MP.u1_init, self.MP.u2_init, self.MP.u3_init])
        print(self.MPPI_ctrl_input)
        
        ###### -  end  - Vars. for PF algorithm ######
        
        
    ### main function
    #.. PF_MPPI_param 
    def PF_MPPI_param(self):
        #.. MPPI algorithm
        MPPI_ctrl_input1, MPPI_ctrl_input2    =   self.MG.run_MPPI_Guidance(self.Q6, self.WP.WPs, self.VT)
        self.MPPI_ctrl_input    =   MPPI_ctrl_input1.copy()
        self.publish_MPPI_output()
        # print("MPPI: [0]=" + str(self.MPPI_ctrl_input[0]) + ", [1]=" + str(self.MPPI_ctrl_input[1]) + ", [2]=" + str(self.MPPI_ctrl_input[2]))
        pass
    
    
                    
    ### publushers

    #.. publish_MPPI_output
    def publish_MPPI_output(self):
        msg                 =   Float64MultiArray()
        msg.data            =   [self.MPPI_ctrl_input[0], self.MPPI_ctrl_input[1], self.MPPI_ctrl_input[2]]
        self.MPPI_output_publisher_.publish(msg)
        self.get_logger().info('publish_MPPI_output msgs: {0}'.format(msg.data))
        # self.get_logger().info("subscript_MPPI_output: [0]=" + str(self.MPPI_ctrl_input[0]) +", [1]=" + str(self.MPPI_ctrl_input[1]) +", [2]=" + str(self.MPPI_ctrl_input[2]))
        pass
        
    
        
    ### subscriptions
    #.. subscript_MPPI_input_int_Q6
    def subscript_MPPI_input_int_Q6(self, msg):
        self.Q6.WP_idx_heading  =   msg.data[0]
        self.Q6.WP_idx_passed   =   msg.data[1]
        self.Q6.Guid_type       =   msg.data[2]
        self.Q6.flag_guid_trans =   msg.data[3]
        # self.get_logger().info('subscript_MPPI_input_int_Q6 msgs: {0}'.format(msg.data))
        pass
    
    #.. subscript_MPPI_input_dbl_Q6
    def subscript_MPPI_input_dbl_Q6(self, msg):
        self.Q6.throttle_hover      =   msg.data[0]
        tmp_float                   =   msg.data[1]
        self.Q6.desired_speed       =   msg.data[2]
        self.Q6.look_ahead_distance =   msg.data[3]
        self.Q6.distance_change_WP  =   msg.data[4]
        self.Q6.Kp_vel              =   msg.data[5]
        self.Q6.Kd_vel              =   msg.data[6]
        self.Q6.Kp_speed            =   msg.data[7]
        self.Q6.Kd_speed            =   msg.data[8]
        self.Q6.guid_eta            =   msg.data[9]
        self.Q6.tau_phi             =   msg.data[10]
        self.Q6.tau_the             =   msg.data[11]
        self.Q6.tau_psi             =   msg.data[12]
        self.Q6.Ri[0]               =   msg.data[13]
        self.Q6.Ri[1]               =   msg.data[14]
        self.Q6.Ri[2]               =   msg.data[15]
        self.Q6.Vi[0]               =   msg.data[16]
        self.Q6.Vi[1]               =   msg.data[17]
        self.Q6.Vi[2]               =   msg.data[18]
        self.Q6.Ai[0]               =   msg.data[19]
        self.Q6.Ai[1]               =   msg.data[20]
        self.Q6.Ai[2]               =   msg.data[21]
        self.Q6.thr_unitvec[0]      =   msg.data[22]
        self.Q6.thr_unitvec[1]      =   msg.data[23]
        self.Q6.thr_unitvec[2]      =   msg.data[24]
        # self.get_logger().info('subscript_MPPI_input_dbl_Q6 msgs: {0}'.format(msg.data))
        pass
            
    #.. subscript_MPPI_input_dbl_VT
    def subscript_MPPI_input_dbl_VT(self, msg):
        self.VT.Ri[0]   =   msg.data[0]
        self.VT.Ri[1]   =   msg.data[1]
        self.VT.Ri[2]   =   msg.data[2]
        # self.get_logger().info('subscript_MPPI_input_int32 msgs: {0}'.format(msg.data))
        pass
    
def main(args=None):
    print("======================================================")
    print("------------- main() in node_MPPI_output.py -------------")
    print("======================================================")
    rclpy.init(args=args)
    MPPI_Output = Node_MPPI_Output()
    rclpy.spin(MPPI_Output)
    MPPI_Output.destroy_node()
    rclpy.shutdown()
    pass
if __name__ == '__main__':
    main()


