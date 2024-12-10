# This file is used to set the waypoint
# ----------------------------------------------------------------------------------------#
# region LIBRARIES
import pandas as pd
import numpy as np
import math

# endregion
# ----------------------------------------------------------------------------------------#


def set_waypoint():
    waypoint = pd.read_csv("waypoint.csv")
    return waypoint


def convert_quaternion2euler(w, x, y, z):

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    Roll = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Pitch = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Yaw = math.atan2(t3, t4)

    return Roll, Pitch, Yaw

# convert Body frame to NED frame
def BodytoNED(vx_body_cmd, vy_body_cmd, vz_body_cmd, dcm):
    vel_cmd_body = np.array(
        [
            vx_body_cmd,
            vy_body_cmd,
            vz_body_cmd,
        ]
    )
    vel_ned_cmd = np.array((dcm @ vel_cmd_body).tolist())
    return vel_ned_cmd

# convert NED frame to Body frame
def NEDtoBody(self):
    vel_body = np.array([self.v_x, self.v_y, self.v_z])

    self.u, self.v, self.w = np.array((self.DCM_nb @ vel_body).tolist())


# calculate DCM matrix
#.. get DCM from Euler angle
def DCM_from_euler_angle( EulerAng ):

    # Local Variable 
    spsi            =   math.sin( EulerAng[2] )
    cpsi            =   math.cos( EulerAng[2] )
    sthe            =   math.sin( EulerAng[1] )
    cthe            =   math.cos( EulerAng[1] )
    sphi            =   math.sin( EulerAng[0] )
    cphi            =   math.cos( EulerAng[0] )
    
    # DCM from 1-frame to 2-frame
    c1_2            =   np.zeros((3,3))
    
    c1_2[0,0]       =   cpsi * cthe 
    c1_2[1,0]       =   cpsi * sthe * sphi - spsi * cphi 
    c1_2[2,0]       =   cpsi * sthe * cphi + spsi * sphi 
    
    c1_2[0,1]       =   spsi * cthe 
    c1_2[1,1]       =   spsi * sthe * sphi + cpsi * cphi 
    c1_2[2,1]       =   spsi * sthe * cphi - cpsi * sphi 
    
    c1_2[0,2]       =   -sthe 
    c1_2[1,2]       =   cthe * sphi 
    c1_2[2,2]       =   cthe * cphi 

    return c1_2
