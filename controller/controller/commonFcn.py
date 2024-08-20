# Library for common functions
import numpy as np
import math


# convert quaternion to euler
def Quaternion2Euler(self, w, x, y, z):

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    Roll = math.atan2(t0, t1) * 57.2958

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Pitch = math.asin(t2) * 57.2958

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Yaw = math.atan2(t3, t4) * 57.2958

    return Roll, Pitch, Yaw


# convert Body frame to NED frame
def BodytoNED(self):
    vel_cmd_body = np.array(
        [self.vel_cmd_body_x, self.vel_cmd_body_y, self.vel_cmd_body_z]
    )

    self.veh_trj_set.vel_NED = np.array((self.DCM_bn @ vel_cmd_body).tolist())


# convert NED frame to Body frame
def NEDtoBody(self):
    vel_body = np.array([self.v_x, self.v_y, self.v_z])

    self.u, self.v, self.w = np.array((self.DCM_nb @ vel_body).tolist())


# calculate DCM matrix
def DCM(_phi, _theta, _psi):
    PHI = math.radians(_phi)
    THETA = math.radians(_theta)
    PSI = math.radians(_psi)

    mtx_DCM = np.array(
        [
            [
                math.cos(PSI) * math.cos(THETA),
                math.sin(PSI) * math.cos(THETA),
                -math.sin(THETA),
            ],
            [
                (-math.sin(PSI) * math.cos(PHI))
                + (math.cos(PSI) * math.sin(THETA) * math.sin(PHI)),
                (math.cos(PSI) * math.cos(PHI))
                + (math.sin(PSI) * math.sin(THETA) * math.sin(PHI)),
                math.cos(THETA) * math.sin(PHI),
            ],
            [
                (math.sin(PSI) * math.sin(PHI))
                + (math.cos(PSI) * math.sin(THETA) * math.cos(PHI)),
                (-math.cos(PSI) * math.sin(PHI))
                + (math.sin(PSI) * math.sin(THETA) * math.cos(PHI)),
                math.cos(THETA) * math.cos(PHI),
            ],
        ]
    )

    return mtx_DCM
