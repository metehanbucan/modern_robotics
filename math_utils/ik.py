import numpy as np
from math_utils.se3 import matrix_log3, so3_to_vec
from math_utils.transformations import TransToRp
from math_utils.jacobian import JacobianSpace
from math_utils.kinematics import FkinSpace, vec_to_se3
def matrix_log6(T):
    R,p = TransToRp(T)
    omgmat = matrix_log3(R)
    omega = so3_to_vec(omgmat)

    if np.linalg.norm(omega) < 1e-6:
        se3mat = np.zeros((4,4))
        se3mat[0:3,3] = p
        return se3mat
    
    theta = np.linalg.norm(omega)
    omgmat_unit = omgmat / theta

    G_inv = (np.eye(3) / theta - 0.5 * omgmat_unit + (1/theta - 0.5 / np.tan(theta / 2)) * (omgmat_unit @ omgmat_unit))

    v = G_inv @ p

    se3mat = np.zeros((4,4))
    se3mat[0:3, 0:3] = omgmat
    se3mat[0:3, 3] = v * theta

    return se3mat

def se3_to_vec(se3mat):
    return np.array([
        se3mat[2,1],
        se3mat[0,2],
        se3mat[1,0],
        se3mat[0,3],
        se3mat[1,3],
        se3mat[2,3]
    ])

def IKinSpace(Slist, M, Tsd, thetalist0, eomg = 1e-3, ev= 1e-3, max_iter = 50):
    thetalist = thetalist0.copy()
    
    for i in range(max_iter):
        Tsb = FkinSpace(M, Slist, thetalist)
        Tbd = np.linalg.inv(Tsb) @ Tsd

        Vb = se3_to_vec(matrix_log6(Tbd))
        err_omg = np.linalg.norm(Vb[0:3])
        err_v = np.linalg.norm(Vb[3:6])

        if err_omg < eomg and err_v < ev:
            return thetalist, True
        
        J = JacobianSpace(Slist, thetalist)
        thetalist += np.linalg.pinv(J) @ Vb

    return thetalist, False

def analytic_inverse_kinematics_2r(T_target):
    R,p = TransToRp(T_target)
    x = p[0]
    y = p[1]

    phi = np.atan2(y,x)
    r= (x**2 + y**2)**0.5
    beta1 = np.arccos(r/2)
    theta1 = phi - beta1

    beta2 = np.arccos((2-r**2) / 2)
    theta2 = np.pi - beta2

    return theta1, theta2
