import numpy as np
from math_utils.kinematics import *
from math_utils.jacobian import *
from math_utils.ik import *

M = np.eye(4)
M[0:3, 3] = np.array([2,0,0])

Slist = np.array([
    [0,0],
    [0,0],
    [1,1],
    [0,0],
    [0,-1],
    [0,0]
])

T_target = FkinSpace(M, Slist, np.array([np.pi/6 , np.pi/3]))

theta_guess = np.array([np.pi/4,np.pi])

theta_sol, success = IKinSpace(Slist, M, T_target, theta_guess, max_iter=500)

print(theta_sol, success)
print(np.array([np.pi/6 , np.pi/3]))
theta1, theta2 = analytic_inverse_kinematics_2r(T_target)
print(theta1, theta2)