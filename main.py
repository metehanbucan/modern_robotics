import numpy as np
from math_utils.kinematics import *
from math_utils.jacobian import *

Slist = np.array([
    [0,0],
    [0,0],
    [1,1],
    [0,0],
    [0,-1],
    [0,0]
])

ThetaList = np.array([np.pi/4 , np.pi/4])

Js = JacobianSpace(Slist, ThetaList)

print(Js)

thetadot = np.array([0.5, 0.5])

V = Js @ thetadot

print("end effector velocity:")
print(V)