import numpy as np
from math_utils.kinematics import *

M = np.eye(4)
M[:,3] = [2,0,0,1]

SList = np.array([
    [0,0],
    [0,0],
    [1,1],
    [0,0],
    [0,-1],
    [0,0]
])

ThetaList = np.array([np.pi/4 , np.pi/4])

T = FkinSpace(M,SList,ThetaList)

print(T)