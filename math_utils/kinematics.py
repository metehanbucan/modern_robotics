import numpy as np
from math_utils.se3 import matrix_exp3, skew, so3_to_vec

def matrix_exp6(se3mat):
    omgmat = se3mat[0:3,0:3]
    v = se3mat[0:3,3]
    if np.linalg.norm(so3_to_vec(omgmat)) < 1e-6:
        T = np.eye(4)
        T[0:3, 3] = v
        return T
    theta = np.linalg.norm(so3_to_vec(omgmat))
    unitomgmat = omgmat / theta
    R = matrix_exp3(omgmat)
    G = (np.eye(3) * theta + (1- np.cos(theta)) * unitomgmat + (theta - np.sin(theta)) * np.dot(unitomgmat, unitomgmat))
    p = np.dot(G, v/theta)
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3,3] = p
    return T

def vec_to_se3(vec):
    omg = vec[0:3]
    v = vec[3:6]
    omgskew = skew(omg)
    se3 = np.zeros((4,4))
    se3[0:3, 0:3] = omgskew
    se3[0:3, 3] = v
    return se3

def FkinSpace(M, SList, ThetaList):
    T = np.eye(4)
    for i in range (len(ThetaList)):
        T = T @ matrix_exp6(vec_to_se3(SList[:,i] * ThetaList[i]))
    return T @ M

