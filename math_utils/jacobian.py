import numpy as np
from math_utils.transformations import TransToRp
from math_utils.kinematics import matrix_exp6, vec_to_se3

def adjoint(T):
    R,p = TransToRp(T)
    skewp = np.array([[0,-p[2], p[1]], 
                      [p[2], 0, -p[0]],
                      [-p[1], p[0], 0]])
    adT = np.zeros((6,6))
    adT[0:3, 0:3] = R
    adT[3:6, 3:6] = R
    adT[3:6, 0:3] = skewp @ R
    
    return adT

def JacobianSpace(Slist, thetalist):
    n = len(thetalist)
    Js = np.zeros((6,n))
    Js[:,0] = Slist[:,0]
    T = np.eye(4)

    for i in range (1,n):
        T = T @ matrix_exp6(vec_to_se3(Slist[:,i-1] * thetalist[i-1]))
        Js[:,i] = adjoint(T) @ Slist[:,i]

    return Js