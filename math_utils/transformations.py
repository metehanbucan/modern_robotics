import numpy as np

def RpToTrans(R, p):
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = p
    return T

def TransToRp(T):
    R = T[0:3,0:3]
    p = T[0:3, 3]
    return R,p

def TransInv(T):
    R,p = TransToRp(T)
    TInv = np.eye(4)
    TInv[0:3,0:3] = R.T
    TInv[0:3, 3] = -np.dot(R.T, p)
    return TInv