import numpy as np
from math_utils.transformations import *

R = np.eye(3)
p = np.array([1,2,3])

T = RpToTrans(R,p)
print("Transformation:")
print(T)

newR, newP = TransToRp(T)
print("recovered:", newR, newP)

TInv = TransInv(T)
print("TransInverse:")
print(TInv)

point = np.array([1,0,0,1])
print("transformed point:")
print(np.dot(T,point))