import numpy as np
from math_utils.se3 import *

omega = np.array([0,0,1])
so3 = vec_to_so3(omega)
R = matrix_exp3(so3)

print('rotation matrix:')
print(R)

log = matrix_log3(R)
print(log)
omeganew = so3_to_vec(log)
print(omeganew)