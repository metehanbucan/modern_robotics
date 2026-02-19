import numpy as np
from math_utils.se3 import *

w = np.array([1,2,3])
so3 = vec_to_so3(w)
vec = so3_to_vec(so3)
print(so3)
print(vec)