import numpy as np 
from numba import njit

# angle between two vectors --> bin number
@njit('uint32(float64[::1], float64[::1])')
def anlge(p, q):

    norm_p = np.sqrt(p[1]*p[1] + p[2]*p[2] + p[3]*p[3])
    norm_q = np.sqrt(q[1]*q[1] + q[2]*q[2] + q[3]*q[3])

    cos_ang = (p[1] * q[1] + p[2] * q[2] + p[3] * q[3]) / (norm_p * norm_q)
    
    i = np.floor(np.arccos(cos_ang) * 100. / np.pi)
    
    return i