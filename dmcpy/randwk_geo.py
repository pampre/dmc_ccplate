import numpy as np
from numba import njit
from dmcpy.randwk import wop_z, bts, wos

# Disk Diffusion
# Current position, Center of disk, Radius of disk --> Spherical Surface
@njit('float64[::1](float64[::1], float64[::1], float64)')
def disk(p, p0, a0):
    
    z = p[3]
    z0 = p0[3]

    # b0 determines whether to use back-to-sphere
    b0 = 1.01 * a0

    r = np.linalg.norm(p - p0)

    if z == z0:
        if r < a0:
            return np.array([1., 1., 0., 0.])
        
        else:
            return wos(p, (r - a0))

    else:
        if r < b0:   
            q = p.copy()
            q[3] = z0
            return wop_z(p, q)
        
        else:
            u = np.random.rand()
            if a0 < (r * u):
                return np.array([1., 0., 0., 0.])

            else:    
                return bts(p, p0, a0)

