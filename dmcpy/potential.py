import numpy as np
from numba import njit
from dmcpy.randwk_geo import ccplate as rw_ccplate

# Disk Potential
@njit('float64(float64[::1], float64[::1], float64)')
def ccplate(p, p0, r0):
    """Number of simulations, Current position, Center of disk, Radius of disk"""
    p[0] = 0.

    while True:
        p = rw_ccplate(p, p0, r0)

        if p[0] == 1.: break

    return p[1]