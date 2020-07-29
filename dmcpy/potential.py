import numpy as np
from numba import njit
from dmcpy.randwk_geo import ccplate as rw_ccplate

# Disk Potential
@njit('float64(uint32, float64[::1], float64[::1], float64)')
def ccplate(n, p, p0, r0):
    """Number of simulations, Current position, Center of disk, Radius of disk"""
	
    v = 0.
    p[0] = 0.
    p_init = p.copy()

    for i in range(n): 
        
        while True:
            p = rw_ccplate(p, p0, r0)
       
            if p[0] == 1.: break
            
        v += p[1]
        p = p_init

    v /= n

    return v
        

if __name__ == '__main__':
    from time import time
    n = 1000000
    t0 = time()
    poten = ccplate(n, np.array([0., 0., 0., np.sqrt(3)]),np.array([0., 0., 0., 0.]), 1.)
    # potential = 1/3
    t1 = time()
    print("Number: {}".format(n))
    print("Potential: {}".format(poten))
    print("Time: {}".format(t1-t0))