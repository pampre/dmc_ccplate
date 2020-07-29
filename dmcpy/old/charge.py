import numpy as np 
from numba import njit
from dmcpy.sample import hemisphere
from dmcpy.potential import disk

# Surface charge density on a disk
# Number of sampling --> n
# Gemoetry of disk: Center of disk, Radius of disk --> r0, [0, x0, y0, z0]  
# Point of investigtion, Radius of LP-hemisphere  --> r1, [0, x1, y1, z1] 
# --> Charge density 
@njit('float64(uint32, float64[::1], float64, float64[::1], float64)')
def disk(n, p0, r0, p1, r1):
    distance = np.linalg.norm(p1 - p0)
    
    if ((distance > r0) or (p0[3] != p1[3]) or ((r0 - distance) < r1)):
        return 0.

    sigma = 0.
    
    factor = 3. / (16 * r1)
    
    for k in range(n):
        p = hemisphere(p1, r1)
        v = disk(1, p, p0, r0)
        sigma += (1-v)
    
    sigma *= factor/n
    
    return sigma


if __name__ == '__main__':
    from time import time        
    n = 100000
    p1 = np.array([0., 0., 0., 0.])
    p0 = np.array([0., 0., 0., 0.])
    t0 = time()
    c = charge_disk(n, p0, 1., p1, 1.)
    t1 = time()
    print("Number: {}".format(n))
    print("Charge: {}".format(c))
    print("Time: {}".format(t1-t0))

