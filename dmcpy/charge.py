import numpy as np 
from numba import njit, jit
from dmcpy.sample import hemisphere, annulus
from dmcpy.potential import ccplate as v_disk

# Surface charge density on a disk
# Number of sampling --> n
# Gemoetry of disk: Center of disk, Radius of disk --> r0, [0, x0, y0, z0]  
# Point of investigtion, Radius of LP-hemisphere  --> r1, [0, x1, y1, z1] 
# --> Charge density 
@njit('float64(uint32, float64[::1], float64, float64[::1], float64)')
def ccplate(n, p0, r0, p1, r1):
    distance = np.linalg.norm(p1 - p0)
    r_in = r0 - distance

    if ((r_in < 0) or (p0[3] != p1[3])):
        return 0.

    else:
        if (r_in >= r1):
            
            sigma = 0.
            factor = 3. / (16 * r1)
            
            for k in range(n):
                p = hemisphere(p1, r1)
                v = v_disk(1, p, p0, r0)
                sigma += (1-v)
            
            sigma *= factor/n
            
            return sigma

        else:          
            sigma1 = 0.
            sigma2 = 0.  
            factor1 = 3. / (16 * r1)
        
            for k in range(n):
                p = hemisphere(p1, r1)
                v = v_disk(1, p, p0, r0)
                sigma1 += (1-v)
            sigma1 *= factor1
            
            r1_inv_3 = np.power(r1, -3)
            for k in range(n):
                p = annulus(p1, r_in, r1)
                s = p[0]
                v = v_disk(1, p, p0, r0)

                if v == 0.:
                    s_inv_3 = np.power(s, -3)
                    sigma2 += (s_inv_3 - r1_inv_3)    
            sigma2 *= (r1*r1 - r_in*r_in) / 16. 

            sigma = (sigma1 + sigma2) / n

            return sigma

if __name__ == '__main__':
    from time import time        
    n = 100000
    p1 = np.array([0., 0., 0., 0.])
    p0 = np.array([0., 0., 0., 0.])
    t0 = time()
    charge = disk(n, p0, 1., p1, 2.)
    t1 = time()
    print("Number: {}".format(n))
    print("Charge: {}".format(charge))
    print("Time: {}".format(t1-t0))