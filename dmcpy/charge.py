import numpy as np
from numba import njit, float64, uint64
from dmcpy.sample import hemisphere, annulus
from dmcpy.potential import ccplate as v_disk

# Surface charge density on a disk
# Number of sampling --> n
# Gemoetry of disk: Center of disk, Radius of disk --> r0, [0, x0, y0, z0]  
# Point of investigtion, Radius of LP-hemisphere  --> r1, [0, x1, y1, z1] 
# --> Charge density 
@njit(float64(uint64))
def ccplate(n):
    # Initialization
    p0 = np.array([0., 0., 0., 0.])
    p1 = np.array([0., 0., 0., 0.])
    r0 = 1.
    r1 = 2.

    distance = np.linalg.norm(p1 - p0)
    r_in = r0 - distance

    if (r_in < 0) or (p0[3] != p1[3]):
        return 0.

    else:
        if r_in >= r1:
            
            sigma = 0.
            factor = 3. / (16 * r1)

            p = hemisphere(p1, r1)
            v = v_disk(p, p0, r0)
            sigma += (1-v)
            
            sigma *= factor
            
            return sigma

        else:
            # sigma1 part
            sigma1 = 0.
            factor1 = 3. / (16 * r1)

            # sigma2 part
            sigma2 = 0.
            r1_inv_3 = np.power(r1, -3)

            for i in range(n):
                p = hemisphere(p1, r1)
                v = v_disk(p, p0, r0)
                sigma1 += (1-v)

                p = annulus(p1, r_in, r1)
                s = p[0]
                v = v_disk(p, p0, r0)

                if v == 0.:
                    s_inv_3 = s ** (-3)
                    sigma2 += (s_inv_3 - r1_inv_3)

            sigma1 *= factor1
            sigma2 *= (r1 * r1 - r_in * r_in) / 16.

            sigma = (sigma1 + sigma2)

            return sigma
