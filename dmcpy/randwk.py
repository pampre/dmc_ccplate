import numpy as np
from numba import njit

#list: Walk-on-sphere, Off-centered-WOS, Back-to-sphere, Walk-on-plane-z.


# Walk-on-sphere
# Current position, Radius of Sphere --> Spherical Surface
@njit('float64[::1](float64[::1], float64)')
def wos(p, a):
    x = p[1]
    y = p[2]
    z = p[3]
    u = np.random.rand()
    phi = np.random.rand() * 2. * np.pi 
    cos_th = 1. - 2. * u
    sin_th = np.sqrt(1 - cos_th * cos_th)
    x += np.cos(phi) * sin_th * a
    y += np.sin(phi) * sin_th * a
    z += cos_th * a

    return np.array([0., x, y, z])


# Off centered wos
# Current Position, Center of Sphere, Radius of Sphere --> Spherical Surface
@njit('float64[::1](float64[::1], float64[::1], float64)')
def offwos(p, p0, a):
  # q is the relative position
    q = p - p0
    x = q[1]
    y = q[2]
    z = q[3]
    r = np.linalg.norm(q)

    # d is the ratio between distance and radius
    d = r / a
    
    # cos_th from the inverse of the cdf
    u = np.random.rand()
    phi = 2. * np.pi * np.random.rand()
    c1 = 1. + d
    c2 = 1. + d*d
    cos_th = (c1*c1 - 2.*c1*c2*u + 2.*d*c2*u*u) / np.square(1.+d-2.*d*u)
    sin_th = np.sqrt(1. - cos_th * cos_th)

    # random sampling point on the enclosing sphere from the point on z-axis with the same r as p.
    vx = sin_th * np.cos(phi) * a
    vy = sin_th * np.sin(phi) * a
    vz = cos_th * a

    if z == r:
        return np.array([0., vx, vy, vz])
    
    # rotation angle 'a' with rotation axis K = (-y, x, 0)
    else:
        cos_a = z / r
        sin_a = np.sqrt(1. - cos_a * cos_a)
        
        K2 = x * x + y * y
        K = np.sqrt(K2)   
        Kv = -y * vx + x * vy

        # x1 = vx * cos_a + (x * vz) * sin_a / K - y * Kv * (1. - cos_a) / K2
        # y1 = vy * cos_a + (y * vz) * sin_a / K + x * Kv * (1. - cos_a) / K2
        # z1 = vz * cos_a - (x * vx + y * vy) * sin_a / K

        opt1 = vz * sin_a / K
        opt2 = Kv * (1 - cos_a) / K2
 
        x1 = vx * cos_a + x * opt1 - y * opt2
        y1 = vy * cos_a + y * opt1 + x * opt2
        z1 = vz * cos_a - (x * vx + y * vy) * sin_a / K

        return np.array([0., x1, y1, z1]) + p0


# Back-to-sphere
# Current Position, Center of Sphere, Radius of Sphere --> Spherical Surface
@njit('float64[::1](float64[::1], float64[::1], float64)')
def bts(p, p0, a):
    # q is the relative position
    q = p - p0
    x = q[1]
    y = q[2]
    z = q[3]
    r = np.linalg.norm(q)
    
    # d is the ratio between distance and radius
    d = r / a
    
    # cos_th from the inverse of the cdf
    u = np.random.rand()
    phi = 2. * np.pi * np.random.rand()
    cos_th = (d*(1. + d)*(1. + d) - 2.*(1. + d + d*d + d*d*d)*u + 2.*(1. + d*d)*u*u) / (d*(1.+d-2.*u)*(1.+d-2.*u))
    sin_th = np.sqrt(1. - cos_th * cos_th)

    # random sampling point on the enclosing sphere from the point on z-axis with the same r as p.
    vx = sin_th * np.cos(phi) * a
    vy = sin_th * np.sin(phi) * a
    vz = cos_th * a

    if z == r:
         return np.array([0., vx, vy, vz])
    
    # Rodrigues' Rotation Formula
    # rotation angle 'a' with rotation axis K = (-y, x, 0)
    else:
        cos_a = z / r
        sin_a = np.sqrt(1. - cos_a * cos_a)
        
        K2 = x * x + y * y
        K = np.sqrt(K2)   
        Kv = -y * vx + x * vy

        # x1 = vx * cos_a + (x * vz) * sin_a / K - y * Kv * (1. - cos_a) / K2
        # y1 = vy * cos_a + (y * vz) * sin_a / K + x * Kv * (1. - cos_a) / K2
        # z1 = vz * cos_a - (x * vx + y * vy) * sin_a / K

        opt1 = vz * sin_a / K
        opt2 = Kv * (1 - cos_a) / K2
 
        x1 = vx * cos_a + x * opt1 - y * opt2
        y1 = vy * cos_a + y * opt1 + x * opt2
        z1 = vz * cos_a - (x * vx + y * vy) * sin_a / K

        return np.array([0., x1, y1, z1]) + p0


# Walk-on-plane-z
# Current position, Foot of Perpendicular --> Plane 
@njit('float64[::1](float64[::1], float64[::1])')
def wop_z(p, q):
    x = q[1]
    y = q[2]
    z = q[3]

    u = np.random.rand()
    d = np.abs(p[3] - q[3])
    phi = 2. * np.pi * np.random.rand()
    s = d * np.sqrt(1 - u*u) / u
       
    x += s * np.cos(phi)
    y += s * np.sin(phi)

    return np.array([0., x, y, z])
