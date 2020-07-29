import numpy as np
from numba import njit

#list: Disk, Annulus, Sphere, Hemisphere.

# Sampling disk parallel to z-plane
# Center of disk, Radius of disk --> Disk
@njit('float64[::1](float64[::1], float64)')
def disk(p, a):
    u = np.random.rand()
    phi = np.random.rand() * 2. * np.pi 
    s = a * np.sqrt(u)
    x = np.cos(phi) * s
    y = np.sin(phi) * s

    return np.array([0., x, y, 0.]) + p


# Sampling annulus parallel to z-plane
# Center of annulus, Inside radius, Outside radius --> Annulus
@njit('float64[::1](float64[::1], float64, float64)')
def annulus(p, a1, a2):
    u = np.random.rand()
    phi = np.random.rand() * 2. * np.pi 
    s = np.sqrt(a1*a1*(1-u) + a2*a2*u)
    x = np.cos(phi) * s
    y = np.sin(phi) * s

    return np.array([s, x, y, 0.]) + p


# Sampling 
# Center of Sphere, Radius of Sphere --> Spherical Surface
@njit('float64[::1](float64[::1], float64)')
def sphere(p, a):
    x = p[1]
    y = p[2]
    z = p[3]
    u = np.random.rand()
    phi = np.random.rand() * 2. * np.pi 
    cos_th = 2.*u*u - 1.
    sin_th = np.sqrt(1 - cos_th * cos_th)
    x += np.cos(phi) * sin_th * a
    y += np.sin(phi) * sin_th * a
    z += cos_th * a

    return np.array([0., x, y, z])


# Sampling hemisphere cut by plane normal to z
# Center of hemisphere, Radius of hemisphere --> Hemisphere
@njit('float64[::1](float64[::1], float64)')
def hemisphere(p, a):
    u = np.random.rand()
    phi = 2. * np.pi * np.random.rand()
    cos_th = np.sqrt(u)
    sin_th = np.sqrt(1. - u)

    x = np.cos(phi) * sin_th * a
    y = np.sin(phi) * sin_th * a
    z = cos_th * a

    return np.array([0., x, y, z]) + p

    