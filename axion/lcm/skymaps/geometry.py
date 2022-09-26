from turtle import color
import numpy as np
from numba import jit

@jit("float64[:,:](float64[:],float64[:],float64,int64)", nopython=True)
def oriented_circle_vertices(circle_center, orientation, r, N):
    """Evaluate Cartesian coordinates of the projection of a circle with radius r with center on the unit sphere at position (alpha, beta) and orientation (theta0, phi0) on the unit sphere.
    Parameters
    ----------
    circle_center : 2-tuple of floats
        Spherical polar angles (theta = alpha, phi = beta) specifying the location of the center of the circle.
        The center of the circle lies ON the unit sphere, hence why only the direction is required as input.
    orientation : 2-tuple of flotas
        Spherical polar angles (theta = theta0, phi = phi0) specifying the orientation of the circle. For example,
        (theta0 = 0., phi0 = 0.) corresponds to a circle whose normal vector points along the z-axis. The case
        (theta0 = alpha, phi0 = beta) corresponds to 'perfectly oriented' circles. That is, circles which lie in the
        tangent plane of the unit sphere at position (alpha, beta).
    r : float
        Radius of circle. If you want to apply this function for projections onto a sphere of radius R0 then take divide
        the true radius of the circle r0 by R0 so that r = r0 / R0. This is valid because this function only returns the
        angular coords of the projection.
    N : int
        Number of vertices"""
    alpha = circle_center[0]
    beta = circle_center[1]
    theta0 = orientation[0]
    phi0 = orientation[1]

    t = np.linspace(0,2*np.pi*(N-1)/N,N)

    # I obtained these formulas analytically in Mathematica. The relevant notebook is "randomly-oriented-circular-loops.nb"

    x = ((np.cos(beta))*((np.sin(alpha)) + \
        (-((r)*((np.sin(phi0))*(np.sin(t))))))) + \
        (-((r)*((np.cos(t))*(((np.cos(alpha))*(((np.sin(beta))*(np.sin(phi0)))\
         + (-((np.cos(beta))*((np.cos(phi0))*(np.cos(theta0))))))) + \
        ((np.cos(phi0))*((np.sin(alpha))*(np.sin(theta0)))))))) + \
        (-((r)*((np.cos(phi0))*((np.cos(theta0))*((np.sin(beta))*(np.sin(t))))\
        )))

    y = (((np.sin(alpha)) + \
        ((r)*((np.cos(alpha))*((np.cos(phi0))*(np.cos(t))))))*(np.sin(beta))) \
        + ((r)*((np.cos(beta))*(((np.cos(phi0))*(np.sin(t))) + \
        ((np.cos(alpha))*((np.cos(t))*((np.cos(theta0))*(np.sin(phi0)))))))) \
        + (-((r)*((np.sin(phi0))*(((np.cos(t))*((np.sin(alpha))*(np.sin(\
        theta0)))) + ((np.cos(theta0))*((np.sin(beta))*(np.sin(t))))))))


    z = (np.cos(alpha)) + \
        ((r)*((np.sin(beta))*((np.sin(t))*(np.sin(theta0))))) + \
        (-((r)*((np.cos(t))*((np.cos(theta0))*(np.sin(alpha)))))) + \
        (-((r)*((np.cos(alpha))*((np.cos(beta))*((np.cos(t))*(np.sin(theta0)))\
        ))))

    vertices = np.vstack((x,y,z)).transpose()
    return vertices

from numba import guvectorize
@guvectorize(["void(float64[:], float64[:], float64[:])"], "(n),(m)->(m)")
def _cartesian_to_long_lat(cart_coord, dummy, out):
    x, y, z = cart_coord
    out[0] = -np.arctan2(y,x)
    out[1] = np.arctan2(z, np.sqrt(x**2 + y**2))

def cartesian_to_long_lat(cart_coord):
    cart_coord = np.asarray(cart_coord)
    if len(cart_coord.shape) == 1:
        cart_coord = np.array([cart_coord])
    NUM_POINTS = cart_coord.shape[0]
    out = np.zeros((NUM_POINTS,2), dtype=float)
    _cartesian_to_long_lat(cart_coord, out, out)
    return out

def spherical_coords_to_mollweide(sph_coord, rtol=1e-6):
    sph_coord = np.asarray(sph_coord)
    if len(sph_coord.shape) == 1:
        sph_coord = np.array([sph_coord])

    NUM_POINTS = sph_coord.shape[0]
    x = np.zeros(NUM_POINTS)
    y = np.zeros(NUM_POINTS)
    

    from scipy.optimize import newton
    def theta(lat, rtol=1e-8): # "auxiliary angle"
        if np.abs(lat - np.pi/2) < 1e-8:
            _theta = np.pi/2
        elif np.abs(lat + np.pi/2) < 1e-8:
            _theta = -np.pi/2
        elif np.abs(lat) < 1e-8:
            _theta = 0
        else:
            try:
                _theta = newton(lambda th : 2*th + np.sin(2*th) - np.pi*np.sin(lat), np.pi/2, rtol=rtol, maxiter=10000)
            except:
                print(lat)
        return _theta
    
    for i in range(NUM_POINTS):
        lon, lat = sph_coord[i]
        lon0=0 # central meridian
        x[i] = 2/np.pi*(lon - lon0)*np.cos(theta(lat, rtol=rtol))
        y[i] = np.sin(theta(lat, rtol=rtol))

    return np.vstack([x, y]).T

color