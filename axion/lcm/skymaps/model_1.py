# -*- coding: utf-8 -*-
"""
This program simulates axion-induced birefringence using the loop-crossing model. See arXiv:2103.10962.
(NOTE: All equation numbers refer to those in arXiv:2103.10962)

In the incarnation of the loop-crossing model considered here, loops at each time-step have identical size, fixed by 𝜁₀. 
They are randomly oriented (orientation vector sampled uniformly at random from 2-sphere.) Number of loops is controlled
by ξ₀, the average energy density of string loops in Hubble volume. String network collapses at redshift zf.
"""
import numpy as np
from scipy.integrate import quad
from healpy import pix2ang, query_polygon

from ...cosmology import H, zcmb, s, alphaEM
from .geometry import oriented_circle_vertices

def loops_in_vol(zeta0, xi0, z_old, z_new):  # compute number of loops in comoving volume between z_n and z_n+1

    n_co = lambda z : xi0 / 2 / np.pi / zeta0 / (1 + z)**3 * H(z)**3
    integrand = lambda z : n_co(z) * 4 * np.pi * s(z)**2 / H(z)
    N_co, err = quad(integrand, z_new, z_old)

    return N_co

def logarithmic_z_steps(n, zf, Nsteps): 
    return (1 + zcmb) * ((1 + zf) / (1 + zcmb)) ** (n / Nsteps) - 1

def get_current_z(n : int, zf, Nsteps : int, delta):

    if n != Nsteps:
        oneplusz = (logarithmic_z_steps(n, zf, Nsteps) + 1) * np.exp(delta) # apply random shift to time steps
        z = oneplusz - 1
    else:
        z = logarithmic_z_steps(n, zf, Nsteps)
    
    return z

def calculate_radius(zeta0, z): return zeta0 * (1 + z) / H(z) / s(z)

def calculate_average_z(step, zf, Nsteps):
    if step == 1:
        z_old = zcmb
    else:
        z_old = logarithmic_z_steps(step-1, zf, Nsteps)
    z_new = logarithmic_z_steps(step, zf, Nsteps)
    
    norm, err = quad(s, z_new, z_old)
    if norm > 0.:
        return quad(lambda z : z * s(z), z_new, z_old)[0] / norm
    else:
        return None

def accumulate_birefringence(sky, step, zeta0, xi0, anomaly, zf, Nsteps, Nvertices, delta, z_old, nside):
    """Simulates birefringence for one time step"""

    npix = 12*nside**2

    z = get_current_z(step, zf, Nsteps, delta)

    # z_for_r is redshift at which radius is evaluated. Chosen to be the average value of z between two redshift slices
    # weighted by the comoving distance to the redshift shell at that time.
    z_for_r = calculate_average_z(step, zf, Nsteps)
    if z_for_r != None:
        r = zeta0 * (1 + z_for_r) / H(z_for_r) / s(z_for_r) # comoving radius of loop in units of s(z).

        p = loops_in_vol(zeta0, xi0, z_old, z) / npix
        temp = np.random.rand(npix) # assign random double between 0 and 1 to each pixel
        ipix = np.where(temp < p)[0] # replace this with analytic ordered statistics
        pixel_center = np.transpose( pix2ang(nside, ipix) ) # convert pixel_centers to spherical coordinate angles

        # iterate over the pixels chosen to center string loop
        for i in range(len(ipix)):

            winding_number = 2*(np.random.randint(2) - 0.5) # assign winding number
            delta_alpha = winding_number * anomaly * alphaEM
            success = False
            while success == False:
                orientation = np.asarray([np.arccos(np.random.rand()), 2*np.pi*np.random.rand()]) # draw direction (theta0, phi0) from bivariate uniform distribution on [costheta_lower, 1.] x [0., 2*pi]
                try:
                    temp = query_polygon(nside, oriented_circle_vertices(pixel_center[i], orientation, r, Nvertices))
                    success = True
                except:
                    print('something went wrong with query_polygon')
                    pass

            sky[temp] += delta_alpha


def simulate_sky_healpix(zeta0=1.0, xi0=1.0, anomaly=1.0, zf=0.0, 
            nside=2048, Nsteps=28, Nvertices=51):
    """Run simulation
    Usage
    -----
    model_1.run_sim(zeta0=1.0, xi=1.0, zf=0.0, anomaly=1.0, Nsteps=28, Nvertices=51)

    Parameters
    ----------
    zeta0 : float
        𝜁₀
    xi0 : float
        ξ₀
    anomaly : float
        Anomaly coefficient 𝒜, parameterises the amount of birefringence due to each loop-crossing (Δα = 𝒜/137).
    zf : float
        Redshift at which to terminate the simulation (i.e., when the string network collapses.)
    nside : int
        HEALpix resolution parameter
    Nsteps : int
        Number of time steps between zcmb = 1100 and zf.
    Nvertices : int
        Number of vertices to sample when drawing ellipses (loops that are not normal to the tangent plane of the sphere.)
    
    Return
    ------
    Birefringence map in HEALPix format (a 1D array with npix = 12*nside**2 pixels)"""
    
    npix = 12*nside**2
    sky = np.zeros(npix)
    z_old = zcmb
    delta = (np.random.rand() - 0.5) * (np.log((1 + zcmb) / (1 + zf)) / Nsteps) # parameter that randomly shifts the time steps

    for step in np.arange(Nsteps) + 1:
        
        z = get_current_z(step, zf, Nsteps, delta)
        
        accumulate_birefringence(sky, step, zeta0, xi0, anomaly, zf, Nsteps, Nvertices, delta, z_old, nside)
        
        z_old = z
        
    return sky
