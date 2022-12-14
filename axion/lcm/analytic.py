# -*- coding: utf-8 -*-
"""
This file contains analytical methods for computing the power spectrum from axion-induced birefringence.

(NOTE: All equation, page, and section numbers refer to arXiv:2103.10962)

We assume the "loop-crossing" framework described in arXiv:2103.1096. Within this framework
there are several models:

    Model 1: Loops at each time-step have identical size, fixed by πβ. Randomly oriented
             (orientation vector sampled uniformly at random from 2-sphere.)

    Model 2: Loop distribution is parameterised by πmin, πmax, fsub, and πΒ²ΞΎβ which
             control the smallest radius, largest radius, fraction of sub-Hubble loops,
             and power spectrum amplitude respectively. (See Β§4.2 for more detailed
             explanation.)

    Model 3: Model 1 + network collapse when axion mass = 3 times Hubble

This file contains various methods to calculate the birefringence power spectrum (in
 units of rad^2) for a given list of multipole moments.

The general strategy for this program is to evaluate the two-point correlation function
 via eqn. (46) then 'project' the result on a basis of Legendre polynomials to obtain 
 the power spectrum (see eqns. (17-18).)

USAGE
-----
"""

import numpy as np
from numpy import pi, sin, cos
from scipy.special import eval_legendre as Legendre
from scipy.optimize import root
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from ..cosmology import H, s, zcmb, alphaEM, hbarH0
from .math import integrate, integrate2d

# LCM-specific constants
Ξ» = 1/(2*np.sqrt(3)) # parameter in the range 0 < Ξ» < 1 which accounts for the loop's reduced angular extent (see pg. 18)

def ma(z_collapse): 
    """Axion mass corresponding to a string network who collapses at redshift z_collapse."""
    return 3*hbarH0*H(z_collapse)

theta_eff = lambda z, π : 2*np.arctan2(Ξ»*π*(1+z), H(z)*s(z)) # eqn. 35

def z_star_fast(π, ΞΈ):
    "Approximates z_star by assuming a matter dominated universe. See eqn. (37)."
    cotHalfTheta = 1/np.tan(0.5*ΞΈ)
    return Ξ»*π*cotHalfTheta + 0.25*Ξ»**2*π**2*cotHalfTheta**2

def z_star(π, ΞΈ):

    def _zstar(theta): 
        f = lambda z : np.array(H(z) * s(z) / (1 + z) - Ξ» * π / np.tan(0.5*theta), dtype=float)
        return root(f, 1, tol=1e-6).x[0]
    
    res = np.frompyfunc(
        _zstar, nin=1, nout=1
    )

    return res(ΞΈ).astype(float)


def z_star_tilde(zstar):
    """Approximation of Min[z_star(π, ΞΈ), zcmb]
    z_star is redshift at which opening angle ΞΈ for two CMB photons equals the effective opening angle ΞΈeff
    of a loop through which they pass (the loop can take arbitrary orientations so ΞΈeff < ΞΈd, its angular
    diameter.) Figure 4. is useful for understanding the meaning of this.

    π : float or 2d ndarray
    ΞΈ : float or 2d ndarray
    """
    return zcmb*zstar / (zcmb + zstar) # see eqn. (42); theta = opening angle

# ---------------------------------------------------------------------------------- #
#                    MODEL 1 CORRELATION FUNCTION AND POWER SPECTRUM                 #
# ---------------------------------------------------------------------------------- #

def model_1_corr(params,  ΞΈ):
    """Approximate two point correlation function for opening angle theta (only valid for model 1)
    PARAMETERS
    ----------
    params : 2-tuple of (π0, amplitude)
        π0 : float
        amplitude : float
    ΞΈ : 1d array of angles at which to evaluate the two-point correlation function.
    """

    π0, amplitude = params

    # theta_transition occurs when theta_effective equals z_transition(zeta) (cf. eq. (39))
    zstar = z_star(π0, ΞΈ)
    zstartilde = z_star_tilde(zstar)

    # ΞΈc = theta_eff(0., π0)
    ΞΈt = 2*np.arctan(2*Ξ») # eqn. (39)

    corr = np.where(
        ΞΈ < ΞΈt, 
        0.25*π0*(np.log(1+zstartilde) - π0/3.), # see eq. (46)
        np.log(1+zstartilde)**3./π0/3. # see footnote 8 on page 21.
    )

    return corr*amplitude*alphaEM**2


def model_1_cl(
    ells,
    params,
    theta_min=1e-4,
    theta_max=pi,
    n=30,
    n_log_space=25,
    interpolation='cubic',
):
    """Compute Model 1 birefringence power spectrum at multipole moments "ells" in units of rad^2.
    USAGE
    -----
    Call signature and default values: model_1_cl(ells, params=[1.0, 1.0], theta_len=[10000,100000])

    PARAMETERS
    ----------
    ells : array_like
        list of multipole moments at which to evaluate power spectrum.
    params : 1d array_like of length 2
        Model parameters in the order (zeta0, amplitude)
        -----------------------------------------------------------------
        zeta0 : float
            πβ = average radius of string loops in units of inverse Hubble.
        amplitude : float
            πΒ²ΞΎβ = amplitude of power spectrum. π is anomaly coefficient.
            ΞΎβ is average energy density of string loops in Hubble volume.
        ------------------------------------------------------------------
    n : int
        Parameter which determines precision of sampling in angle theta on
        [0, Pi]. Number of samples = (l + 1) * n, where l is the multipole moment.
    n_log_space : int
        Parameter which determines number of samples to take on a log-scale
        when evaluating the correlation function. These points are then 
        interpolated when integrating against Legendre polynomials.
    interpolation : str
        Type of interpolation to tell scipy.interpolate.interp1d to use (see
        their documentation for "kind" kwarg.)

    RETURNS
    -------
    cl : ndarray
        Value of the Model1(πβ, πΒ²ΞΎβ) birefringence power spectrum at multipole moments provided by "ells" parameter in units of rad^2
    """
    zeta0, ampl = params

    cl = np.zeros_like(ells, dtype=float)

    logth = np.linspace(np.log10(theta_min), np.log10(theta_max), n_log_space)
    theta = 10**logth
    corr = model_1_corr([zeta0, ampl], theta)
    interpf = interp1d(theta, corr, kind=interpolation, fill_value='extrapolate')

    # Convert correlation function to power spectrum through Legendre polynomial transform
    for i, l in enumerate(ells):
        theta = np.linspace(theta_min, theta_max, (l + 1)*n)
        dtheta = theta[1] - theta[0]

        # calculate two-point correlation function and smooth it (to get rid of unphysical kinks or discontinuities)
        corr = interpf(theta)
        Pl = Legendre(l, cos(theta))
        cl[i] = 2*pi*integrate(sin(theta)*corr*Pl,dtheta)

    return cl


# ---------------------------------------------------------------------------------- #
#                    MODEL 3 CORRELATION FUNCTION AND POWER SPECTRUM                 #
# ---------------------------------------------------------------------------------- #

def model_3_corr(params, ΞΈ, p=3./2.):
    """Analytical approximation of integral of Q(π,z,0) from zcollapse to
    a smoothed z_star_tilde --> (z_collapse**n + z_star_tilde**n)**(1/n) (n = 3/2).
    (cf. eq. 43.)"""

    π0, amplitude, z_collapse = params

    corr = np.zeros_like(ΞΈ)
    zstar = z_star(π0, ΞΈ)
    zstartilde = (z_star_tilde(zstar)**p + z_collapse**p)**(1/p) #smoothed zstartilde

    ΞΈt = 2*np.arctan(2*Ξ») # eqn. (39)
    ΞΈc = theta_eff(z_collapse, π0)

    cond1 = ΞΈc < ΞΈt
    cond2 = np.logical_and(ΞΈ < ΞΈt, ΞΈt < ΞΈc)
    cond3 = np.logical_and(ΞΈ >= ΞΈt, ΞΈc >= ΞΈt)

    # compute correlation function in the three regimes enumerated above
    corr[cond1] = 0.25*π0*(np.log(1 + zstartilde[cond1]) - np.log(1 + z_collapse)) # see eqn. 54
    corr[cond2] = 0.25*π0*np.log(1 + zstartilde[cond2]) - π0**2/12. - np.log(1 + z_collapse)**3/3/π0 # see footnote 11 on page 28
    corr[cond3] = (np.log(1 + zstartilde[cond3])**3 - np.log(1 + z_collapse)**3)/3/π0 

    return corr*amplitude*alphaEM**2

def model_3_cl(
    ells,
    params,
    theta_min=1e-4,
    theta_max=pi,
    n=30,
    n_log_space=25,
    interpolation='cubic',
    p=3./2.
):
    """Compute Model 1 birefringence power spectrum at multipole moments "ells" in units of rad^2.
    USAGE
    -----
    Call signature and default values: model_1_cl(ells, params, theta_min=1e-10, theta_len=[10000,100000])

    PARAMETERS
    ----------
    ells : array_like
        list of multipole moments at which to evaluate power spectrum.
    params : 1d array_like of length 4
        Model parameters in the order (zeta0, amplitude, zc)
        -----------------------------------------------------------------
        zeta0 : float
            πβ = average radius of string loops in units of inverse Hubble.
        amplitude : float
            πΒ²ΞΎβ = amplitude of power spectrum. π is anomaly coefficient.
            ΞΎβ is average energy density of string loops in Hubble volume.
        log10ma or ma or zc : float
            Depending on what input I want, I have to change the code to
            accept either the axion mass in eV, the log10 of the mass in
            eV, or the collapse redshift. In each case I need to physically
            change the code below.
        ------------------------------------------------------------------
    theta_min : float
        Minimum opening angle (radians) to use. Can't be zero because we must compute 1/tan(0.5*theta),
        which would be infinite for theta = 0.
    theta_max : float
        Maximum opening angle (radians) to use. Should <= pi.
    n : int
        Factor that adjusts how many times to sample Legendre polynomial. The number of samples is 
        (l + 1) * n, where l is the order of the Legendre polynomial.
    n_log_space : int
        Number of samples to take on a log scale. This number should be fairly small so that the interpolated
        correlation will be sufficiently smooth. If n_log_space is too large then there will be a kink and a
        discontinuity in the interpolated function, which exacerbates unphysical wiggles in the power spectrum.
    p : float
        An exponent that smooths z_star_tilde like
                     z_star_tilde --> (z_star_tilde ** p + z_collapse ** p) ** (1 / p)

    RETURNS
    -------
    cl : ndarray
        Value of the Model1(πβ, πΒ²ΞΎβ) birefringence power spectrum at multipole moments provided by "ells" parameter in units of rad^2
    """
    zeta0, ampl, log10ma = params
    if log10ma > np.log10(ma(zcmb)): 
        return np.zeros_like(ells, dtype=float)
    
    elif log10ma <= np.log10(ma(0)): 
        return model_1_cl(ells, [zeta0, ampl])
    
    else:
        zc = root(lambda z : np.log10(ma(z)) - log10ma, 0.).x[0]

        cl = np.zeros_like(ells, dtype=float)

        theta = np.logspace(np.log10(theta_min), np.log10(theta_max), n_log_space)
        corr = model_3_corr([zeta0, ampl, zc], theta, p=p)

        interpf = interp1d(theta, corr, kind=interpolation, fill_value='extrapolate')
        # Convert correlation function to power spectrum through Legendre polynomial transform
        for i, l in enumerate(ells):
            theta = np.linspace(theta_min, theta_max, (l+1)*n)
            dtheta = theta[1] - theta[0]

            # calculate two-point correlation function and smooth it (to get rid of unphysical kinks or discontinuities)
            corr = interpf(theta)

            Pl = Legendre(l, cos(theta))
            cl[i] = 2*pi*integrate(sin(theta)*corr*Pl,dtheta)

        return cl






# EXPERIMENTAL!! #
# ------------------------------------------------------------------------------------------------------ #
#                             MODEL 2 CORRELATION FUNCTION AND POWER SPECTRUM                            #
# ------------------------------------------------------------------------------------------------------ #


def W(π, ΞΈ):
    """Analytical approximation of integral of Q(π,z,0) from 0 to z_star_tilde
    (cf. eq. 43 and replace Ο with the model 2 loop distribution function.)"""
    _W = np.zeros_like(ΞΈ)
    zt = np.exp(π/2) - 1
    zstar = z_star(π, ΞΈ)
    zstartilde = z_star_tilde(zstar)
    cond1 = zstartilde < zt
    cond2 = zstartilde >= zt


    _W[cond1] = np.log(1 + zstar[cond1])**3 / 3 / π[cond1]
    _W[cond2] = (0.25 * π[cond2] * np.log(1 + zstartilde[cond2]) - π[cond2]**2 / 12.)

    return _W


def model_2_corr(
    model_params,
    num_params=[1e-10, 3500j, 10j]
):
    """Model 2 correlation function
    model_params : 4-tuple
        πmin, πmax, fsub, amplitude
    num_params : 3-tuple
        ΞΈmin, ΞΈlen, πlen"""
    # unpack params
    ΞΈmax = pi
    ΞΈmin, ΞΈlen, πlen = num_params
    πmin, πmax, fsub, amplitude = model_params
    π, ΞΈ = np.mgrid[πmin:πmax:πlen, ΞΈmin:ΞΈmax:ΞΈlen] # initialise π and ΞΈ as 2D grids

    # first term
    corr = model_1_corr([πmax, amplitude*(1-fsub)], ΞΈ[0,:])

    # second term
    dπ = π[1,0] - π[0,0]
    integral = integrate2d(W(π, ΞΈ), dπ)

    # combine both terms to get the correlation function
    corr += amplitude*fsub*alphaEM**2*integral/(πmax - πmin)

    return ΞΈ[0,:], corr

def model_2_cl(
    ells,
    params=[0.1, 1, 0.5, 1],
    theta_min=1e-4,
    theta_len=3500,
    zeta_len=5000
):
    """Compute Model 2 birefringence power spectrum at multipole moments "ells" in units of rad^2.
    USAGE
    -----
    Call signature and default values: model_1_cl(ells, [zeta0, amplitude], dtheta=1e-4, theta_min=1e-10)

    PARAMETERS
    ----------
    ells : array_like
        list of multipole moments at which to evaluate power spectrum.
    params : 1d array_like of length 4
        Model parameters in the order (πmin, πmax, fsub, amplitude)
        -----------------------------------------------------------------
        πmin : float
            Smallest radius of string loops in units of inverse Hubble.
        πmax : float
            Largest radius of string loops in units of inverse Hubble.
        fsub : float (between 0. and 1.)
            Fraction of sub-Hubble loops
        amplitude : float
            πΒ²ΞΎβ = amplitude of power spectrum. π is anomaly coefficient.
            ΞΎβ is average energy density of string loops in Hubble volume.
        ------------------------------------------------------------------
    theta_len : int
        Length for array of thetas between theta_min and Pi
    theta_min : float
        Minimum opening angle (radians) to use. Can't be zero because we must compute 1/tan(0.5*theta),
        which would be infinite for theta = 0.

    RETURNS
    -------
    cl : ndarray
        Value of the Model2(πmin, πmax, fsub, πΒ²ΞΎβ) birefringence power spectrum at multipole
        moments ells in units of rad^2
    """

    πmin, πmax, fsub, amplitude = params

    # initialise arrays
    theta = np.linspace(theta_min, pi, theta_len)
    dtheta= theta[1] - theta[0]
    cl = np.zeros_like(ells, dtype=float)

    # calculate two-point correlation function and smooth it (to get rid of unphysical kinks or discontinuities)
    theta, corr = model_2_corr(model_params=[πmin, πmax, fsub, amplitude], num_params=[1e-10, theta_len*1j, zeta_len*1j])

    costheta = cos(theta)
    integrand = sin(theta)*corr

    for i,l in enumerate(ells):
        Pl = Legendre(l, costheta)
        cl[i] = 2*pi*integrate(integrand*Pl,dtheta)
    return cl

