# -*- coding: utf-8 -*-
"""
This file contains analytical methods for computing the power spectrum from axion-induced birefringence.

(NOTE: All equation, page, and section numbers refer to arXiv:2103.10962)

We assume the "loop-crossing" framework described in arXiv:2103.1096. Within this framework
there are several models:

    Model 1: Loops at each time-step have identical size, fixed by 𝜁₀. Randomly oriented
             (orientation vector sampled uniformly at random from 2-sphere.)

    Model 2: Loop distribution is parameterised by 𝜁min, 𝜁max, fsub, and 𝒜²ξ₀ which
             control the smallest radius, largest radius, fraction of sub-Hubble loops,
             and power spectrum amplitude respectively. (See §4.2 for more detailed
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
λ = 1/(2*np.sqrt(3)) # parameter in the range 0 < λ < 1 which accounts for the loop's reduced angular extent (see pg. 18)

def ma(z_collapse): 
    """Axion mass corresponding to a string network who collapses at redshift z_collapse."""
    return 3*hbarH0*H(z_collapse)

theta_eff = lambda z, 𝜁 : 2*np.arctan2(λ*𝜁*(1+z), H(z)*s(z)) # eqn. 35

def z_star_fast(𝜁, θ):
    "Approximates z_star by assuming a matter dominated universe. See eqn. (37)."
    cotHalfTheta = 1/np.tan(0.5*θ)
    return λ*𝜁*cotHalfTheta + 0.25*λ**2*𝜁**2*cotHalfTheta**2

def z_star(𝜁, θ):

    def _zstar(theta): 
        f = lambda z : np.array(H(z) * s(z) / (1 + z) - λ * 𝜁 / np.tan(0.5*theta), dtype=float)
        return root(f, 1, tol=1e-6).x[0]
    
    res = np.frompyfunc(
        _zstar, nin=1, nout=1
    )

    return res(θ).astype(float)


def z_star_tilde(zstar):
    """Approximation of Min[z_star(𝜁, θ), zcmb]
    z_star is redshift at which opening angle θ for two CMB photons equals the effective opening angle θeff
    of a loop through which they pass (the loop can take arbitrary orientations so θeff < θd, its angular
    diameter.) Figure 4. is useful for understanding the meaning of this.

    𝜁 : float or 2d ndarray
    θ : float or 2d ndarray
    """
    return zcmb*zstar / (zcmb + zstar) # see eqn. (42); theta = opening angle

# ---------------------------------------------------------------------------------- #
#                    MODEL 1 CORRELATION FUNCTION AND POWER SPECTRUM                 #
# ---------------------------------------------------------------------------------- #

def model_1_corr(params,  θ):
    """Approximate two point correlation function for opening angle theta (only valid for model 1)
    PARAMETERS
    ----------
    params : 2-tuple of (𝜁0, amplitude)
        𝜁0 : float
        amplitude : float
    θ : 1d array of angles at which to evaluate the two-point correlation function.
    """

    𝜁0, amplitude = params

    # theta_transition occurs when theta_effective equals z_transition(zeta) (cf. eq. (39))
    zstar = z_star(𝜁0, θ)
    zstartilde = z_star_tilde(zstar)

    # θc = theta_eff(0., 𝜁0)
    θt = 2*np.arctan(2*λ) # eqn. (39)

    corr = np.where(
        θ < θt, 
        0.25*𝜁0*(np.log(1+zstartilde) - 𝜁0/3.), # see eq. (46)
        np.log(1+zstartilde)**3./𝜁0/3. # see footnote 8 on page 21.
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
            𝜁₀ = average radius of string loops in units of inverse Hubble.
        amplitude : float
            𝒜²ξ₀ = amplitude of power spectrum. 𝒜 is anomaly coefficient.
            ξ₀ is average energy density of string loops in Hubble volume.
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
        Value of the Model1(𝜁₀, 𝒜²ξ₀) birefringence power spectrum at multipole moments provided by "ells" parameter in units of rad^2
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

def model_3_corr(params, θ, p=3./2.):
    """Analytical approximation of integral of Q(𝜁,z,0) from zcollapse to
    a smoothed z_star_tilde --> (z_collapse**n + z_star_tilde**n)**(1/n) (n = 3/2).
    (cf. eq. 43.)"""

    𝜁0, amplitude, z_collapse = params

    corr = np.zeros_like(θ)
    zstar = z_star(𝜁0, θ)
    zstartilde = (z_star_tilde(zstar)**p + z_collapse**p)**(1/p) #smoothed zstartilde

    θt = 2*np.arctan(2*λ) # eqn. (39)
    θc = theta_eff(z_collapse, 𝜁0)

    cond1 = θc < θt
    cond2 = np.logical_and(θ < θt, θt < θc)
    cond3 = np.logical_and(θ >= θt, θc >= θt)

    # compute correlation function in the three regimes enumerated above
    corr[cond1] = 0.25*𝜁0*(np.log(1 + zstartilde[cond1]) - np.log(1 + z_collapse)) # see eqn. 54
    corr[cond2] = 0.25*𝜁0*np.log(1 + zstartilde[cond2]) - 𝜁0**2/12. - np.log(1 + z_collapse)**3/3/𝜁0 # see footnote 11 on page 28
    corr[cond3] = (np.log(1 + zstartilde[cond3])**3 - np.log(1 + z_collapse)**3)/3/𝜁0 

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
            𝜁₀ = average radius of string loops in units of inverse Hubble.
        amplitude : float
            𝒜²ξ₀ = amplitude of power spectrum. 𝒜 is anomaly coefficient.
            ξ₀ is average energy density of string loops in Hubble volume.
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
        Value of the Model1(𝜁₀, 𝒜²ξ₀) birefringence power spectrum at multipole moments provided by "ells" parameter in units of rad^2
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


def W(𝜁, θ):
    """Analytical approximation of integral of Q(𝜁,z,0) from 0 to z_star_tilde
    (cf. eq. 43 and replace χ with the model 2 loop distribution function.)"""
    _W = np.zeros_like(θ)
    zt = np.exp(𝜁/2) - 1
    zstar = z_star(𝜁, θ)
    zstartilde = z_star_tilde(zstar)
    cond1 = zstartilde < zt
    cond2 = zstartilde >= zt


    _W[cond1] = np.log(1 + zstar[cond1])**3 / 3 / 𝜁[cond1]
    _W[cond2] = (0.25 * 𝜁[cond2] * np.log(1 + zstartilde[cond2]) - 𝜁[cond2]**2 / 12.)

    return _W


def model_2_corr(
    model_params,
    num_params=[1e-10, 3500j, 10j]
):
    """Model 2 correlation function
    model_params : 4-tuple
        𝜁min, 𝜁max, fsub, amplitude
    num_params : 3-tuple
        θmin, θlen, 𝜁len"""
    # unpack params
    θmax = pi
    θmin, θlen, 𝜁len = num_params
    𝜁min, 𝜁max, fsub, amplitude = model_params
    𝜁, θ = np.mgrid[𝜁min:𝜁max:𝜁len, θmin:θmax:θlen] # initialise 𝜁 and θ as 2D grids

    # first term
    corr = model_1_corr([𝜁max, amplitude*(1-fsub)], θ[0,:])

    # second term
    d𝜁 = 𝜁[1,0] - 𝜁[0,0]
    integral = integrate2d(W(𝜁, θ), d𝜁)

    # combine both terms to get the correlation function
    corr += amplitude*fsub*alphaEM**2*integral/(𝜁max - 𝜁min)

    return θ[0,:], corr

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
        Model parameters in the order (𝜁min, 𝜁max, fsub, amplitude)
        -----------------------------------------------------------------
        𝜁min : float
            Smallest radius of string loops in units of inverse Hubble.
        𝜁max : float
            Largest radius of string loops in units of inverse Hubble.
        fsub : float (between 0. and 1.)
            Fraction of sub-Hubble loops
        amplitude : float
            𝒜²ξ₀ = amplitude of power spectrum. 𝒜 is anomaly coefficient.
            ξ₀ is average energy density of string loops in Hubble volume.
        ------------------------------------------------------------------
    theta_len : int
        Length for array of thetas between theta_min and Pi
    theta_min : float
        Minimum opening angle (radians) to use. Can't be zero because we must compute 1/tan(0.5*theta),
        which would be infinite for theta = 0.

    RETURNS
    -------
    cl : ndarray
        Value of the Model2(𝜁min, 𝜁max, fsub, 𝒜²ξ₀) birefringence power spectrum at multipole
        moments ells in units of rad^2
    """

    𝜁min, 𝜁max, fsub, amplitude = params

    # initialise arrays
    theta = np.linspace(theta_min, pi, theta_len)
    dtheta= theta[1] - theta[0]
    cl = np.zeros_like(ells, dtype=float)

    # calculate two-point correlation function and smooth it (to get rid of unphysical kinks or discontinuities)
    theta, corr = model_2_corr(model_params=[𝜁min, 𝜁max, fsub, amplitude], num_params=[1e-10, theta_len*1j, zeta_len*1j])

    costheta = cos(theta)
    integrand = sin(theta)*corr

    for i,l in enumerate(ells):
        Pl = Legendre(l, costheta)
        cl[i] = 2*pi*integrate(integrand*Pl,dtheta)
    return cl

