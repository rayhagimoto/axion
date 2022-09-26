from .classes import Simulator, Ensemble, Skymap

def generate_skymap(
    zeta0=1.,
    xi0=1.,
    anomaly=1.,
    zf=None,
    ma=None,
    log10ma=None,
    nside=2048,
    Nsteps=28,
    Nvertices=51,
    degrees=False
):
    """Simulate an all-sky map of birefringence in HEALPix format.
    Parameters
    ----------
    zeta0 : float
        𝜁₀
    xi0 : float
        ξ₀
    anomaly : float
        Anomaly coefficient 𝒜, parameterises the amount of birefringence due to each loop-crossing 
        (Δα = 𝒜/137).
    zf : float
        Redshift at which to terminate the simulation (i.e., when the string network collapses.) If 
        more than one of ma, log10ma, or zf is specified then the program throws an error.
    ma : float
        Axion mass in eV. Used to determine zf, if zf is not given. If more than one of ma, log10ma, 
        or zf is specified then the program throws an error.
    log10ma : float
        Base-10 log of axion mass in eV. (This is useful if you want to scan parameter space linearly 
        in log10ma.) Used to determine zf by the relation 3*H(zf) = ma, if zf is not given. If more than 
        one of ma, log10ma, or zf is specified then the program throws an error.
    nside : int
        HEALpix resolution parameter
    Nsteps : int
        Number of time steps between zcmb = 1100 and zf.
    Nvertices : int
        Number of vertices to sample when drawing ellipses (loops that are not normal to the tangent 
        plane of the sphere.)
    degrees : bool
        Boolean that decided whether to return the skymap in units of degrees or not.
    """
    simulator = Simulator(
        zeta0=zeta0,
        anomaly=anomaly,
        xi0=xi0,
        ma=ma,
        zf=zf,
        log10ma=log10ma,
        nside=nside,
        Nsteps=Nsteps,
        Nvertices=Nvertices,
        degrees=degrees
        )
    return simulator.generate_skymap()

def generate_ensemble_cl(
    zeta0,
    xi0,
    anomaly,
    zf=None,
    ma=None,
    log10ma=None,
    nside=2048,
    Nsteps=28,
    Nvertices=51,
    num_sims=10,
    lmax=300,
    degrees=False,
    progressbar=True
):
    """Simulate a suite of all-sky birefringence maps then compute and store their power
    spectra in ndarray.

    Parameters
    ----------
    zeta0 : float
        𝜁₀
    xi0 : float
        ξ₀
    anomaly : float
        Anomaly coefficient 𝒜, parameterises the amount of birefringence due to each loop-crossing 
        (Δα = 𝒜/137).
    zf : float
        Redshift at which to terminate the simulation (i.e., when the string network collapses.)
        If more than one of ma, log10ma, or zf is specified then the program throws an error.
    ma : float
        Axion mass in eV. Used to determine zf, if zf is not given. If more than one of ma,
        log10ma, or zf is specified then the program throws an error.
    log10ma : float
        Base-10 log of axion mass in eV. (This is useful if you want to scan parameter space 
        linearly in log10ma.) Used to determine zf by the relation 3*H(zf) = ma, if zf is not 
        given. If more than one of ma, log10ma, or zf is specified then the program throws an 
        error.
    nside : int
        HEALpix resolution parameter
    Nsteps : int
        Number of time steps between zcmb = 1100 and zf.
    Nvertices : int
        Number of vertices to sample when drawing ellipses (loops that are not normal to the 
        tangent plane of the sphere.)
    num_sims : int
        Number of simulations to run for the ensemble
    lmax : int
        Maximum multipole up to which to calculate the power spectrum of the birefringence
        map.
    degrees : bool
        Boolean that decided whether to return the skymap in units of degrees or not.
    """
    simulator = Simulator(
        zeta0=zeta0,
        anomaly=anomaly,
        xi0=xi0,
        ma=ma,
        zf=zf,
        log10ma=log10ma,
        nside=nside,
        Nsteps=Nsteps,
        Nvertices=Nvertices,
        degrees=degrees
        )
    return simulator.generate_ensemble_cl(num_sims, lmax, progressbar=progressbar)

