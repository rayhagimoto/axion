"Ensemble, Skymap and Simulator classes are defined here."

from .model_1 import simulate_sky_healpix, logarithmic_z_steps, calculate_radius
from ...cosmology import hbarH0, H
from ...utils import _check_resolution_sufficiency, _check_only_one

import numpy as np
from healpy import anafast
from scipy.optimize import root

import numpy as np
from .plotutils import plot_power
from fastprogress.fastprogress import progress_bar

# ------------------------------------ MISC. ------------------------------------ #

def _display_lcm_params(lcm_params):
    res = ''
    i = 0
    pairs = list(lcm_params.items())
    if len(pairs) == 1:
        key, value = pairs[0]
        res += f'{key} : {value}'
        return res
    else:
        for i, pair in enumerate(pairs):
            key, value = pair
            if i < len(pairs) - 1:
                res += f'{key} : {value}, '
            else: 
                res += f'{key} : {value}'
        return res


# -------------------------------- CORE CLASSES --------------------------------- #

class Ensemble(np.ndarray):
    """An ensemble of birefringence power spectra. Ensemble refers to
    a set of samples (also referred to as realisations) of a random variable. 
    In this case the ensemble is a set of birefringence power spectra obtained 
    from independent simulations of birefringence skymaps using the loop-crossing
    model.
    
    The ensemble is a numpy ndarray of shape (num_sims, lmax + 1) where num_sims
    refers to the number of realisations, and lmax is the highest multipole up to
    which the power spectrum is computed. The Ensemble class is an extension of
    the numpy ndarray class and adds the attributes "num_sims" and "lcm_params"
    which are useful for bookkeeping purposes. This class also has the methods
    plot() and to_hdf() which make plotting and saving a little easier.
    """
    def __new__(cls, arr, lcm_params):
        if len(arr.shape) < 2:
            arr = np.asarray([arr])
        obj = np.asarray(arr).view(cls)
        obj.lcm_params = lcm_params
        obj.num_sims = arr.shape[0]
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.lcm_params = getattr(obj, "lcm_params", None)
    
    def __repr__(self):
        return "ensemble(%s, {%s}, num_sims=%s)" % (np.asarray(self), _display_lcm_params(self.lcm_params), self.num_sims)

    def plot(self, **kwargs):
        num_sims, lmax_plus_1 = np.array(self).shape
        ell = np.arange(lmax_plus_1)
        plot_power(self.lcm_params, ell, np.array(self), **kwargs)

    def to_hdf(self, output_directory='output', fn=None):
        import h5py as h5
        if fn is None:
            fn = f"zeta0={self.zeta0:.2f}-xi0={self.xi0:.2f}-anomaly={self.anomaly:.2f}-zf={int(self.zf)}"
        # write simulations to disk so that we can open them again later.
        h5f = h5.File(output_directory + '/' + fn + '.h5', 'w')
        h5f.create_dataset('pwr', data=np.asarray(self))
        h5f.close()
    
    def mean(self, **kwargs):
        return np.asarray(self).mean(**kwargs)




class Skymap(np.ndarray):

    def __new__(cls, arr, lcm_params : dict, units : str):
        """
        Params
        ------
        lcm_params : dict
            Dict containing zeta0, xi0, anomaly, and ma

        """
        obj = np.asarray(arr).view(cls)
        obj.lcm_params = lcm_params
        obj.units = units
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.lcm_params = getattr(obj, "lcm_params", None)
        self.units = getattr(obj, "units", None)
    
    def __repr__(self):
        return "skymap(%s, {%s}, units=%s)" % (np.asarray(self), _display_lcm_params(self.lcm_params), self.units)

    def plot(self, **kwargs):
        from healpy import mollview
        mollview(np.asarray(self), **kwargs)

    def plot_power(self, **kwargs):
        cl = anafast(np.asarray(self))
        Ensemble(cl, self.lcm_params).plot(**kwargs)



class Simulator():
    """Class containing methods for simulating loop-crossing model

    Most useful methods are generate_skymap() and generate_ensemble_cl()

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
        map.
    degrees : bool
        Boolean that decided whether to return the skymap in units of degrees or not.
    """

    def __init__(
            self,
            zeta0,
            anomaly,
            xi0,
            ma=None,
            zf=None,
            log10ma=None,
            nside=2048,
            Nsteps=28,
            Nvertices=51,
            degrees=False
            ):

        self.zeta0 = zeta0
        self.anomaly = anomaly
        self.xi0 = xi0
        
        _check_only_one(ma=ma, zf=zf, log10ma=log10ma)

        if ma is not None:
            self.ma = ma
        if log10ma is not None:
            self.ma = 10**log10ma
        if zf is not None:
            self.zf = zf

        self.nside=nside
        self.Nsteps=Nsteps
        self.Nvertices=Nvertices
        self.degrees=degrees
        
        min_radius = calculate_radius(self.zeta0, logarithmic_z_steps(1, self.zf, self.Nsteps))
        _check_resolution_sufficiency(self.nside, min_radius)

    @property
    def ma(self): return self._ma
    
    @ma.setter
    def ma(self, ma):
        self._zf = root(lambda z : 3*hbarH0*H(z) - ma, 0).x[0]
        self._ma = ma
    
    @property
    def zf(self): 
        return self._zf
    
    @zf.setter
    def zf(self, zf):
        self._ma = 3*hbarH0*H(zf)
        self._zf = zf
    
    @property
    def lcm_params(self):
        return dict(zeta0=self.zeta0, anomaly=self.anomaly, xi0=self.xi0, ma=self.ma)

    def generate_skymap(self):
        sky = simulate_sky_healpix(zeta0=self.zeta0,
                                    xi0=self.xi0,
                                    anomaly=self.anomaly,
                                    zf=self.zf,
                                    Nsteps=self.Nsteps,
                                    nside=self.nside,
                                    Nvertices=self.Nvertices
                                   )
        units = 'rad'
        if self.degrees:
            sky *= 180/np.pi
            units = 'deg'
        sky = Skymap(sky, self.lcm_params, units)
        self.sky = sky
        return sky

    def generate_ensemble_cl(self, num_sims=10, lmax=300, progressbar=True):

        simulations_pwr = np.zeros((num_sims, lmax + 1))

        iterable = range(num_sims)
        if progressbar:
            iterable = progress_bar(range(num_sims))
        for i in iterable:
            sky = self.generate_skymap()
            cl = anafast(sky, lmax=lmax)
            simulations_pwr[i,:] = cl
        
        ensemble = Ensemble(simulations_pwr, self.lcm_params)
        return ensemble



    