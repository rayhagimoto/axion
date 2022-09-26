import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import root
from ...cosmology import hbarH0, H
from ...utils import _init_kwargs_dict

from ..analytic import model_3_cl


def plot_power(
    model_params, 
    ell, 
    simulations_pwr, 
    ax=None,
    showanalytic=False,
        legendre_n=20, 
    spaghetti=False,
    degrees=True,
    color=None,
    xmin=3,
    xmax=250,
    ymin=None,
    ymax=None,
    subplots_kwargs=None,
    plot_kwargs=None
    ):

    zeta0 = model_params['zeta0']
    anomaly = model_params['anomaly']
    xi0 = model_params['xi0']
    ma = model_params['ma']
    zf = root(lambda z : 3*hbarH0*H(z) - ma, x0=1.).x[0]

    fn = f"zeta0={zeta0:.2f}-xi0={xi0:.2f}-anomaly={anomaly:.2f}-zf={int(zf)}"
    
    num_sims, lmax_plus_1 = simulations_pwr.shape
    mean = np.mean(simulations_pwr, axis=0)
    
    mask = np.logical_and(ell >= xmin, ell <= xmax)
    _ell = ell[mask]
    _y = _ell*(_ell+1)/2/np.pi*mean[mask]
    if ymin is None:
        ymin = 3e-1*_y.min()
    if ymax is None:
        ymax = 3*_y.max()
    if degrees: 
        units = r'$[\mathrm{deg}^2]$'
    else: 
        units = r'$[\mathrm{rad}^2]$'
    

    if ax is None:
        subplots_kwargs = _init_kwargs_dict(subplots_kwargs)
        fig, ax = plt.subplots(**subplots_kwargs)
    if color is None:
        color = ax._get_lines.get_next_color()
    
    plt.title(
        f"$\\zeta_0={zeta0:.2f},\\ \mathcal{{A}}^2\\xi_0={anomaly**2*xi0:.2f}\\ m_a={ma:.2f},\\ N_{{\\mathrm{{sims}}}}={num_sims}$",
        pad = 20
        )

    ax.plot(ell, ell*(ell+1)/2/np.pi*mean, color=color, linewidth=1, label='mean of sims')
    if spaghetti:
        for i in range(num_sims):
            ax.plot(ell, ell*(ell+1)/2/np.pi*simulations_pwr[i,:], 'k-', alpha=0.5*np.tanh(20/num_sims), linewidth=1)
    else:
        upper68 = np.zeros_like(ell, dtype=float)
        lower68 = np.zeros_like(ell, dtype=float)
        for j, l in enumerate(ell):
            upper68[j], lower68[j] = np.percentile(l*(l+1)/2/np.pi*simulations_pwr[:,l],[16,84])
        ax.fill_between(ell, lower68, upper68, color=color, edgecolor=None, alpha=0.125, zorder=1)
    
    if showanalytic:
        log10ma = np.log10(ma)
        ells = np.arange(xmin, xmax + 1, 1)
        cl = model_3_cl(ells, params=[zeta0, anomaly**2*xi0, log10ma], n=legendre_n)
        if degrees:
            cl *= (180/np.pi)**2
        ax.plot(ells, ells*(ells+1)/2/np.pi*cl, color='0.3', linestyle='--', linewidth=1, label='analytic')
    
    plt.xscale('log'); plt.yscale('log');
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$\ell(\ell+1)\,C_\ell^{\Phi\Phi}\, / 2\pi$  ' + units)


    plt.legend(frameon=False)
    fig.patch.set_alpha(0.)
    ax.patch.set_alpha(0.)
    
    return ax

def plot_sky_count(
    model_params, 
    sky, 
    count, 
     ):
    from healpy import mollview

    zeta0 = model_params['zeta0']
    anomaly = model_params['anomaly']
    xi0 = model_params['xi0']
    ma = model_params['ma']
    zf = root(lambda z : 3*hbarH0*H(z) - ma, 0)[0]
    

    mollview(
        count, min=np.min(count), max=np.max(count), cmap='binary_r'
        )

    mollview(
        sky, min=np.min(sky), max=np.max(sky)
        )