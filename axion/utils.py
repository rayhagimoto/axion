from numpy import pi, cos
import colorsys
import numpy as np

# -------------------- CONSTANTS ------------------- #

degree = pi / 180

# ----------------- MISC. FUNCTIONS ---------------- #


def _init_kwargs_dict(kwargs):
    if kwargs == None:
        return {}
    else:
        return kwargs


def _angle_to_ell(ang, degrees=False):
    if not degrees:
        ang *= 180 / pi
    if ang < 30:
        ell = 100 / ang


def _check_resolution_sufficiency(nside, r, degrees=False):
    """Check that HEALPix resolution can resolve circles with angular
    radius r (r is given in radians or degrees)"""
    # if degrees:
    #     r *= pi/180
    # npix = 12*nside**2
    # pix_area = 4*pi/npix
    # circ_area = 2*pi*(1-cos(r))
    # if pix_area < circ_area:
    #     import warnings
    #     warnings.warn("HEALPix resolution may not be fine enough. Try increasing Nside for better results.")
    pass


def _check_only_one(**kwargs):
    res = 0
    error_msg = f"Only one of {', '.join(kwargs.keys())} expected but more than one was provided."
    for val in kwargs.values():
        if val is not None:
            res += 1
            if res > 1:
                raise ValueError(error_msg)


# COLOR METHODS 

def rgba(r,g,b,a):
    """rgba format color compatible with matplotlib.    
    r, g, b must be floats between 0 and 255.
    a must be a float between 0 and 1."""
    return np.array([r,g,b,a*255])/255

def set_opacity(color, alpha, format='RGBA'):
    """Change alpha channel of color to value of alpha.
        Parameters
        ----------
        color : 4-tuple
            Color whose opacity you want to change. Can be in any format specified by format arg.
        alpha : float
            Value to change alpha channel to. E.g. alpha=0. corresponds to completely transparent
            alpha = 1. corresopnds to completely opaque.
        format : str
            Format to interpret color as. Accepted values are 'RGB', 'RGBA', 'HSL', 'HSV'. Values
            of 3-tuple or 4-tuple must be between 0 and 1 (same convention as colorsys.)
        
        Returns
        ------
        _color : 4-tuple
            Color in RGBA format (suitable for matplotlib)
    """
    if format in ['RGBA', 'RGB', 'HSV', 'HSL']:

        if format == 'RGBA':
            _color = np.copy(color)
        
        else:
            _color = np.zeros(4)
        if format == 'RGB':
            _color[:-1] = color
        if format == 'HSV':
            _color[:-1] = colorsys.hsv_to_rgb(*color)
        if format == 'HSL':
            _color[:-1] = colorsys.hsl_to_rgb(*color)
        _color[3] = alpha
        return _color
    
    else:
        raise ValueError('Invalid format. Format must be either RGBA, RGB, HSV, or HSL.')

def set_lightness(color, b, format='RGBA'):
    assert 0 < b < 1

    if format in ['RGBA', 'RGB', 'HSV', 'HSL']:

        if format == 'RGBA':
            rgb_color = np.copy(color[:3])
            alpha = color[3]
            hls_color = np.array(colorsys.rgb_to_hls(*rgb_color))
            hls_color[1] = b 
            rgb_color = np.array(colorsys.hls_to_rgb(*hls_color))
            rgba_color = np.zeros(4)
            rgba_color[:3] = rgb_color
            rgba_color[3] = alpha
            return rgba_color
        if format == 'RGB':
            hls_color = np.array(colorsys.rgb_to_hls(*color))
        if format == 'HSV':
            hls_color = np.array(colorsys.hsv_to_hls(*color))
        if format == 'HSL':
            hls_color = np.array(color)
        

        return hls_color
    
    else:
        raise ValueError('Invalid format. Format must be either RGBA, RGB, HSV, or HSL.')


del pi, cos