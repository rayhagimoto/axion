import numpy as np

def integrate(f,dx): #
    """Computes definite integral of f(x) via trapezoidal integration
     on the interval (a,b) were f(a) = f[0] and f(b) = f[-1]"""
    return 0.5*dx*(f[0] + f[-1]) + dx*np.sum(f[1:-1]) # trapezoidal integration

def integrate2d(f,dx,axis=0): #
    """Takes a function f defined on a 2D array and uses trapezoidal integration
    to integrate along the 0th axis (zero-based indexing)"""
    return 0.5*dx*(f[0,:] + f[-1,:]) + dx*np.sum(f[1:-1,:], axis=axis) # trapezoidal integration