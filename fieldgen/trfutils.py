"""
Transforms utils to go back and forth between angular power spectra and correlation functions.
"""

import numpy as np
import flt

def cl2xi(Cls: np.ndarray, closed = False):

    ls = np.arange(0, len(Cls[0, 0]))

    factorcl = (2*ls+1)/(4*np.pi)

    coeffs = Cls*factorcl[None, None, :]

    inverse = lambda coeffs: flt.idlt(coeffs, closed = closed)
    corr_func = np.apply_along_axis(inverse, 2, coeffs)

    return corr_func

def theta(n, closed = False):
    '''
    Returns the theta for which the cl2xi are calculated, for a given n
    '''
    return flt.theta(n, closed = closed)


def xi2cl(xis: np.ndarray, closed = False):
    
    Ntheta = xis.shape[-1]

    ls = np.arange(0, Ntheta)
    factorcl = (2*ls+1)/(4*np.pi)

    forward = lambda vals: flt.dlt(vals, closed = closed)
    Clsout = np.apply_along_axis(forward, 2, xis)/factorcl[None, None, :]
    return Clsout


def cls_to_gcls(cls: np.ndarray, alpha_matrix: np.ndarray):
    xisinput = cl2xi(cls)/alpha_matrix[..., None]
    xigaussian = np.log(xisinput+1)
    clgaussian = xi2cl(xigaussian)
    return clgaussian