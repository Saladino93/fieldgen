"""
Utilities.
"""

from typing import Any
import numpy as np

import itertools

import scipy

import healpy as hp


def list_alm_copy(alm_list: np.ndarray or list, mmaxin:int or None, lmaxout:int, mmaxout:int):
    if type(alm_list) == list:
        return [alm_copy(a, mmaxin, lmaxout, mmaxout) for a in alm_list]
    else:
        return alm_copy(alm_list, mmaxin, lmaxout, mmaxout)
#copied from delensalot
def alm_copy(alm:np.ndarray, mmaxin:int or None, lmaxout:int, mmaxout:int):
    """Copies the healpy alm array, with the option to change its lmax

        Parameters
        ----------
        alm :ndarray
            healpy alm arrays to copy.
        mmaxin: int or None
            mmax parameter of input array (can be set to None or negative for default)
        lmaxout : int
            new alm lmax
        mmaxout: int
            new alm mmax


    """
    alms = np.atleast_2d(alm)
    ret = []
    for alm in alms:
        lmaxin = hp.Alm.getlmax(alm.size, mmaxin)
        if mmaxin is None or mmaxin < 0: mmaxin = lmaxin
        if (lmaxin == lmaxout) and (mmaxin == mmaxout):
            ret.append(np.copy(alm))
        else:
            _ret = np.zeros(hp.Alm.getsize(lmaxout, mmaxout), dtype=alm.dtype)
            lmax_min = min(lmaxout, lmaxin)
            for m in range(0, min(mmaxout, mmaxin) + 1):
                idx_in =  m * (2 * lmaxin + 1 - m) // 2 + m
                idx_out = m * (2 * lmaxout+ 1 - m) // 2 + m
                _ret[idx_out: idx_out + lmax_min + 1 - m] = alm[idx_in: idx_in + lmax_min + 1 - m]
            ret.append(_ret)
    ret = np.array(ret)
    if ret.shape[0] == 1:
        return ret[0]
    else:
        return ret


def divide(a, b):
    return np.divide(a, b, out = np.zeros(a.shape, dtype = float), where = b!= 0)


def get_indices_for_cls_list(N: int):
    '''
    This function returns the indices for the cls list useful for generating spectra in healpy with new order

    Parameters
    ----------
    N : int
        Number of fields
    Returns
    -------
    indices : tuple
        Tuple of two lists, one for the first field and one for the second field
    '''

    xs = [x for x in itertools.chain(*[[i for i in range(N-j)] for j in range(N)])]
    ys = [x for x in itertools.chain(*[[i+j for i in range(N-j)] for j in range(N)])]
    indices = (xs, ys)

    return indices




class SpectraGetter(object):
    def __init__(self, cls: np.ndarray, realized_field_index: int = 0):
        self.cls = cls
        self.realized_field_index = realized_field_index

    def get_AB(self, a: int, b: int):
        """
        Returns the cross-correlation power spectrum between field A and B.

        Has to be implemented by the user.

        Here an example.
        """

        a = a+1 if a != self.realized_field_index else 0
        b = b+1 if b != self.realized_field_index else 0

        return self.cls[a, b]

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.get_AB(*args, **kwds)
    


def simple_binning(cl, bin_edges = np.arange(10, 4000, 200)):
    l = np.arange(0, len(cl))
    lcl = l*cl
    sums = scipy.stats.binned_statistic(l, l, statistic = "sum", bins = bin_edges)
    cl = scipy.stats.binned_statistic(l, lcl, statistic = "sum", bins = bin_edges)
    cl = cl[0] / sums[0]
    cents = (bin_edges[1:]+bin_edges[:-1])/2
    return cents, cl