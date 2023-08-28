"""
Utilities.
"""

from typing import Any
import numpy as np

import itertools

import scipy


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