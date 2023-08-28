"""
Generates conditioned Gaussian simulations from a given map, and power spectra.
"""

import numpy as np

from plancklens import shts

import healpy as hp

from typing import Callable, List, Tuple

from . import utils



class ConditionedSims(object):
    """
    Class to generate conditioned Gaussian simulations from a given map, and power spectra.
    """

    def __init__(self, Nfields: int, get_AB_gaussian: Callable, realized_field_index: int = 0):
        """
        Parameters
        ----------
        Nfields : int
            Number of fields.
        get_AB_gaussian : callable
            Function that returns the cross-correlation power spectrum between field A and B.
        realized_field_index : int
            Index of the fixed field in the power spectra function. e.g. "k" for a fixed CMB lensing realization, with "kk" for CMB lensing power spectrum
        """

        self.Nfields = Nfields
        self.get_AB_gaussian = get_AB_gaussian
        indices = utils.get_indices_for_cls_list(Nfields) #indices for the extra Nfields

        self.filter_correlated, self.filter_uncorrelated = self.procesess_cls_list(Nfields, indices, get_AB_gaussian, realized_field_index)


    def generate_alm(self, seed: int, input_alms: np.ndarray):
        """
        Generates the conditioned Gaussian simulations.
        """

        #call this to generate a random seed for each realization
        #np.random.RandomState
        #rng = np.random.default_rng(seed = seed)
        np.random.seed(seed = seed)
        correlated_alms = self.get_correlated_part(input_alms, self.filter_correlated)

        uncorrelated_alms = self.get_uncorrelated_part(input_alms, self.filter_uncorrelated)

        total_alms = [np.nan_to_num(u+c) for u, c in zip(uncorrelated_alms, correlated_alms)]

        return total_alms
    
    def generate_maps(self, seed:int, input_alms: np.ndarray, nside: int):
        total_alms = self.generate_alm(seed, input_alms)
        alm2map = lambda alm: shts.alm2map(alm.copy(), nside)
        maps = list(map(alm2map, total_alms))
        return maps
    

    def get_correlated_part(self, input_alms: np.ndarray, filters: list) -> List[np.ndarray]:
        '''
        filters: list of filters of the type kg/gg
        '''
        correlated_alms = [hp.sphtfunc.almxfl(input_alms, filter_) for filter_ in filters]
        return correlated_alms

    def get_uncorrelated_part(self, input_alms: np.ndarray, uncorr_cov_part: list) -> np.ndarray:
        '''
        uncorr_cov_part = $\Sigma_g = \Sigma_{{g_\mathrm{i}}{g_\mathrm{j}}}-\frac{\vec{C}^{\kappa g}\vec{C}^{\kappa g,T}}{C^{\kappa\kappa}}$
        '''
        lmax = hp.Alm.getlmax(input_alms.size)
        uncorrelated_alms = hp.sphtfunc.synalm(uncorr_cov_part, lmax = lmax, new = True)
        return uncorrelated_alms


    @staticmethod
    def procesess_cls_list(Nfields: int, indices: Tuple[List, List], get_AB_gaussian: Callable, fixed_index: int = 0):
        """
        Processes the power spectra list to be used in the simulations generation in a healpy friendly format.

        The filter_correlated, uncorrelated lists will be used to generate the new correlated alm in this way:

        alm = filter_correlated * alm_fixed + Gaussian(0 mean, uncorrelated)

        Parameters
        ----------
        Nfields : int
            Number of fields in addition the to the fixed one.
        indices : list
            List of indices for the cls list.
        get_AB_gaussian : callable
            Function that returns the cross-correlation power spectrum between field A and B.
        fixed_index : int
            Index of the fixed field in the power spectra function.
        
        Returns
        -------
        filter_correlated, uncorrelated : list, list
            List of the filtered and unfiltered power spectra.
        """
        
        clsfields = [get_AB_gaussian(i, j) for i, j in zip(indices[0], indices[1])]
        clsfieldscross = [get_AB_gaussian(i, fixed_index) for i in range(Nfields)]

        #now, make some processing so that I have ready info for sims generation
        clsfieldscross = np.array(clsfieldscross)
        clsfields = np.array(clsfields)

        kk = get_AB_gaussian(fixed_index, fixed_index)
        
        filter_correlated = clsfieldscross/kk #this is a numpy array, shape of (Nfield, Nls)

        clsfieldscross = clsfieldscross/np.sqrt(kk) 
 
        matrix = np.einsum('i..., j...', clsfieldscross, clsfieldscross)
        matrix = np.array([matrix[:, i, j] for i, j in zip(indices[0], indices[1])])
        covij = np.array(clsfields)
        uncorrelated = covij-matrix 

        return filter_correlated, uncorrelated

