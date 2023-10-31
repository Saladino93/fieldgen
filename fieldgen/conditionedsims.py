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

        self.filter_correlated, self.filter_uncorrelated, self.gg_parts = self.procesess_cls_list(Nfields, indices, get_AB_gaussian, realized_field_index)


    @staticmethod
    def alm2map(alm, nside):
        return shts.alm2map(alm.copy(), nside = nside)
        #return hp.alm2map(alm, nside)
    
    @staticmethod
    def map2alm(mappa, lmax):
        return shts.map2alm(mappa.copy(), lmax = lmax)
        #return hp.map2alm(mappa, lmax)

    def generate_alm(self, seed: int, input_alms: np.ndarray, nside: int = None):
        """
        Generates the conditioned Gaussian simulations.
        """

        #call this to generate a random seed for each realization
        #np.random.RandomState
        #rng = np.random.default_rng(seed = seed)
        np.random.seed(seed = seed)
        correlated_alms = self.get_correlated_part(input_alms, self.filter_correlated)

        uncorrelated_alms = self.get_uncorrelated_part(input_alms, self.filter_uncorrelated, nside, self.gg_parts)

        correlated_alms = correlated_alms if nside is None else utils.list_alm_copy(correlated_alms, hp.Alm.getlmax(correlated_alms[0].size), 3*nside-1, 3*nside-1)
        total_alms = [np.nan_to_num(u+c) for u, c in zip(uncorrelated_alms, correlated_alms)]

        return total_alms
    
    def generate_maps(self, seed:int, input_alms: np.ndarray, nside: int, cast_to_nside_lmax: False):
        """
        The input_alms might have a lower lmax compared to the required lmax from nside (3*nside-1). 
        In some cases, e.g. cov matrix from Namaster, it might be useful to copy the input_alms to 
        have same lmax from nside.

        Note, by construction, only the uncorrelated part will have non-zero modes beyond the correlated
        part defined by input_alms.
        """
        total_alms = self.generate_alm(seed, input_alms, nside = nside if cast_to_nside_lmax else None)
        alm2map = lambda alm: self.alm2map(alm, nside)
        maps = list(map(alm2map, total_alms))
        return maps
    

    def get_correlated_part(self, input_alms: np.ndarray, filters: list) -> List[np.ndarray]:
        '''
        filters: list of filters of the type kg/gg
        '''
        correlated_alms = [hp.sphtfunc.almxfl(input_alms, filter_) for filter_ in filters]
        return correlated_alms

    def get_uncorrelated_part(self, input_alms: np.ndarray, uncorr_cov_part: list, nside: int = None, gg_parts: list = None) -> np.ndarray:
        '''
        uncorr_cov_part = $\Sigma_g = \Sigma_{{g_\mathrm{i}}{g_\mathrm{j}}}-\frac{\vec{C}^{\kappa g}\vec{C}^{\kappa g,T}}{C^{\kappa\kappa}}$
        '''

        lmax = hp.Alm.getlmax(input_alms.size)

        #would be better to put in when getting lists..., cleaner
        if (nside is not None) and (lmax < 3*nside-1):
            for i in range(len(uncorr_cov_part)):
                lmax_n = 3*nside-1
                zeros = np.zeros(lmax_n)
                zeros[:lmax] = uncorr_cov_part[i][:lmax]
                zeros[lmax:lmax_n] = gg_parts[i][lmax:lmax_n]
                #if kappa is band limitied, and its lmax is smaller than the required resolution, generate extra power spectra
                uncorr_cov_part[i] = zeros
            lmax = lmax_n

        uncorrelated_alms = hp.sphtfunc.synalm(uncorr_cov_part, lmax = lmax, new = True)

        return uncorrelated_alms


    @staticmethod
    def procesess_cls_list(Nfields: int, indices: Tuple[List, List], get_AB: Callable, fixed_index: int = 0):
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
        get_AB : callable
            Function that returns the cross-correlation power spectrum between field A and B.
        fixed_index : int
            Index of the fixed field in the power spectra function.
        
        Returns
        -------
        filter_correlated, uncorrelated : list, list
            List of the filtered and unfiltered power spectra.
        """
        
        clsfields = [get_AB(i, j) for i, j in zip(indices[0], indices[1])]
        clsfieldscross = [get_AB(i, fixed_index) for i in range(Nfields)]

        #now, make some processing so that I have ready info for sims generation
        clsfieldscross = np.array(clsfieldscross)
        clsfields = np.array(clsfields)

        kk = get_AB(fixed_index, fixed_index)
        
        filter_correlated = utils.divide(clsfieldscross, kk) #clsfieldscross/kk #this is a numpy array, shape of (Nfield, Nls)

        clsfieldscross = utils.divide(clsfieldscross, np.sqrt(kk)) #clsfieldscross/np.sqrt(kk) 
 
        matrix = np.einsum('i..., j...', clsfieldscross, clsfieldscross)
        matrix = np.array([matrix[:, i, j] for i, j in zip(indices[0], indices[1])])
        covij = np.array(clsfields)
        uncorrelated = covij-matrix 

        return filter_correlated, uncorrelated, clsfields