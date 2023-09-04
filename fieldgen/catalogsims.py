import numpy as np

import healpy as hp

from . import catutils

import pathlib

import logging

log = logging.getLogger(__name__)

class SimCatalog(object):
    
    def __init__(self, nside: int, directory: pathlib.Path):
        self.nside = nside
        self.directory = directory

    @staticmethod
    def check_if_binary_mask(mask: np.ndarray):
        return np.all(np.isin(mask, [0, 1]))

    def _delta_g_to_number_counts(self, No_observed, delta_g_input_real, seed, mask, weights, alpha):
        '''
        This function transforms a delta_g map to a number counts map. For alpha < 1 see https://arxiv.org/pdf/1708.01536.pdf

        Parameters
        ----------
        No_observed : int
            Number of observed galaxies.
        delta_g_input_real : np.ndarray
            Delta_g map.
        seed : int
            Seed for the random number generator.
        mask : np.ndarray
            Mask of the map. e.g. completeness mask.
        weights : np.ndarray
            Weights of the map. This is useful if you want to contaminate the map with systematics.
        alpha : float
            Scaling factor for the delta_g map.

        Returns
        -------
        np.ndarray
            Number counts map.
        '''

        #this is here so that if I do not want to sample, e.g. because I have a kappa map, I can just return the input map
        if No_observed == 0:
            return delta_g_input_real

        mappa = (delta_g_input_real*alpha+1.)
        
        Npix = np.sum(mask) #len(mappa[mask != 0])
        Ngal = mappa*No_observed/Npix
        #Ngal = hp.alm2map(hp.map2alm(Ngal, pol = False, use_pixel_weights = True), nside, pixwin = True)#NOTE: PIXWIN
        Ngal = np.nan_to_num(Ngal)*mask*np.nan_to_num(1/weights)/alpha**2.

        if alpha == 1:
            Ngal[np.where(Ngal < 0.)] = 0. #setting to -1 the delta_g_input_real
            
        return (self.poisson_sampling(Ngal, seed)).astype(int)

    @staticmethod
    def poisson_sampling(counts_map: int, seed: int):
        rng = np.random.default_rng(seed = seed)
        return rng.poisson(lam = counts_map)
    
    
    def create_field_with_systematics(self, Ngal: np.ndarray, weight_map: np.ndarray, seed: int) -> np.ndarray:
        Ngal_corrected = Ngal/weight_map
        return self.poisson_sampling(Ngal_corrected, seed)

    
    def delta_g_to_number_counts(self, seed, No_observed, delta_g_input_real, mask, weights, alpha):
        if type(No_observed) is not list:
            No_observed = [No_observed]
            delta_g_input_real = [delta_g_input_real]
            weights = [weights]

        return [self._delta_g_to_number_counts(No, delta_g, seed, mask, weight, alpha) for No, delta_g, weight in zip(No_observed, delta_g_input_real, weights)]
    

    def save_catalog(self, filename: str, seed: int, No_observed: int, delta_g_input_real: np.ndarray, mask: np.ndarray, weights: np.ndarray, alpha: float, z: np.ndarray, nz: np.ndarray, columnsnames: list = ['RA', 'DEC', 'Z', 'ZBIN', 'WEIGHT'], sample_z: bool = False):
        number_counts = self.delta_g_to_number_counts(seed, No_observed, delta_g_input_real, mask, weights, alpha)
        Npix = len(mask)
        catutils.save_catalog(filename = self.directory/f'{filename}.fits', columnsnames = columnsnames, columns = self._get_catalog(seed, Npix, self.nside, number_counts, z, nz, sample_z = sample_z))
        return number_counts

    @staticmethod
    def _get_catalog(seed: int, Npix: int, nside: int, number_counts: np.ndarray, z: np.ndarray, nz: np.ndarray, sample_z: bool = True):
        ipix = np.arange(0, Npix)
        ras, decs = hp.pixelfunc.pix2ang(nside, ipix, lonlat = True)
        coords = np.vstack((ras, decs))

        if type(number_counts) is not list:
            number_counts = [number_counts]
            z = [z]
            nz = [nz]

        #sample angular coordinates
        catalog = np.hstack([np.repeat(coords, number_counts_.astype(int), axis = -1) for number_counts_ in number_counts])


        if sample_z:
            log.info("Sampling redshifts")
            #sample redshifts
            zs = np.hstack([catutils.sample_from_nz_pdf(z[ii], nz[ii], np.sum(number_counts_), seed = seed) for ii, number_counts_ in enumerate(number_counts)])
        else:
            zs = np.hstack([np.repeat(ii, np.sum(number_counts_)) for ii, number_counts_ in enumerate(number_counts)])

        ras = catalog[0, :]
        decs = catalog[1, :]

        del catalog

        weights_maps = [np.ones(Npix) for _ in range(len(number_counts))]

        fake_weights = np.concatenate([np.repeat(weights, number_counts_.astype(int), axis = -1) for weights, number_counts_ in zip(weights_maps, number_counts)])

        weights = fake_weights

        #just a label to identify each redshift bin
        zbin = np.concatenate([i*np.ones(np.sum(number_counts_)) for i, number_counts_ in enumerate(number_counts)])

        return (ras, decs, zs, zbin, weights)
