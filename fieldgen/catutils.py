import numpy as np

from scipy import interpolate as sinterp

import astropy.io as astro_io

import numba as nb

import healpy as hp


def read_catalog(filename: str):
    hdul = astro_io.fits.open(name = filename)
    data = hdul[1].data
    return data

def save_catalog(filename: str, columnsnames: list, columns: list, overwrite: bool = True):
    '''
    https://het.as.utexas.edu/HET/Software/Astropy-1.0/io/fits/index.html#creating-a-new-table-file
    '''

    cols = [astro_io.fits.Column(name, format = 'E', array = arr) for name, arr in zip(columnsnames, columns)]
    cols = astro_io.fits.ColDefs(cols)
    tbhdu = astro_io.fits.BinTableHDU.from_columns(cols)
    return tbhdu.writeto(filename, overwrite = overwrite)




#https://stackoverflow.com/questions/66874819/random-numbers-with-user-defined-continuous-probability-distribution
#simple rejection sampling
#see also https://stackoverflow.com/questions/60559616/how-to-sample-from-a-distribution-given-the-cdf-in-python
#https://www.wikiwand.com/en/Inverse_transform_sampling#/Intuition
#https://web.mit.edu/urban_or_book/www/book/chapter7/7.1.3.html

def sample_from_function(pdff, pdffmax, Nitems, xmin, xmax, seed: int):

    items = np.array([])
    rng = np.random.default_rng(seed)
    
    Nresidual = Nitems
    somma = 0
    while (Nresidual > 0):
        x = rng.uniform(xmin, xmax, Nresidual)
        y = rng.uniform(0, pdffmax, Nresidual)
        pdf = pdff(x)
        
        selection = y < pdf
        Nselected = np.sum(selection)
        items = np.append(items, x[selection])
        somma += Nselected
        Nresidual -= Nselected
    
    return items


def get_interp(x, y):
    @nb.njit
    def interp_nb(x_vals):
        return np.interp(x_vals, x, y)
    return interp_nb

@nb.njit(parallel = True)
def _sample_from_function(pdff, pdffmax, Nitems, xmin, xmax, seed: int):

    items = np.empty(Nitems)
    np.random.seed(seed)
    
    Nresidual = Nitems
    somma = 0
    while (Nresidual > 0):
        x = np.random.uniform(xmin, xmax, Nresidual)
        y = np.random.uniform(0, pdffmax, Nresidual)
        pdf = pdff(x)
        
        selection = y < pdf
        Nselected = np.sum(selection)
        items = np.append(items, x[selection])
        somma += Nselected
        Nresidual -= Nselected
    
    return items


def sample_from_nz_pdf(z, dndz, Nitems, zmin = None, zmax = None, seed: int = None):
    print("Nitems", Nitems)
    #pdff = get_interp(z, dndz/np.trapz(dndz, z)) 
    pdff = sinterp.interp1d(z, dndz/np.trapz(dndz, z), kind = 'cubic')
    pdffmax = np.max(pdff(z))
    zmin = z.min() if zmin is None else zmin
    zmax = z.max() if zmax is None else zmax
    return sample_from_function(pdff, pdffmax, Nitems, zmin, zmax, seed)



def poisson_sampling(counts_map: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed = seed)
    return rng.poisson(lam = counts_map)


def delta_g_to_number_counts(delta_g_input_real: np.ndarray, No_observed: int, seed: int, mask: np.ndarray = 1, weights: np.ndarray = 1, alpha: float = 1., set_zero: bool = False) -> np.ndarray:

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
        set_zero: bool
            If True, sets to zero negative values of numbers.

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
    Ngal = np.nan_to_num(Ngal)*mask*np.nan_to_num(1/weights)/alpha**2.

    if (alpha == 1) and set_zero:
            print("alpha == 1, might set to 0 negative values")
            Ngal[np.where(Ngal < 0.)] = 0. #setting to -1 the delta_g_input_real
        
    return (poisson_sampling(Ngal, seed)).astype(int)


def get_catalog(seed: int, Npix: int, nside: int, number_counts: np.ndarray, z: np.ndarray, nz: np.ndarray, sample_z: bool = False):

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
        zs = np.hstack([sample_from_nz_pdf(z[ii], nz[ii], np.sum(number_counts_), seed = seed) for ii, number_counts_ in enumerate(number_counts)])
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