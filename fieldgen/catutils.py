import numpy as np

from scipy import interpolate as sinterp

import astropy.io as astro_io

import numba as nb


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


def delta_g_to_number_counts(delta_g_input_real: np.ndarray, No_observed: int, seed: int, mask: np.ndarray = 1, weights: np.ndarray = 1, alpha: float = 1.) -> np.ndarray:
    
    #this is here so that if I do not want to sample, e.g. because I have a kappa map, I can just return the input map
    if No_observed == 0:
        return delta_g_input_real
    
    mappa = (delta_g_input_real*alpha+1.)
    
    Npix = np.sum(mask) #len(mappa[mask != 0])

    Ngal = mappa*No_observed/Npix
    Ngal = np.nan_to_num(Ngal)*mask*np.nan_to_num(1/weights)/alpha**2.
        
    return (poisson_sampling(Ngal, seed)).astype(int)