import numpy as np

try:
    import numba
    from numba import float64

    @numba.njit(float64(float64[:]), parallel = True)
    def get_mean_from_map(mappa: np.ndarray):
        '''
        Gets the mean from a map using numba
        '''
        Npix = len(mappa)
        somma = 0
        for i in numba.prange(Npix):
            somma += mappa[i]
        return somma/Npix

    @numba.njit(float64(float64[:]), parallel = True)
    def get_variance_from_map(mappa: np.ndarray):
        '''
        Gets the variance from a map using numba
        '''
        Npix = len(mappa)
        somma = 0
        for i in numba.prange(Npix):
            somma += mappa[i]**2.
        return somma/Npix-get_mean_from_map(mappa)**2.

    
    @numba.njit(float64(float64[:]), parallel = True)
    def get_skew_from_map(mappa: np.ndarray):
        '''
        Gets the skewness from a map using numba
        '''
        Npix = len(mappa)
        somma = 0
        for i in numba.prange(Npix):
            somma += mappa[i]**3.
        return somma/Npix/get_variance_from_map(mappa)**1.5
except:

    print("Numba not installed, using numpy instead")

    def get_mean_from_map(mappa: np.ndarray):
        return np.mean(mappa)

    def get_variance_from_map(mappa: np.ndarray):
        return np.mean(mappa**2.)-np.mean(mappa)**2.

    def get_skew_from_map(mappa: np.ndarray):
        return np.mean((mappa-get_mean_from_map(mappa))**3.)/np.mean(mappa**2.)**1.5





def y_skew(skew):
    '''
    Formula (12) from https://arxiv.org/pdf/1602.08503.pdf
    '''
    result = 2+skew**2.+skew*np.sqrt(4+skew**2.)
    result /= 2
    return np.power(result, 1/3.)

def get_lambda_from_skew(skew, var, mu):
    lmbda = np.sqrt(var)/skew*(1+y_skew(skew)+1/y_skew(skew))-mu
    return lmbda 

def get_alpha(mu, lmbda):
    '''
    Below formula (7) from https://arxiv.org/pdf/1602.08503.pdf
    '''
    return mu+lmbda

def get_mu_gauss(alpha, var):
    '''
    Gets the mu parameter for the Gaussian distribution for the log-normal
    '''
    result = np.log(alpha**2./np.sqrt(var+alpha**2.))
    return result

def get_sigma_gauss(alpha, var):
    '''
    Gets the sigma parameter for the Gaussian distribution for the log-normal.

    Here the variance is the variance of the wanted log-normal field.
    '''
    result = np.log(1+var/alpha**2.)
    result = np.sqrt(result)
    return result

def get_lambda_from_mappa(mappa: np.ndarray):
    '''
    Gets the lambda parameter for the log-normal distribution from a map.
    '''
    skewness = get_skew_from_map(mappa)
    variance = get_variance_from_map(mappa)
    mean = get_mean_from_map(mappa)
    lmbda = get_lambda_from_skew(skewness, variance, mean)
    return lmbda

suppress = lambda l, lsup, supindex: np.exp(-1.0*np.power(l/lsup, supindex))

def process_cl(inputcl: np.ndarray, lsup: float = 7000, supindex: float = 10):
    ls = np.arange(0, len(inputcl))
    result = inputcl*suppress(ls, lsup, supindex)
    return result
