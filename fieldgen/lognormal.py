"""
Class to generate lognormal conditioned simulations or just new correlated lognormal simulations.
"""

from plancklens import shts
import numpy as np
from . import utils, logutils, conditionedsims, trfutils


class LognormalConditionedSims(conditionedsims.ConditionedSims):

    def __init__(self, Nfields: int, get_AB: utils.SpectraGetter, lambdas: np.ndarray, means_of_fields: np.ndarray, realized_field_index: int = None):
        """
        Parameters
        ----------
        Nfields : int
            Number of extra fields.
        get_AB: utils.SpectraGetter
            Object that returns the cross-correlation power spectrum between field A and B.
        realized_field_index : int
            Index of the fixed field in the power spectra function. e.g. "k" for a fixed CMB lensing realization, with "kk" for CMB lensing power spectrum
        lambdas: np.ndarray
            Array of lambda parameters for the lognormal fields. The first element is the lambda for the realized field, the others are for the other fields.
        means_of_fields: np.ndarray
            Array of the means of the fields. The first element is the mean of the realized field, the others are for the other fields.
        """

        self.lambdas = lambdas
        self.lambda_others = lambdas[1:] #lambda for the other fields (not the realized one)

        realized_field_index = get_AB.realized_field_index if realized_field_index is None else realized_field_index

        self.means_of_fields = means_of_fields #mean of the fields (not the realized one)
        self.means_of_fields_others = means_of_fields[1:] #mean of the fields (not the realized one)

        alphas = logutils.get_alpha(means_of_fields, lambdas)
        alpha_matrix = np.outer(alphas, alphas)
        self.alpha_matrix = np.nan_to_num(alpha_matrix)

        self.get_AB = get_AB
        self.get_AB_gaussian = self._transform_cls_getter_to_gaussian()

        super().__init__(Nfields, self.get_AB_gaussian, realized_field_index)


    def _transform_cls_getter_to_gaussian(self):
        """
        Transforms the power spectra getter to a getter for the Gaussian field related to the lognormal field.
        """
        cls = self.get_AB.cls
        clsg = trfutils.cls_to_gcls(cls, self.alpha_matrix)
        return utils.SpectraGetter(clsg, self.get_AB.realized_field_index)

    def _get_gaussian_alms(self, seed: int, input_alms: np.ndarray):
        alms = super().generate_alm(seed, input_alms)
        return alms
       
    def generate_maps(self, seed: int, input_alms: np.ndarray, nside: int):
        alms = self._get_gaussian_alms(seed, input_alms)
        alm2map = lambda alm: shts.alm2map(alm.copy(), nside)
        maps_gaussian = list(map(alm2map, alms))
        vargauss = np.array([np.var(m) for m in maps_gaussian])
        expmu = (self.means_of_fields_others+self.lambda_others)*np.exp(-vargauss*0.5)
        #mapsout = ne.evaluate('exp(maps_gaussian)')

        mapsout = np.exp(maps_gaussian)
        mapsout *= expmu[..., None]
        mapsout -= self.lambda_others[..., None]
        maps = np.array(mapsout)
        return maps
    

    def _process_input(self, mappa, nside):
        lambda_k = self.lambdas[0]
        #y_kappa = np.log(kappa/lambda_k + 1)
        y_kappa = np.log(mappa + lambda_k)
        y_kappa_lm = shts.map2alm(y_kappa.copy(), lmax = nside*3-1)
        return y_kappa_lm

    def generate_alm(self, seed: int, input_alms: np.ndarray, nside: int, lmax = None, out_real: bool = False, process: bool = False):
        """
        Generates the conditioned Lognormal simulations.
        """

        input_alms = self._process_input(input_alms, nside) if process else input_alms
        
        lmax = 3*nside-1 if lmax is None else lmax
        maps = self.generate_maps(seed, input_alms, nside)
        map2alm = lambda m: shts.map2alm(m.copy(), lmax)
        result = list(map(map2alm, maps))
        result = (result, maps) if out_real else result
        return result


