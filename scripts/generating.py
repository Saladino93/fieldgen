import healpy as hp

from plancklens.utils import alm_copy
from plancklens import shts

import numpy as np

import pathlib

from fieldgen import lognormal, conditionedsims as cs, utils as fgutils, logutils


Nfields = 3
lambdas = np.array([4, 1.089, 1.106])

#maps
nside = 1024

lmax = 6000

fixed_index = 2 #fixed index for the CMB lensing convergence field


nside_gen = 2048

scratch = pathlib.Path("/home/users/d/darwish/scratch")

direc_phi = scratch/pathlib.Path("signal_sims")

out_dir = scratch/pathlib.Path("prova")


def prepare_get_functions_for_conditioned_sims(theoryinputdir, lmax = 3072):

    clkk = np.loadtxt(pathlib.Path(theoryinputdir/"clkk_input.dat"))[:lmax]

    get_kk = lambda: clkk[:lmax]

    get_kg = lambda b: np.loadtxt(theoryinputdir/f'fit_kg_{b}.txt').T[1][:lmax]

    def get_gg(a, b):
        try:
            return np.loadtxt(theoryinputdir/f'fit_gg_{a}_{b}.txt').T[1][:lmax]
        except:
            return np.loadtxt(theoryinputdir/f'fit_gg_{b}_{a}.txt').T[1][:lmax]
    
    return get_kk, get_kg, get_gg


def get_get_AB(theoryinputdir, fixed_index = 0, lmax = 6000, nside = 1024, apply_pixwin = True):


    #lmax = min(lmax, 3*nside)

    get_kk, get_kg, get_gg = prepare_get_functions_for_conditioned_sims(theoryinputdir, lmax)

    pixwin = hp.pixwin(nside)[:lmax] if apply_pixwin else 1.0

    def get_AB(a, b):
        if (a == fixed_index) or (b == fixed_index):
            if a == b:
                return get_kk()
            elif a == fixed_index:
                return get_kg(b)*pixwin
            else:
                return get_kg(a)*pixwin
        else:
            return get_gg(a, b)*pixwin**2
    return get_AB



spectra_dir = pathlib.Path("/home/users/d/darwish/ACTdr6xDESy3/products/sims/fits")
get_AB = get_get_AB(spectra_dir, fixed_index = fixed_index, lmax = lmax, nside = nside, apply_pixwin = False)
#create a 3D matrix from get_AB
cls = np.zeros((Nfields, Nfields, lmax))

for i in range(Nfields-1):
    for j in range(Nfields-1):
        cls[i+1, j+1] = get_AB(i, j)
    cls[0, i+1] = get_AB(fixed_index, i)
    cls[i+1, 0] = cls[0, i+1]
cls[0, 0] = get_AB(fixed_index, fixed_index)

del get_AB

Getter = fgutils.SpectraGetter(cls, fixed_index)


seeds = np.arange(10)
for seed in seeds:
    name_phi = f"fullskyPhi_alm_{seed:05}.fits"
    #for some reaons pxlensing giving weird results, so do by hand
    philm = hp.read_alm(direc_phi/name_phi).astype(np.complex128)
    lmax = hp.Alm.getlmax(philm.size)
    lphi = np.arange(lmax + 1)
    factor = lphi*(lphi + 1)/(2)
    klm = hp.almxfl(philm, factor)
    kappa = shts.alm2map(klm.copy(), nside = nside_gen)

    mean_kappa = logutils.get_mean_from_map(kappa) #note mean calc at nside_gen
    LC = lognormal.LognormalConditionedSims(Nfields-1, Getter, realized_field_index = fixed_index, lambdas = lambdas, means_of_fields = np.repeat(mean_kappa, Nfields))
    alms, maps = LC.generate_alm(seed, kappa, nside, out_real = True, process = True)

    for i, alm in enumerate(alms):
        hp.write_alm(str(out_dir/f"alm_{i}_{seed:05}.fits"), alm, overwrite = True)

    out_lambdas_other = [logutils.get_lambda_from_mappa(m) for m in maps]
    print(seed, out_lambdas_other/lambdas[1:]-1)

    G = cs.ConditionedSims(Nfields-1, Getter, realized_field_index = fixed_index)
    Galms = G.generate_alm(seed, klm)

    for i, alm in enumerate(Galms):
        hp.write_alm(str(out_dir/f"Galm_{i}_{seed:05}.fits"), alm, overwrite = True)

    lmaxkappa = hp.Alm.getlmax(klm.size)
    mapsout_lm = [alm_copy(m, min(lmax, 3*nside-1)) for m in alms]


    Gmapsout_lm = [alm_copy(m, min(lmax, 3*nside-1)) for m in Galms]
    kappa_map = klm
    klm_lm_copy = alm_copy(kappa_map, min(lmax, 3*nside-1))
    cls_out_cross_input = [hp.alm2cl(mlm, klm_lm_copy) for mlm in mapsout_lm]
    Gcls_out_cross_input = [hp.alm2cl(mlm, klm_lm_copy) for mlm in Gmapsout_lm]

    cls_out_gg_cross = [hp.alm2cl(mapsout_lm[i], mapsout_lm[j]) for i in range(Nfields-1) for j in range(i, Nfields-1)]
    Gcls_out_gg_cross = [hp.alm2cl(Gmapsout_lm[i], Gmapsout_lm[j]) for i in range(Nfields-1) for j in range(i, Nfields-1)]
    
    ls = np.arange(len(cls_out_cross_input[0]))

    cls_out_cross_input = np.array(cls_out_cross_input)
    Gcls_out_cross_input = np.array(Gcls_out_cross_input)
    cls_out_gg_cross = np.array(cls_out_gg_cross)
    Gcls_out_gg_cross = np.array(Gcls_out_gg_cross)

    calcs_matrix = np.c_[ls, cls_out_cross_input.T, cls_out_gg_cross.T, Gcls_out_cross_input.T, Gcls_out_gg_cross.T]
    np.savetxt(out_dir/f"cls_{seed:05}.txt", calcs_matrix)

