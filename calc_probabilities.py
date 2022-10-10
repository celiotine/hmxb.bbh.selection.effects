import numpy as np
import pandas as pd
import sys

import astropy.units as u
from astropy.cosmology import z_at_value
from astropy.cosmology import Planck15 as cosmo

##----------------------------------------------------------------------------------------------
## calculate the comoving volume elements for an array of bins in lookback time, "tbins"
def calc_cosmo_factors(tbins):
    
    dVdz = np.array([])

    for t_low, t_high in zip(tbins[:-1], tbins[1:]):
    
        ## get comoving volume element for this lookback time                         
        mid = (t_high - t_low)/2
        z = z_at_value(cosmo.lookback_time,  (t_low + mid) * u.Myr)
        
        this_dVdz = cosmo.differential_comoving_volume(z).value * 1/(1+z)                                                      
        dVdz = np.append(dVdz, this_dVdz)

    return dVdz

##-----------------------------------------------------------------------------------------------
## calculate individual detectable source weights
## "source start" is an array of lookback times when the sources "begin"
## "source end" is an array of lookback times when the sources "end"
## "p_det" is an array of the relative detection probability once a source exceeds a detection threshold
def calc_weights_norms(source_start, source_end, p_det, dVdz, tbins):
    
    n_sources = len(source_start)
    weights = np.array([])

    binsize = np.abs(tbins[1]-tbins[0])
    
    for source in range(n_sources):

        source_cut = np.where((tbins <= source_start[source]+binsize) & (tbins > source_end[source]-binsize))
        dVdz_cut = dVdz[source_cut[0][:-1]]

        this_binary_weight = p_det[source] * dVdz_cut * binsize            
        weights = np.append(weights, np.sum(this_binary_weight))
         
    return weights

##-------------------------------------------------------------------------------
## read population files
df = pd.read_csv(sys.argv[1])        ## sampled universe population parameters
df_local = pd.read_csv(sys.argv[2])  ## sampled local population parameters

##--------------------------------------------------------------------------------
## population cuts
n_samp = df.shape[0]              ## total sample size
n_samp_local = df_local.shape[0]

## bbh cuts; bbh includes all bound bbhs (alive & merged)
bbh_h = df[df['this_BBH_H']]
bbh_h_local = df_local[df_local['this_BBH_H']]

bbh_z0 = df[df['this_BBH_z0']]
bbh_z0_local = df_local[df_local['this_BBH_z0']]

bbh_h_alive = bbh_h[bbh_h['bin_state'] == 0]     ## bound bbhs that have not merged in a Hubble time
bbh_h_alive_local = bbh_h_local[bbh_h_local['bin_state'] == 0]

## merging bbh cuts
bbhm_h = df[df['this_BBHm_H']]
bbhm_h_local = df_local[df_local['this_BBHm_H']]

bbhm_z0 = df[df['this_BBHm_z0']]
bbhm_z0_local = df_local[df_local['this_BBHm_z0']]

## hmxb cuts
hmxb_h = df[df['this_HMXB_H']]
hmxb_h_local = df_local[df_local['this_HMXB_H']]

hmxb_z0 = df[df['this_HMXB_z0']]
hmxb_z0_local = df_local[df_local['this_HMXB_z0']]

## combined cuts
hmxb_z0_bbhm_h = hmxb_z0[hmxb_z0["this_BBHm_H"]]
hmxb_z0_bbhm_h_local = hmxb_z0_local[hmxb_z0_local["this_BBHm_H"]]

##----------------------------------------------------------------------------------

zmin = 0         ## redshift bounds should match those from sampling                          
zmax = 20
zmax_local = 0.05

nbins = 100000
nbins_local = 5000

## create time bins for weight calculation
tmin = cosmo.lookback_time(zmin).to(u.Myr).value
tmax = cosmo.lookback_time(zmax).to(u.Myr).value
tmax_local = cosmo.lookback_time(zmax_local).to(u.Myr).value

tbins = np.linspace(tmin, tmax, nbins)
tbins_local = np.linspace(tmin, tmax_local, nbins_local)  

binsize = np.abs(tbins[1] - tbins[0])
binsize_local = np.abs(tbins_local[1] - tbins_local[0])

try: dVdz = np.load("dVdz.npy")
except: dVdz = calc_cosmo_factors(tbins)

try: dVdz_local = np.load("dVdz_local.npy")
except: dVdz_local = calc_cosmo_factors(tbins_local) 

np.save("dVdz.npy", dVdz)
np.save("dVdz_local.npy", dVdz_local)

##----------------------------------------------------------------------------------
## calculate normalization for all alive binaries
universe_norms =  calc_weights_norms(df['tL_bin_form'].values, df['tL_bin_end'].values, np.ones(df.shape[0]), dVdz, tbins)
local_norms = calc_weights_norms(df_local['tL_bin_form'].values, df_local['tL_bin_end'].values, np.ones(df_local.shape[0]), dVdz_local, tbins_local)

## calculate binary population weights
bbh_h_weights =  calc_weights_norms(bbh_h['tL_bin_form'].values, bbh_h['tL_bin_end'].values, np.ones(bbh_h.shape[0]), dVdz, tbins)
bbh_h_weights_local = calc_weights_norms(bbh_h_local['tL_bin_form'].values, bbh_h_local['tL_bin_end'].values, np.ones(bbh_h_local.shape[0]), dVdz_local, tbins_local)

bbh_z0_weights =  calc_weights_norms(bbh_z0['tL_bin_form'].values, bbh_z0['tL_bin_end'].values, np.ones(bbh_z0.shape[0]), dVdz, tbins)
bbh_z0_weights_local = calc_weights_norms(bbh_z0_local['tL_bin_form'].values, bbh_z0_local['tL_bin_end'].values, np.ones(bbh_z0_local.shape[0]), dVdz_local, tbins_local)

bbhm_h_weights =  calc_weights_norms(bbhm_h['tL_bin_form'].values, bbhm_h['tL_bin_end'].values, np.ones(bbhm_h.shape[0]), dVdz, tbins)
bbhm_h_weights_local = calc_weights_norms(bbhm_h_local['tL_bin_form'].values, bbhm_h_local['tL_bin_end'].values, np.ones(bbhm_h_local.shape[0]), dVdz_local, tbins_local)

bbhm_z0_weights =  calc_weights_norms(bbhm_z0['tL_bin_form'].values, bbhm_z0['tL_bin_end'].values, np.ones(bbhm_z0.shape[0]), dVdz, tbins)
bbhm_z0_weights_local = calc_weights_norms(bbhm_z0_local['tL_bin_form'].values, bbhm_z0_local['tL_bin_end'].values, np.ones(bbhm_z0_local.shape[0]), dVdz_local, tbins_local)

hmxb_h_weights =  calc_weights_norms(hmxb_h['tL_bin_form'].values, hmxb_h['tL_bin_end'].values, np.ones(hmxb_h.shape[0]), dVdz, tbins)
hmxb_h_weights_local = calc_weights_norms(hmxb_h_local['tL_bin_form'].values, hmxb_h_local['tL_bin_end'].values, np.ones(hmxb_h_local.shape[0]), dVdz_local, tbins_local)

hmxb_z0_weights =  calc_weights_norms(hmxb_z0['tL_bin_form'].values, hmxb_z0['tL_bin_end'].values, np.ones(hmxb_z0.shape[0]), dVdz, tbins)
hmxb_z0_weights_local = calc_weights_norms(hmxb_z0_local['tL_bin_form'].values, hmxb_z0_local['tL_bin_end'].values, np.ones(hmxb_z0_local.shape[0]), dVdz_local, tbins_local)


## calculate weights for detectable HMXB sources
tL_XRB_emitend = hmxb_z0['tL_XRB_emitobs'].values - hmxb_z0['emit15'].values
hmxb_obs_weights = calc_weights_norms(hmxb_z0['tL_XRB_emitobs'].values, tL_XRB_emitend, np.ones(hmxb_z0.shape[0]), dVdz, tbins)

tL_XRB_emitend_local = hmxb_z0_local['tL_XRB_emitobs'].values - hmxb_z0_local['emit15'].values
hmxb_obs_weights_local = calc_weights_norms(hmxb_z0_local['tL_XRB_emitobs'].values, tL_XRB_emitend_local, np.ones(hmxb_z0_local.shape[0]), dVdz_local, tbins_local)

tL_XRB_emitend_bbhmh = hmxb_z0_bbhm_h['tL_XRB_emitobs'].values - hmxb_z0_bbhm_h['emit15'].values
hmxb_obs_bbhm_h_weights = calc_weights_norms(hmxb_z0_bbhm_h['tL_XRB_emitobs'].values, tL_XRB_emitend_bbhmh, np.ones(hmxb_z0_bbhm_h.shape[0]), dVdz, tbins)

tL_XRB_emitend_bbhmh_local = hmxb_z0_bbhm_h_local['tL_XRB_emitobs'].values - hmxb_z0_bbhm_h_local['emit15'].values
hmxb_obs_bbhm_h_weights_local = calc_weights_norms(hmxb_z0_bbhm_h_local['tL_XRB_emitobs'].values, tL_XRB_emitend_bbhmh_local, np.ones(hmxb_z0_bbhm_h_local.shape[0]), dVdz_local, tbins_local)

## calculate weights for detectable BBHm sources
T = 0.0001    ## observing window [Myr]
tau = binsize
tau_local = binsize_local

tL_merge = cosmo.lookback_time(bbhm_z0['z_m']).to(u.Myr).value
bbhm_obs_weights = T/tau * calc_weights_norms(tL_merge, tL_merge, bbhm_z0['p_det'].values, dVdz, tbins)

tL_merge_local = cosmo.lookback_time(bbhm_z0_local['z_m']).to(u.Myr).value
bbhm_obs_weights_local = T/tau_local * calc_weights_norms(tL_merge_local, tL_merge_local, bbhm_z0_local['p_det'].values, dVdz_local, tbins_local)

##----------------------------------------------------------------------------------
## calculate probabilities
pBBH_H = np.sum(bbh_h_weights)/np.sum(universe_norms)
pBBH_H_local = np.sum(bbh_h_weights_local)/np.sum(local_norms)

pBBH_z0 = np.sum(bbh_z0_weights)/np.sum(universe_norms)
pBBH_z0_local = np.sum(bbh_h_weights_local)/np.sum(local_norms)

pBBHm_H = np.sum(bbhm_h_weights)/np.sum(universe_norms)
pBBHm_H_local = np.sum(bbhm_h_weights_local)/np.sum(local_norms)

pBBHm_z0 = np.sum(bbhm_z0_weights)/np.sum(universe_norms)
pBBHm_z0_local = np.sum(bbhm_z0_weights_local)/np.sum(local_norms)

pBBHm_obs = np.sum(bbhm_obs_weights)/np.sum(local_norms)
pBBHm_obs_local = np.sum(bbhm_obs_weights_local)/np.sum(local_norms)

pHMXB_H = np.sum(hmxb_h_weights)/np.sum(universe_norms)
pHMXB_H_local = np.sum(hmxb_h_weights_local)/np.sum(local_norms)

pHMXB_z0 = np.sum(hmxb_z0_weights)/np.sum(universe_norms)
pHMXB_z0_local = np.sum(hmxb_z0_weights_local)/np.sum(local_norms)

pHMXB_obs = np.sum(hmxb_obs_weights)/np.sum(universe_norms)
pHMXB_obs_local = np.sum(hmxb_obs_weights_local)/np.sum(local_norms)

## conditional probabilities

# p(BBHm_H | HMXB_obs)
pBBHm_H_n_HMXB_obs = np.sum(hmxb_obs_bbhm_h_weights)/np.sum(universe_norms)
pBBHm_H_HMXB_obs = pBBHm_H_n_HMXB_obs/pHMXB_obs

pBBHm_H_n_HMXB_obs_local = np.sum(hmxb_obs_bbhm_h_weights_local)/np.sum(local_norms)
pBBHm_H_HMXB_obs_local = pBBHm_H_n_HMXB_obs_local/pHMXB_obs_local

# p(HMXB_obs | BBHm_H)
pHMXB_obs_BBHm_H = pBBHm_H_n_HMXB_obs/pBBHm_H
pHMXB_obs_BBHm_H_local = pBBHm_H_n_HMXB_obs_local/pBBHm_H_local

# p(HMXB_obs | HMXB_z0)
pHMXB_obs_HMXB_z0 = pHMXB_obs/pHMXB_z0
pHMXB_obs_HMXB_z0_local = pHMXB_obs_local/pHMXB_z0_local

# p(BBHm_obs | BBHm_z0)
pBBHm_obs_BBHm_z0 = pBBHm_obs/pBBHm_z0
pBBHm_obs_BBHm_z0_local = pBBHm_obs_local/pBBHm_z0_local
##----------------------------------------------------------------------------------
## print and save output
f = open("probabilities.txt", "w")

f.write("UNIVERSE PROBABILITIES\n")
f.write("p(BBHm_H): " + str(pBBHm_H) + "\n")
f.write("p(BBHm_z0): " + str(pBBHm_z0) + "\n")
f.write("p(BBHm_obs): " + str(pBBHm_obs) + "\n")
f.write("p(BBHm_obs | BBHm_z0): " + str(pBBHm_obs_BBHm_z0) + "\n")
f.write("\n")
f.write("p(HMXB_H): " + str(pHMXB_H) + "\n")
f.write("p(HMXB_z0): " + str(pHMXB_z0) + "\n")
f.write("p(HMXB_obs): " + str(pHMXB_obs) + "\n")
f.write("p(HMXB_obs | HMXB_z0): " + str(pHMXB_obs_HMXB_z0) + "\n")
f.write("\n")
f.write("p(BBHm_H n HMXB_obs): " + str(pBBHm_H_n_HMXB_obs) + "\n")
f.write("p(BBHm_H | HMXB_obs): " + str(pBBHm_H_HMXB_obs))
f.write("\n")
f.write("p(HMXB_obs | BBHm_H): " + str(pHMXB_obs_BBHm_H) + "\n")

f.write("\n-----------------------------------------\n")
f.write("\nLOCAL PROBABILITIES\n")
f.write("p(BBHm_H): " + str(pBBHm_H_local) + "\n")
f.write("p(BBHm_z0): " + str(pBBHm_z0_local) + "\n")
f.write("p(BBHm_obs): " +str(pBBHm_obs_local) + "\n")
f.write("p(BBHm_obs | BBHm_z0): " + str(pBBHm_obs_BBHm_z0_local) + "\n")
f.write("\n")
f.write("p(HMXB_H): " + str(pHMXB_H_local) + "\n")
f.write("p(HMXB_z0): " + str(pHMXB_z0_local) + "\n")
f.write("p(HXMB_obs): " + str(pHMXB_obs_local) + "\n")
f.write("p(HMXB_obs | HMXB_z0): " + str(pHMXB_obs_HMXB_z0_local) + "\n")
f.write("\n")
f.write("p(BBHm_H n HMXB_obs): " + str(pBBHm_H_n_HMXB_obs_local) + "\n")
f.write("p(BBHm_H | HMXB_obs): " + str(pBBHm_H_HMXB_obs_local) + "\n")
f.write("\n")
f.write("p(HMXB_obs | BBHm_H): " + str(pHMXB_obs_BBHm_H_local) + "\n")

f.close()
