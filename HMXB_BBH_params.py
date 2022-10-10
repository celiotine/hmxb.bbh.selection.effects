import pandas as pd
import numpy as np
import glob
import sys
import re
from scipy import interpolate
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
import astropy.units as u
from cosmic.evolve import Evolve
from cosmic.sample.initialbinarytable import InitialBinaryTable

#----------------------------------------------------------------------------------                                

## phsyical constants                                                                            
c = 2.99e10        ## speed of light in cm/s                                                  
secyr = 3.154e7    ## seconds per year                                                            
Myr = 1e6          ## years per Myr                                                            
Msun = 1.989e33    ## grams per solar mass
Rsun = 6.957e8     ## meters per solar radius                                                      
Lsun = 3.839e33    ## erg/sec per solar luminosity
G = 6.67e-11       ## in m^3 kg^-1 s^-2                                               

#-----------------------------------------------------------------------------------                    
## analytic approximation for P(omega) CDF from Dominik et al. 2015                                                                 
def P_omega(omega_values):
    return 0.374222*(1-omega_values)**2 + 2.04216*(1-omega_values)**4 - 2.63948*(1-omega_values)**8 + 1.222543*(1-omega_values)**10
 
#----------------------------------------------------------------------------------                                                                  
### Monte Carlo sampling for detections above the given SNR threshold                                                          \
                                                                                                                                
def calc_detection_prob(m1, m2, z_merge):

    ## constants that reflect LIGO design sensitivity                                                                           
    d_L8 = 1  ## in Gpc                                                                                                         
    M_8 = 10  ## in Msun                                                                                                       \
                                                                                                                                
    SNR_thresh = 8

    ## approximate typical SNR from Fishbach et al. 2018                                                                        
    M_chirp = (m1*m2)**(3./5)/(m1+m2)**(1./5)
    d_C = cosmo.comoving_distance(z_merge).to(u.Gpc).value
    d_L = (1+z_merge)*d_C

    rho_0 = 8*(M_chirp*(1+z_merge)/M_8)**(5./6)*d_L8/d_L   ## this is the "typical/optimal" SNR                                 
    if (rho_0 < SNR_thresh): return 0

    omega_thresh = SNR_thresh/rho_0
    p_det = P_omega(omega_thresh)

    return p_det

#-----------------------------------------------------------------------------------# 

def calc_flux(current_BH_mass, initial_BH_mass, mdot_BH, d_L):
    
    bolometric_correction = 0.8
    where_lower_masses = current_BH_mass < np.sqrt(6)*initial_BH_mass

    eta_lower_masses = 1 - np.sqrt(1-(current_BH_mass/(3*initial_BH_mass))**2)
    eta = np.where(where_lower_masses, eta_lower_masses, 0.42)

    ## make sure accretion is Eddington-limited                                                                                
    X = 1
    M_edd = 2.6e-7 * current_BH_mass/10 * ((1+X)/1.7)**-1 * (0.1/eta)  ## in M_sun/yr                                                                       
    acc_rate = mdot_BH/(1-eta)  ## accretion rate in Msun/year
    

    try: index = np.where(acc_rate > M_edd)[0]
    except: index = False
    ## if super-Eddington accretion occurs (due to bug in COSMIC), replace emission timestep with previous timestep
    if index is not False: acc_rate.values[index] = acc_rate.values[index+1]

    luminosity = bolometric_correction * eta * acc_rate * c**2 * Msun/secyr   ## accretion luminosity in erg/sec  

    ## convert luminosity to flux
    flux = luminosity/(4 * np.pi * d_L**2)   ## flux in erg/s/cm^2

    return acc_rate, M_edd, luminosity, flux

#----------------------------------------------------------------------------------

columns=['bin_num', 'metallicity', 'merger_type', 'bin_state', 'delay_time', 'z_f', 'tL_bin_form', 'tL_bin_end', 'z_m', 'p_det', 'tL_XRB_start', 'tL_XRB_end', 'tL_XRB_emitobs', 'ZAMS_mass_k1','ZAMS_mass_k2', 'final_mass_k1', 'final_mass_k2', 'final_k1', 'final_k2', 'BH_mass_i', 'donor_mass_i', 'donor_type', 'XRB_sep_i', 'XRB_porb_i', 'XRB_ecc_i', 'avg_lum', 'max_lum', 'max_flux', 'max_mdot_BH', 'max_mdot_acc', 'mdot_edd', 'emit11', 'emit13', 'emit15', 'emit_tot', 'this_BBH_H', 'this_BBH_z0', 'this_BBHm_H', 'this_BBHm_z0', 'this_HMXB_H', 'this_HMXB_z0']
df_all = pd.DataFrame(columns=columns)

sample_initC = pd.read_hdf(sys.argv[1])  ## initC file for binary population sampled from COSMIC
run_ID = int(sys.argv[2])                ## SLURM job array ID, used to slice initC file
this_sample_section = sample_initC.truncate(before=run_ID*599, after=run_ID*599+599)  


## set COSMIC re-evolution timestep to be small during the XRB phase
dtp = 0.001
timestep_conditions = [['kstar_1<10', 'kstar_2=14', 'mass_1 > 5', 'dtp=0.001'], ['kstar_1=14', 'kstar_2<10', 'mass_2 > 5', 'dtp=0.001']]

#----------------------------------------------------------------------------------

for index, binary in this_sample_section.iterrows():
    
    bin_num = binary['bin_num']    
    met = binary['metallicity']

    #-------------------------------------------------------------------------
    ## SET INITIAL EVOLUTION PARAMETERS                                                                                                                                              
    z_merge = -1; p_det = -1; p_cosmic_zm = -1; delay_time = -1

    BH_mass_i = -1; donor_mass_i = -1; donor_type = -1; XRB_sep_i = -1
    XRB_porb_i = -1; XRB_ecc_i = -1; avg_lum = -1; max_lum = -1; max_flux = -1; emit11 = -1; emit13 = -1
    emit15 = -1; emit_tot = -1; lookback_time_XRB = -1; lookback_time_end_XRB = -1; lookback_time_XRB_emitobs = -1
    max_mdot_BH = -1; max_mdot_acc = -1; mdot_edd = -1

    ## boolean state params for this binary
    this_BBH_z0 = False; this_BBHm_z0 = False
    this_HMXB_z0 = False
    this_BBH_H = False; this_BBHm_H = False
    this_HMXB_H = False
    #-------------------------------------------------------------------------

    ## re-evolve this single binary with fine timestep resolution
    bpp, bcm, initC, kick_info = Evolve.evolve(initialbinarytable = binary.drop(['z_ZAMS','Z','tlb_ZAMS']).to_frame().T, timestep_conditions=timestep_conditions)

    merger_type = int(bcm['merger_type'].iloc[-1])   ## get final merger state of the binary
    bin_state = bcm['bin_state'].iloc[-1]            ## get final bin state of the binary

    z_f = binary['z_ZAMS']                                           ## redshift of binary ZAMS formation
    d_L = (1+z_f)*cosmo.comoving_distance(z_f).to(u.cm).value        ## luminosity distance of binary ZAMS formation, in cm for flux calculation
    lookback_time_form = cosmo.lookback_time(z_f).to(u.Myr).value    ## lookback time of binary ZAMS formation

    ## find when the binary lifetime ends (merged or disrupted)
    try: t_bin_end_index = np.where(bpp['sep'] <= 0)[0][0]
    except: t_bin_end_index = -1

    t_bin_end = bpp['tphys'].iloc[t_bin_end_index]
    tL_bin_end = lookback_time_form - t_bin_end          ## lookback time of binary end
    if (tL_bin_end < 0): tL_bin_end = 0                  ## if binary has not ended by today, set end to be the tL=0

    ## "cosmological weight" of the system using comoving volume element
    dVdz = cosmo.differential_comoving_volume(z_f).value   ## in Mpc^3 sr^-1

    ## get ZAMS masses for the binary                                                
    ZAMS_mass_k1 = bpp['mass_1'].iloc[0]
    ZAMS_mass_k2 = bpp['mass_2'].iloc[0]

    ## get final COSMIC merge types for the binary                                 
    final_k1 = bpp['kstar_1'].iloc[-1]
    final_k2 = bpp['kstar_2'].iloc[-1]

    ## check if binary is bound BBH after a Hubble time (i.e. is an alive BBH)
    if (final_k1 == 14 and final_k2 == 14 and bin_state == 0): 
        this_BBH_H = True

    ## check if binary forms a bound BBH by z=0
    try: bbh_form_index = np.where((bpp['kstar_1']==14) & (bpp['kstar_2']==14) & (bpp['sep']>0))[0][0]
    except: bbh_form_index = None
    
    ## if binary DOES form a bound BBH, check if it has formed by today (z=0)
    if (bbh_form_index is not None):
        bbh_form_time = bpp['tphys'].iloc[bbh_form_index]
        lookback_time_bbh_form = lookback_time_form - bbh_form_time       ## lookback time of BBH formation
            
        if (lookback_time_bbh_form > 0):
            this_BBH_z0 = True
                
    #----------------------------------------------------------------------------------

    ## CASE 1: system merges                                                                  
    ## alive or disrupted system have merge_index = -1 in COSMIC (do not want these)                                            
    if (merger_type != -1):

        ## check if BBH merger in COSMIC
        if (merger_type == 1414): 
            this_BBH_H = True; this_BBHm_H = True

        try: merge_index = np.where(bpp['evol_type']==6)[0][0]      ## find index of BBH merger
        except: merge_index = -2   ## common envelope merger
        ## COSMIC does not set evol_type = 6 for CE mergers
        ## system always merges at second-to-last timestep in evolution
            
        final_mass_k1 = bpp['mass_1'].iloc[merge_index-1]        ## final mass of object 1
        final_mass_k2 = bpp['mass_2'].iloc[merge_index-1]        ## final mass of object 2

        if (merge_index > 0):     ## read, "if binary is not a commone envelope merger"

            delay_time = bpp['tphys'].iloc[merge_index]                ## delay time of merger
            lookback_time_merge = lookback_time_form - delay_time      ## lookback time of merger

            if (lookback_time_merge > 0):     ## if binary has merged by today...

                z_merge = z_at_value(cosmo.lookback_time, lookback_time_merge*u.Myr)       ## merger redshift
                p_det = calc_detection_prob(final_mass_k1, final_mass_k2, z_merge)         ## detection probability for this merger
                    
                if (this_BBHm_H):  ## if binary has merged by today AND is a BBH merger...
                    this_BBH_z0 = True; this_BBHm_z0 = True

                dVdz_zm = cosmo.differential_comoving_volume(z_merge).value   ## in Mpc^3 sr^-1                                          
                p_cosmic_zm = dVdz_zm * (1+z_merge)**-1


    ## CASE 2: system does not merge           
    else:
        final_mass_k1 = bpp['mass_1'].iloc[-1]
        final_mass_k2 = bpp['mass_2'].iloc[-1]

    #----------------------------------------------------------------------------------

    ## CASE A: system undergoes an HMXB phase (defined only for BH accretors)                                            
    ## if so, there are 3+ rows in the re-evolved bcm frame                      
    if (bcm.shape[0] >= 3):

        ## get bcm index where first BH is formed                                   
        try: BH1_index = np.where(bcm['kstar_1'] == 14)[0][0]
        except: BH1_index = np.infty

        try: BH2_index = np.where(bcm['kstar_2'] == 14)[0][0]
        except: BH2_index = np.infty

        ## CASE Ai: BH1 (kstar_1) is formed first                                
        if (BH2_index > BH1_index):
                
            XRB_index = BH1_index
            BHobj = "kstar_1"; donorObj = "kstar_2"
            BHmass = "mass_1"; donorMass = "mass_2"
            BHmdot = "deltam_1"; donormdot = "deltam_2" 
            donorRad = "rad_2"

        # CASE Aii: BH2 (kstar_2) is formed first                                  
        else:
            XRB_index = BH2_index
            BHobj = "kstar_2"; donorObj = "kstar_1"
            BHmass = "mass_2"; donorMass = "mass_1"
            BHmdot = "deltam_2"; donormdot = "deltam_1"
            donorRad = "rad_1"

        XRB_sep_i = bcm['sep'].iloc[XRB_index]

        ## ensure system is not disrupted (has sep=-1 in COSMIC)
        if (XRB_sep_i > 0):
                
            this_HMXB_H = True

            ## check if HMXB is formed by z=0
            XRB_index_bpp = np.where(bpp[BHobj] == 14)[0][0]
            t_begin_XRB = bpp['tphys'].iloc[XRB_index_bpp]
            lookback_time_XRB = lookback_time_form - t_begin_XRB   

            if (lookback_time_XRB > 0):   ## if HMXB has formed by today...
                this_HMXB_z0 = True
                z_HMXB = z_at_value(cosmo.lookback_time,  lookback_time_XRB * u.Myr)   ## redshift of HMXB formation
                d_L_HMXB = (1+z_HMXB)*cosmo.comoving_distance(z_HMXB).to(u.cm).value   ## luminosity distance of HMXB formation
        

            ## find end of HMXB phase using bcm array (only evolves during HMXB lifetime)
            t_end_XRB = bcm['tphys'].iloc[-2]       
            lookback_time_end_XRB = lookback_time_form - t_end_XRB

            ## get binary parameters at beginning of XRB phase                             
            donor_mass_i = bcm[donorMass].iloc[XRB_index]
            donor_type = bcm[donorObj].iloc[XRB_index]
            BH_mass_i = bcm[BHmass].iloc[XRB_index]
            XRB_sep_i = bcm['sep'].iloc[XRB_index]
            XRB_porb_i = bcm['porb'].iloc[XRB_index]
            XRB_ecc_i = bcm['ecc'].iloc[XRB_index]

            if (this_HMXB_z0):       ## if HMXB has formed by today, calculate emission parameters
                
                acc_rate, edd_rate, luminosity, flux = calc_flux(bcm[BHmass], np.ones(len(bcm[BHmass]))*BH_mass_i, bcm[BHmdot], d_L_HMXB)

                #------------------------------------------------------------------------------------------
                ## CE occurs in one timestep in COSMIC, which generates inaccurate HMXB emission
                ## must remove emission from this timestep if CE occurs
                ## find if/when CE begins during XRB phase                    
                try: CE = np.where((bpp['evol_type'] == 7) & (bpp['tphys'] >= t_begin_XRB))[0][0]
                except: CE = None

                if (CE is not None):     ## if a comomon envelope occurs...
                    
                    CE_begin_end = np.where((bpp['evol_type'] == 7) & (bpp['tphys'] >= t_begin_XRB))[0]   ## (start, end) index values of CE phase(s)
                    t_CE = bpp['tphys'].iloc[CE_begin_end].values

                    ## loop through all CE phases
                    for i in range(len(t_CE)):
                        index = np.abs(bcm['tphys'] - t_CE[i]).values.argmin()
                        flux.values[index] = flux.values[index-1]   ## set timestep of CE emission to previous timestep's emission
                #------------------------------------------------------------------------------------------
        
                ## replace NaN and inf flux values with zeros
                flux = np.nan_to_num(flux, nan=0, posinf=0, neginf=0)                

                ## final emission parameters
                max_flux = max(flux)
                max_lum = max(luminosity)
                avg_lum = np.sum(luminosity[2:]*dtp)/np.abs(t_end_XRB - t_begin_XRB)

                
                max_mdot_BH = max(bcm[BHmdot])
                max_mdot_acc = max(acc_rate)
                mdot_edd = edd_rate.values[np.argmax(acc_rate.values)]   ## Eddington accretion rate with the eta of max accretion rate

                ## calculate duration of observable emission
                emit15 = len(np.where(flux > 5e-15)[0])*dtp
                emit13 = len(np.where(flux > 1e-13)[0])*dtp
                emit11 = len(np.where(flux > 1e-11)[0])*dtp

                ## total duration of XRB phase                                                                         
                emit_tot = bcm['tphys'].iloc[-2] - bcm['tphys'].iloc[XRB_index]

                if (emit15 > 0):
                    ## find the lookback time of the start of observable XRB emission                                                                       
                    emit_obs_start = np.where(flux > 5e-15)[0][0]
                    t_begin_emit_obs = bcm['tphys'].iloc[emit_obs_start]
                    lookback_time_XRB_emitobs = lookback_time_form - t_begin_emit_obs

                    ## time(observable emission) will differ significantly from time(HMXB) for extended mass transfer WD systems
                    ## do not want to count these as observable
                    if (lookback_time_XRB_emitobs < 0): 
                        emit15 = -1; emit13 = -1; emit11 = -1; lookback_time_XRB_emitobs = -1

        #----------------------------------------------------------------------------------
    ## save all parameters to data frame
    df = pd.DataFrame([[bin_num, met, merger_type, bin_state, delay_time, z_f, lookback_time_form, tL_bin_end, z_merge, p_det, lookback_time_XRB, lookback_time_end_XRB, lookback_time_XRB_emitobs, ZAMS_mass_k1, ZAMS_mass_k2, final_mass_k1, final_mass_k2, final_k1, final_k2, BH_mass_i, donor_mass_i, donor_type, XRB_sep_i, XRB_porb_i, XRB_ecc_i, avg_lum, max_lum, max_flux, max_mdot_BH, max_mdot_acc, mdot_edd, emit11, emit13, emit15, emit_tot, this_BBH_H, this_BBH_z0, this_BBHm_H, this_BBHm_z0, this_HMXB_H, this_HMXB_z0]], columns=columns)

    df_all = df_all.append(df, sort=False, ignore_index=True)

df_all.to_csv("HMXB_output/sampled_HMXB_params_" + str(run_ID) + ".csv", index=False)

