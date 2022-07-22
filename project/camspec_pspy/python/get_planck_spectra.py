'''
This script is used to compute the power spectra of the Planck data.
To run it:
python get_planck_spectra.py global.dict
'''

import numpy as np
import healpy as hp
from pspy import so_dict, so_map, so_mcm, sph_tools, so_spectra, pspy_utils
import sys
import time
import planck_utils
from copy import deepcopy

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

windows_dir = "windows"
mcms_dir = "mcms"
spectra_dir = "spectra"

pspy_utils.create_directory(spectra_dir)

cmbfreqs = d["cmb_freqs"]
cleanfreqs = d["clean_freqs"]
freqs = cmbfreqs + cleanfreqs
niter = d["niter"]
lmax = d["lmax"]
cltype = d["type"]

misspix_flag = d["misspix_flag"] # 'none', 'external', or 'internal'
largevals_flag = d["largevals_flag"] # 'none', 'zero', or 'error'
binning_file = d["binning_file"]
remove_mono_dipo_t = d["remove_mono_dipo_T"]
remove_mono_dipo_pol = d["remove_mono_dipo_pol"]
remove_mono_george = d["remove_mono_george"]

use_pol = d["use_pol"]
do_decouple = d["do_decouple"]
beam_template = d["planck_beam_template"] # None for no beam deconv, otherwise % (freq1, split1, freq2, split2)

pixwin = d["pixwin"] # True to deconv pixwin, else False
splits = d["splits"]
planck_dipole = d["planck_dipole"]
halfringsplits = d["halfringsplits"]

experiment = "Planck" 

if use_pol:
    spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
    beamspectra = spectra
    spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]
    ncomp = 3
else:
    spectra = None
    beamspectra = ["TT"]
    spin_pairs = None
    ncomp = 1

speckinds = d["speckinds"]
if speckinds == 'both':
    speckinds = ['TEB', 'TQU']
elif speckinds == 'TEB' or speckinds == 'TQU':
    speckinds = [speckinds]
else:
    raise ValueError("'speckinds' is %s; should be 'TEB', 'TQU', or 'both'" % speckinds)

if cltype == 'both':
    types = ["Cl", "Dl"]
else:
    types = [cltype]

if splits == "" or splits == ["full"]:
    #splittags = ["GHz"] * len(splits)
    splittags = [""] * len(splits)
else:
    splittags = splits

alms = {}

print("Compute Planck 2018 spectra")

if planck_dipole is not None:
    dipole = so_map.read_map(planck_dipole).data
else:
    dipole = 0
    
for freq in freqs:
    print('Calculating Alms %s' % freq)
    for hrsplit in halfringsplits:
        maps = d["map_%s%s" % (freq, hrsplit)]
        for hm, map in zip(splits, maps):
            print(hm)
            # Read in map and subtract planck dipole
            if use_pol:
                pl_map = so_map.read_map("%s" % map, fields_healpix=(0, 1, 2))
                pl_map.data[0] -= dipole
            else:
                pl_map = so_map.read_map("%s" % map, fields_healpix=0)
                pl_map.data -= dipole
            pl_map.data *= 10**6 # Convert to uK
            # Apply missing additional pixel mask if necessary
            if misspix_flag == 'external':
                wcov_dictname = "wcov_map_%s%s" % (freq, hrsplit)
            else:
                wcov_dictname = None
            pl_map.data = planck_utils.zero_misspix(pl_map.data, map, misspix_flag, wcov_dictname, d, ncomp=ncomp)    # Apply missing pixels mask
            # Special for cleaning frequencies
            if freq in cleanfreqs:
                window_freqs = cmbfreqs # Calculate with masks of all frequencies, same split
            else:
                window_freqs = [freq]  # Just this frequency's mask
            # Load mask(s)
            pl_map_orig = pl_map
            for window_freq in window_freqs:
                pl_map = deepcopy(pl_map_orig)
                window = d["window_%s_%s" % (window_freq, hm)]
                window_t = so_map.read_map(window, fields_healpix=0)
                if use_pol:
                    window_pol = so_map.read_map(window, fields_healpix=1)        
                    window_tuple = (window_t, window_pol)
                else:
                    window_pol = window_t
                    window_tuple = window_t # for T-only get_alms doesn't actually want a tuple, it wants an so_map

                pl_map.data = planck_utils.zero_large_values(pl_map.data, (window_t.data, window_pol.data), largevals_flag, map, ncomp=ncomp) # Zero large values or raise error
                del window_t, window_pol

                ## Remove mono/dipole
                if (remove_mono_george and (remove_mono_dipo_t or remove_mono_dipo_pol)):
                    raise ValueError("raise_mono_george and raise_mono_dipo cannot both be True")
                if remove_mono_george:
                    if use_pol:
                        mean = np.sum(pl_map.data[0]*window_tuple[0].data) / np.sum(window_tuple[0].data)
                        pl_map.data[0] -= mean                        
                    else:
                        mean = np.sum(pl_map.data*window_tuple.data) / np.sum(window_tuple.data)
                        pl_map.data -= mean

                elif remove_mono_dipo_t:
                    if use_pol:
                        pl_map.data[0] = planck_utils.subtract_mono_di(pl_map.data[0], window_tuple[0].data, pl_map.nside)
                    else:
                        pl_map.data = planck_utils.subtract_mono_di(pl_map.data, window_tuple.data, pl_map.nside)
                        
                if remove_mono_dipo_pol:
                    if not use_pol: raise ValueError("remove_mono_dipo_pol = True but use_pol is False")
                    pl_map.data[1] = planck_utils.subtract_mono_di(pl_map.data[1], window_tuple[1].data, pl_map.nside)
                    pl_map.data[2] = planck_utils.subtract_mono_di(pl_map.data[2], window_tuple[1].data, pl_map.nside)

                ## Calculate alms
                for kind in speckinds:
                    print(kind)
                    if kind == 'TQU':
                        use_pol_alm = False
                        raise ValueError("Only TEB possible with un-modified pspy")
                    elif kind == 'TEB':
                        use_pol_alm = True
                    else:
                        raise ValueError("'kind' is %s; should be 'TEB' or 'TQU'" % speckinds)
                    if freq == window_freq: joinfreq = freq
                    else: joinfreq = freq + '_' + window_freq
                    alms[hm+hrsplit, joinfreq, kind] = sph_tools.get_alms(pl_map, window_tuple, niter, lmax)#, use_pol=use_pol_alm)
                    del pl_map, window_tuple
                    print('Alms Done')
                    
joinfreqs = cmbfreqs
for cleanfreq in cleanfreqs:
    joinfreqs += [cleanfreq + '_' + x for x in cmbfreqs] # [545_143, 545_217]
for kind in speckinds:
    for c1, freq1 in enumerate(joinfreqs):
        for c2, freq2 in enumerate(joinfreqs):
            print(freq1, freq2)
            if '_' in freq1:
                mapfreq1 = freq1[:3]
                maskfreq1 = freq1[-3:]
            else:
                mapfreq1, maskfreq1 = freq1, freq1
            if '_' in freq2:
                mapfreq2 = freq2[:3]
                maskfreq2 = freq2[-3:]
            else:
                mapfreq2, maskfreq2 = freq2, freq2
            #if c1 > c2: continue
            if int(maskfreq1) > int(maskfreq2): continue # Skip 217x545(143) that is not in mcms and similar
            if (int(maskfreq2) <= int(maskfreq1)) and ((mapfreq1 in cleanfreqs) and (mapfreq2 not in cleanfreqs)): continue # Skip 545x143 since you've already done 143x545
            for s1, hm1 in enumerate(splits):
                for s2, hm2 in enumerate(splits):
                    for hri1, hr1 in enumerate(halfringsplits):
                        for hri2, hr2 in enumerate(halfringsplits):
                            if (s1 > s2) & (c1 == c2): continue
                            if (s1 > s2) & (maskfreq1 == maskfreq2): # swap splits in mask if same mask frequency (since AB=BA but only AB is saved)
                                prefix= "%s/%s_%sx%s_%s-%sx%s" % (mcms_dir, experiment, maskfreq1, experiment, maskfreq2, hm2, hm1) # for reading mcms
                            else:
                                prefix= "%s/%s_%sx%s_%s-%sx%s" % (mcms_dir, experiment, maskfreq1, experiment, maskfreq2, hm1, hm2) # for reading mcms
                            l, ps = so_spectra.get_spectra(alms[hm1+hr1,freq1, kind], alms[hm2+hr2,freq2, kind], spectra=spectra)
                            if spectra is None:
                                ps = {'TT': ps} # convert to dict to be the same as the pol case
                            spec_name = "%s_%sx%s_%s-%sx%s" % (experiment, freq1, experiment, freq2, hm1+hr1, hm2+hr2) # for writing spectra

                            if beam_template is not None:
                                splittag1, splittag2 = splittags[s1], splittags[s2]
                                bls = [planck_utils.load_beam_planck_wl(beam_template%(mapfreq1, splittag1, mapfreq2, splittag2), spec) for spec in beamspectra]
                                if pixwin == True:
                                    pw = hp.pixwin(2048)
                                    maxlen = min(len(bls[0]), len(pw))
                                    bls = [bl[:maxlen] * pw[:maxlen] for bl in bls]
                                wls = [bl[:lmax]**2 for bl in bls]
                                for wl in wls:
                                    if len(wl) != lmax:
                                        raise ValueError("len(wl)=%s. Should be lmax=%s" % (len(wl), lmax))
                            else:
                                 wls = None
                                 if pixwin == True:
                                     raise NotImplementedError("Cannot deconvolve pixwin without beam")

                            if do_decouple:
                                mcm_inv, Bbl = so_mcm.read_coupling(prefix=prefix, spin_pairs=spin_pairs)
                            else:
                                mcm_inv = None


                            for cltype in types:
                                newl, cl, lb, Db = planck_utils.process_planck_spectra(l,
                                                                       ps,
                                                                       binning_file,
                                                                       lmax,
                                                                       mcm_inv=mcm_inv,
                                                                       spectra=beamspectra,
                                                                       cltype=cltype,
                                                                       wls=wls)

                                so_spectra.write_ps("%s/spectra%s_%s_%s.dat" % (spectra_dir, kind, cltype, spec_name), lb ,Db, type=cltype, spectra=beamspectra)
                            so_spectra.write_ps("%s/spectra%s_unbin_%s.dat" % (spectra_dir, kind, spec_name), newl, cl, type=cltype, spectra=beamspectra)


