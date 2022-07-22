'''
This script is used to compute the mode coupling matrices of the Planck data.
The inputs for the script are the Planck beam and likelihood masks.
To run it:
python get_planck_mcm_Bbl.py global.dict
'''
import numpy as np
from pspy import so_dict, so_map, so_mcm, pspy_utils
import sys
from glob import glob
d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

windows_dir = "windows"
mcm_dir = "mcms"

pspy_utils.create_directory(windows_dir)
pspy_utils.create_directory(mcm_dir)

freqs = d["cmb_freqs"]
niter = d["niter"]
lmax = d["lmax"]
type = d["type"]
binning_file = d["binning_file"]
pixwin = d["pixwin"]
splits = d["splits"]
use_pol = d["use_pol"]
experiment = "Planck"

print("Compute Planck 2018 mode coupling matrices")
if use_pol:
    suffix = "_mode_coupling_inv_spin2xspin2.npy"
else:
    suffix = "_mode_coupling_inv.npy"
doneset = set(glob(mcm_dir+'/*'+suffix))

for f1, freq1 in enumerate(freqs):
    for count1, hm1 in enumerate(splits):
        window_1 = d["window_%s_%s" % (freq1, hm1)]
        # hp field 0 = T, field 1 = P
        win_t1 = so_map.read_map(window_1, fields_healpix=0)
        window_tuple1 = win_t1
        if use_pol:
            win_pol1 = so_map.read_map(window_1, fields_healpix=1)
            window_tuple1 = (win_t1, win_pol1)
            del win_pol1
        del win_t1        
        # win_t1.write_map("%s/window_T_%s_%s-%s.fits" % (windows_dir, experiment, freq1, hm1))
        # win_pol1.write_map("%s/window_P_%s_%s-%s.fits" % (windows_dir, experiment, freq1, hm1))
    
        for f2, freq2 in enumerate(freqs):
            if f1 > f2: continue
            for count2, hm2 in enumerate(splits):
                if (count1 > count2) & (f1 == f2): continue

                save_file="%s/%s_%sx%s_%s-%sx%s" % (mcm_dir, experiment, freq1, experiment, freq2, hm1, hm2)
                if save_file+suffix in doneset:
                    print('skipping %s' % save_file)
                    continue
                print(freq1+hm1, freq2+hm2)
                window_2 = d["window_%s_%s" % (freq2, hm2)]
                win_t2 = so_map.read_map(window_2, fields_healpix=0)
                window_tuple2 = win_t2
                if use_pol:                
                    win_pol2 = so_map.read_map(window_2, fields_healpix=1)
                    window_tuple2 = (win_t2, win_pol2)
                    del win_pol2
                del win_t2
                if use_pol:
                    mcm_and_bbl = so_mcm.mcm_and_bbl_spin0and2
                else:
                    mcm_and_bbl = so_mcm.mcm_and_bbl_spin0
                mcm_inv, Bbl = mcm_and_bbl(win1=window_tuple1,
                                           win2=window_tuple2,
                                           binning_file=binning_file,
                                           bl1=None,
                                           bl2=None,
                                           lmax=lmax,
                                           niter=niter,
                                           type=type,
                                           binned_mcm=False,
                                           save_file="%s/%s_%sx%s_%s-%sx%s" % (mcm_dir, experiment, freq1, experiment, freq2, hm1, hm2))




