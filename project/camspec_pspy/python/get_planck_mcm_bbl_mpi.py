'''
This script is used to compute the mode coupling matrices of the Planck data.
The inputs for the script are the Planck beam and likelihood masks.
To run it:
python get_planck_mcm_Bbl.py global.dict
'''
import numpy as np
from pspy import so_dict, so_map, so_mcm, pspy_utils, so_mpi
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

windows_dir = "windows"
mcm_dir = "mcms"

pspy_utils.create_directory(windows_dir)
pspy_utils.create_directory(mcm_dir)

freqs = d["freqs"]
niter = d["niter"]
lmax = d["lmax"]
type = d["type"]
binning_file = d["binning_file"]
pixwin = d["pixwin"]
splits = d["splits"]
use_pol = d["use_pol"]
experiment = "Planck"

print("Compute Planck 2018 mode coupling matrices")

freq1_list, hm1_list, freq2_list, hm2_list = [], [], [], []
n_mcms = 0
for f1, freq1 in enumerate(freqs):
    for count1, hm1 in enumerate(splits):
        for f2, freq2 in enumerate(freqs):
            if f1 > f2: continue
            for count2, hm2 in enumerate(splits):
                if (count1 > count2) & (f1 == f2): continue
                
                freq1_list += [freq1]
                freq2_list += [freq2]
                hm1_list += [hm1]
                hm2_list += [hm2]
                n_mcms += 1
                
print("number of mcm matrices to compute : %s" % n_mcms)
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=n_mcms - 1)
print(subtasks)
for task in subtasks:
    task = int(task)
    freq1, hm1, freq2, hm2 = freq1_list[task], hm1_list[task], freq2_list[task], hm2_list[task]
    
    window_1 = d["window_%s_%s" % (freq1, hm1)]
    win_t1 = so_map.read_map(window_1, fields_healpix=0)
    window_tuple1 = win_t1
    if use_pol:
        win_pol1 = so_map.read_map(window_1, fields_healpix=1)
        window_tuple1 = (win_t1, win_pol1)
        del win_pol1
    del win_t1
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
    mcm_inv, mbb_inv, Bbl = mcm_and_bbl(win1=window_tuple1,
                                        win2=window_tuple2,
                                        binning_file=binning_file,
                                        bl1=None,
                                        bl2=None,
                                        lmax=lmax,
                                        niter=niter,
                                        type=type,
                                        binned_mcm=False,
                                        save_file="%s/%s_%sx%s_%s-%sx%s" % (mcm_dir, experiment, freq1, experiment, freq2, hm1, hm2))




