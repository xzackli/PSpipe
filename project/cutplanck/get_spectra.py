"""
This script compute all power spectra and write them to disk.
"""
from pspy import pspy_utils, so_dict, so_map, sph_tools, so_mcm, so_spectra
import numpy as np
import sys
import data_analysis_utils

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

freqs = d["freqs"]
splits = d["splits"]
regions = d["regions"]
lmax = d["lmax"]
niter = d["niter"]
type = d["type"]
binning_file = d["binning_file"]
write_all_spectra = d["write_splits_spectra"]
planck_kfilt = d["planck_kfilt"]
window_dir = "windows"
mcm_dir = "mcms"
specDir = "spectra"
plot_dir = "plots/maps/"

pspy_utils.create_directory(plot_dir)
pspy_utils.create_directory(specDir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]

ncomp = 3

master_alms = {}
nsplit = {}
for reg in regions:
    for freq in freqs:
        for split in splits:
            win_T = so_map.read_map(d["planck_mask_car_%s_%s_%s_T" % (freq, split, reg)])
            win_pol = so_map.read_map(d["planck_mask_car_%s_%s_%s_P" % (freq, split, reg)])
            window_tuple = (win_T, win_pol)
            del win_T, win_pol

            map = so_map.read_map(d["planck_car_%s_%s_%s" % (freq, split, reg)])
            if d["remove_mean"] == True:
                map = data_analysis_utils.remove_mean(map, window_tuple, ncomp)
            master_alms[reg, freq, split] = sph_tools.get_alms(map, window_tuple, niter, lmax)

ps_dict = {}
_, _, lb, _ = pspy_utils.read_binning_file(binning_file, lmax)

for reg in regions:
    if planck_kfilt:
        tf_planck = d["tf_planck_%s" % reg]
        if tf_planck is None: raise ValueError("tf_planck_%s is None but planck_kfilt is True" % reg)
         _, _, tf1, _ = np.loadtxt(tf_planck, unpack=True)
    else:
        tf1 = np.ones(len(lb))
    tf2 = tf1
    for f1, freq1 in enumerate(freqs):
        for c1, split1 in enumerate(splits):
            for f2, freq2 in enumerate(freqs):
                if f1 > f2: continue
                for c2, split2 in enumerate(splits):
                    if (f1 == f2) and c1 > c2: continue
                    mbb_inv, Bbl = so_mcm.read_coupling(prefix="%s/planck_%s_%s_%sx%s_%s" % (mcm_dir, reg, freq1, split1, freq2, split2),
                                                    spin_pairs=spin_pairs)

                    l, ps_master = so_spectra.get_spectra_pixell(master_alms[reg, freq1, split1],
                                                                 master_alms[reg, freq2, split2],
                                                                 spectra=spectra)

                    spec_name="%s_%s_%s_%sx%s_%s" % (type, reg, freq1, split1, freq2, split2)
                    lb, ps = so_spectra.bin_spectra(l,
                                                    ps_master,
                                                    binning_file,
                                                    lmax,
                                                    type=type,
                                                    mbb_inv=mbb_inv,
                                                    spectra=spectra)

                    data_analysis_utils.deconvolve_tf(lb, ps, tf1, tf2, ncomp, lmax)
                    so_spectra.write_ps(specDir + "/%s.dat" % spec_name, lb, ps, type, spectra=spectra)
