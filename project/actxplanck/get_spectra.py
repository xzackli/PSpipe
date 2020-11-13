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
regions = d["regions"]
surveys = d["surveys"]
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
        win_T = so_map.read_map(d["planck_mask_car_%s_%s_T" % (freq, reg)])
        win_pol = so_map.read_map(d["planck_mask_car_%s_%s_P" % (freq, reg)])
        window_tuple = (win_T, win_pol)
        del win_T, win_pol

        map = so_map.read_map(d["planck_car_%s_%s" % (freq, reg)])
        if d["remove_mean"] == True:
            map = data_analysis_utils.remove_mean(map, window_tuple, ncomp)
        master_alms[reg, freq] = sph_tools.get_alms(map, window_tuple, niter, lmax)

for sv in surveys:
    arrays = d["arrays_%s" % sv]
    for ar in arrays:
        win_T = so_map.read_map(d["act_window_T_%s_%s" % (sv, ar)])
        win_pol = so_map.read_map(d["act_window_pol_%s_%s" % (sv, ar)])
        window_tuple = (win_T, win_pol)
        
        map = d["maps_%s_%s" % (sv, ar)][0] # No splits

        cal = d["cal_%s_%s" % (sv, ar)]

        if win_T.pixel == "CAR":
            split = so_map.read_map(map, geometry=win_T.data.geometry)
            if d["use_kspace_filter"]:
                binary = so_map.read_map("%s/binary_%s_%s.fits" % (window_dir, sv, ar))
                split = data_analysis_utils.get_filtered_map(split,
                                                             binary,
                                                             vk_mask=d["vk_mask"],
                                                             hk_mask=d["hk_mask"])

        elif win_T.pixel == "HEALPIX":
            raise NotImplementedError("Not tested: You shouldn't be using healpix maps here")
            split = so_map.read_map(map)

        split.data *= cal
        if d["remove_mean"] == True:
            split = data_analysis_utils.remove_mean(split, window_tuple, ncomp)

        #split.plot(file_name="%s/split_%d_%s_%s" % (plot_dir, k, sv, ar), color_range=[250, 100, 100])

        master_alms[sv, ar] = sph_tools.get_alms(split, window_tuple, niter, lmax)


ps_dict = {}
_, _, lb, _ = pspy_utils.read_binning_file(binning_file, lmax)


for reg in regions:
    if planck_kfilt:
        tf_planck = d["tf_planck_%s" % reg]
        if tf_planck is None: raise ValueError("tf_planck_%s is None but planck_kfilt is True" % reg)
        _, _, tf1, _ = np.loadtxt(tf_planck, unpack=True)
    else:
        tf1 = np.ones(len(lb))
    for freq in freqs:
    
        for id_sv2, sv2 in enumerate(surveys):
            arrays_2 = d["arrays_%s" % sv2]
            
            if d["tf_%s" % sv2] is not None:
                _, _, tf2, _ = np.loadtxt(d["tf_%s" % sv2], unpack=True)
            else:
                tf2 = np.ones(len(lb))

            for id_ar2, ar2 in enumerate(arrays_2):
                mbb_inv, Bbl = so_mcm.read_coupling(prefix="%s/planck_%s_%sx%s_%s" % (mcm_dir, reg, freq, sv2, ar2),
                                                    spin_pairs=spin_pairs)

                l, ps_master = so_spectra.get_spectra_pixell(master_alms[reg, freq],
                                                             master_alms[sv2, ar2],
                                                             spectra=spectra)

                spec_name="%s_%s_%sx%s_%s" % (type, reg, freq, sv2, ar2)
                lb, ps = so_spectra.bin_spectra(l,
                                                ps_master,
                                                binning_file,
                                                lmax,
                                                type=type,
                                                mbb_inv=mbb_inv,
                                                spectra=spectra)
                data_analysis_utils.deconvolve_tf(lb, ps, tf1, tf2, ncomp, lmax)
                so_spectra.write_ps(specDir + "/%s.dat" % spec_name, lb, ps, type, spectra=spectra)

