"""
This script computes the mode coupling matrices and the binning matrices Bbl
for the different surveys and arrays
"""

from pspy import so_map, so_mcm, pspy_utils, so_dict
import sys
import namaster_tools as nmt_tools
import healpy as hp

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

mcm_dir = "mcms"
pspy_utils.create_directory(mcm_dir)

freqs   = d["freqs"]
niter   = d["niter"]
type    = d["type"]
lmax = d["lmax"]
regions = d["regions"]
surveys = d["surveys"]

print("Computing mode coupling matrices:")
for reg in regions:
    for freq in freqs:
        mask1_t = so_map.read_map(d["planck_mask_car_%s_%s_T" % (freq, reg)])
        mask1_pol = so_map.read_map(d["planck_mask_car_%s_%s_P" % (freq, reg)])
        bl1_t = nmt_tools.load_beam_planck_wl(d["planck_beam_%s" % freq],'TT')
        bl1_pol = nmt_tools.load_beam_planck_wl(d["planck_beam_%s" % freq], 'EE')

        for id_sv2, sv2 in enumerate(surveys):
            arrays_2 = d["arrays_%s" % sv2]
            for id_ar2, ar2 in enumerate(arrays_2):
                l, bl2 = pspy_utils.read_beam_file(d["act_beam_%s_%s" % (sv2, ar2)])
                win2_T = so_map.read_map(d["act_window_T_%s_%s" % (sv2, ar2)])
                win2_pol = so_map.read_map(d["act_window_pol_%s_%s" % (sv2, ar2)])
                print("Planck_%s x %s_%s" % (freq, sv2, ar2))
                mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0and2(win1=(mask1_t, mask1_pol),
                                                                win2=(win2_T, win2_pol),
                                                                bl1=(bl1_t, bl1_pol),
                                                                bl2=(bl2, bl2),
                                                                binning_file=d["binning_file"],
                                                                niter=niter,
                                                                lmax=lmax,
                                                                type=type,
                                                                l_exact=None,
                                                                l_band=None,
                                                                l_toep=None,
                                                                save_file="%s/planck_%s_%sx%s_%s"%(mcm_dir, reg, freq, sv2, ar2))



