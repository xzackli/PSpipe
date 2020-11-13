"""
This script computes the mode coupling matrices and the binning matrices Bbl
for the different surveys and arrays
"""

from pspy import so_map, so_mcm, pspy_utils, so_dict
import sys
import namaster_tools as nmt_tools

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

mcm_dir = "mcms"
pspy_utils.create_directory(mcm_dir)

freqs   = d["freqs"]
niter   = d["niter"]
type    = d["type"]
lmax = d["lmax"]
regions = d["regions"]
splits = d["splits"]

print("Computing mode coupling matrices:")
for reg in regions:
    for f1, freq1 in enumerate(freqs):
        for c1, split1 in enumerate(splits):
            mask1_t = so_map.read_map(d["planck_mask_car_%s_%s_%s_T" % (freq1, split1, reg)])
            mask1_pol = so_map.read_map(d["planck_mask_car_%s_%s_%s_P" % (freq1, split1, reg)])
            for f2, freq2 in enumerate(freqs):
                if f1 > f2: continue
                for c2, split2 in enumerate(splits):
                    if f1==f2 and c1>c2: continue
                    mask2_t = so_map.read_map(d["planck_mask_car_%s_%s_%s_T" % (freq2, split2, reg)])
                    mask2_pol = so_map.read_map(d["planck_mask_car_%s_%s_%s_P" % (freq2, split2, reg)])

                    bl1_t = nmt_tools.load_beam_planck_wl(d["planck_beam_template"]%(freq1,split1,freq2,split2),'TT')
                    bl2_t = nmt_tools.load_beam_planck_wl(d["planck_beam_template"]%(freq1,split1,freq2,split2),'TT')                
                    bl1_pol = nmt_tools.load_beam_planck_wl(d["planck_beam_template"]%(freq1,split1,freq2,split2),'EE')
                    bl2_pol = nmt_tools.load_beam_planck_wl(d["planck_beam_template"]%(freq1,split1,freq2,split2),'EE')

                    mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0and2(win1=(mask1_t, mask1_pol),
                                                                    win2=(mask2_t, mask2_pol),
                                                                    bl1=(bl1_t, bl1_pol),
                                                                    bl2=(bl2_t, bl2_pol),
                                                                    binning_file=d["binning_file"],
                                                                    niter=niter,
                                                                    lmax=lmax,
                                                                    type=type,
                                                                    l_exact=None,
                                                                    l_band=None,
                                                                    l_toep=None,
                                                                    save_file="%s/planck_%s_%s_%sx%s_%s"%(mcm_dir, reg, freq1, split1, freq2, split2))



