"""
This script analyze the simulations generated by mc_get_spectra.py
it estimates the mean and numerical covariances from the simulations
"""


from pspy import pspy_utils, so_dict, so_spectra
import numpy as np
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

type = d["type"]
surveys = d["surveys"]
iStart = d["iStart"]
iStop = d["iStop"]
lmax = d["lmax"]


spec_dir = "sim_spectra_for_tf"
tf_dir = "transfer_functions"
bestfit_dir = "best_fits"

plot_dir = "plots/transfer_functions"

pspy_utils.create_directory(tf_dir)
pspy_utils.create_directory(plot_dir)


spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]
nsims = iStop - iStart



spec_list = []
for id_sv1, sv1 in enumerate(surveys):
    arrays_1 = d["arrays_%s" % sv1]
    for id_ar1, ar1 in enumerate(arrays_1):
        for id_sv2, sv2 in enumerate(surveys):
            arrays_2 = d["arrays_%s" % sv2]
            for id_ar2, ar2 in enumerate(arrays_2):
                if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                if  (id_sv1 > id_sv2) : continue
                spec_list += ["%s_%sx%s_%s" % (sv1, ar1, sv2, ar2)]


clfile = "%s/lcdm.dat" % bestfit_dir
lth, Dlth = pspy_utils.ps_lensed_theory_to_dict(clfile, output_type=type, lmax=lmax, start_at_zero=False)


for sid, spec in enumerate(spec_list):

    prefix= "%s/%s" % (mcm_dir, spec)

    mbb_inv, Bbl = so_mcm.read_coupling(prefix=prefix,spin_pairs=spin_pairs)

    # we will compare simulation power spectrum to theory
    # we need to add foreground in TT
    n1, n2 = spec.split("x")
    nu_eff_1 = d["nu_eff_%s" % (n1)]
    nu_eff_2 = d["nu_eff_%s" % (n2)]
    _, flth = np.loadtxt("%s/fg_%sx%s_TT.dat" %(bestfit_dir, nu_eff_1, nu_eff_2), unpack=True)
    Dlth["TT"]  = Dlth["TT"] + flth[:lmax]
    
    bin_theory = so_mcm.apply_Bbl(Bbl, Dlth, spectra=spectra)

    mean, std = {}, {}
    
    for spectrum in ["TT", "EE", "BB"]:
    
        nofilt_list = []
        filt_list = []
        tf_list = []
        
        for iii in range(iStart, iStop):
        
            spec_name_no_filter = "%s_%s_nofilter_%05d" % (type, spec, iii)
            spec_name_filter = "%s_%s_filter_%05d" % (type, spec, iii)

            lb, ps_nofilt = so_spectra.read_ps(spec_dir + "/%s.dat" % spec_name_no_filter, spectra=spectra)
            lb, ps_filt = so_spectra.read_ps(spec_dir + "/%s.dat" % spec_name_filter, spectra=spectra)
        
            nofilt_list += [ps_nofilt[spectrum]]
            filt_list += [ps_filt[spectrum]]
            tf_list += [ps_filt[spectrum]/ps_nofilt[spectrum]]
            
        mean[spectrum, "nofilt"] = np.mean(nofilt_list, axis = 0)
        mean[spectrum, "filt"] = np.mean(filt_list, axis = 0)
        mean[spectrum, "tf"] = np.mean(tf_list, axis = 0)

        std[spectrum, "nofilt"] = np.std(nofilt_list, axis = 0)
        std[spectrum, "filt"] = np.std(filt_list, axis = 0)
        std[spectrum, "tf"] = np.std(tf_list, axis = 0)

        # First let make sure that the spectrum without filter is unbiased
        if spectrum == "TT":
            plt.semilogy()
        
        plt.plot(lth, Dlth[spectrum], color="grey", alpha=0.4)
        plt.plot(lb, bin_theory[spectrum])
        plt.errorbar(lb, mean[spectrum, "nofilt"], std[spectrum, "nofilt"] , fmt=".", color="red")
        plt.errorbar(lb, mean[spectrum, "filt"], std[spectrum, "nofilt"] , fmt=".")
        plt.title(r"$D_{\ell}$", fontsize=20)
        plt.xlabel(r"$\ell$", fontsize=20)
        plt.savefig("%s/%s_%s.png" % (plot_dir, spec, spectrum), bbox_inches="tight")
        plt.clf()
        plt.close()
                
        plt.errorbar(lb, (mean[spectrum, "nofilt"] - bin_theory[spectrum])/ (std[spectrum, "nofilt"]  / np.sqrt(nsims)), fmt=".", color="red")
        plt.title(r"$\Delta D_{\ell}$" , fontsize=20)
        plt.xlabel(r"$\ell$", fontsize=20)
        plt.savefig("%s/frac_%s_%s.png" % (plot_dir, spec, spectrum), bbox_inches="tight")
        plt.clf()
        plt.close()
        
        # Then lets plot the transfer function

        plt.errorbar(lb, mean[spectrum, "tf"], std[spectrum, "tf"]/np.sqrt(nsims), fmt=".", color="red")
        plt.title(r"$\Delta D_{\ell}$" , fontsize=20)
        plt.xlabel(r"$\ell$", fontsize=20)
        plt.savefig("%s/frac_%s_%s.png" % (plot_dir, spec, spectrum), bbox_inches="tight")
        plt.clf()
        plt.close()



