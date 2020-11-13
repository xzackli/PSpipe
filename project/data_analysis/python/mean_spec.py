from pspy import pspy_utils, so_dict, so_spectra, so_cov
import numpy as np
import sys
import data_analysis_utils

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

type = d["type"]
surveys = d["surveys"]
lmax = d["lmax"]

bestfit_dir = "best_fits"
cov_dir = "covariances"
specDir = "spectra"
mcm_dir = "mcms"

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
n_spectra = data_analysis_utils.get_nspec(d)

kind = "cross"
for spec in ["TT", "TE", "ET", "EE"]:
    if spec != "ET":
        specs150 = []
        stds150 = []
        specs90 = []
        stds90 = []
    specsx = []
    stdsx = []

    for id_sv1, sv1 in enumerate(surveys):
        arrays_1 = d["arrays_%s" % sv1]
        for id_ar1, ar1 in enumerate(arrays_1):
            for id_sv2, sv2 in enumerate(surveys):
                arrays_2 = d["arrays_%s" % sv2]
                for id_ar2, ar2 in enumerate(arrays_2):

                    if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                    if  (id_sv1 > id_sv2) : continue

                    combin = "%s_%sx%s_%s" % (sv1, ar1, sv2, ar2)
                    spec_name = "%s_%s_%s" % (type, combin, kind)

                    lb, Db = so_spectra.read_ps("%s/%s.dat" % (specDir, spec_name), spectra=spectra)

                    if ar1 == "pa3_f090" and ar2 != "pa3_f090":
                        use_spec = spec[::-1] # Swap TE and ET for this case so that TE is always 150x90
                    else:
                        use_spec = spec
                        
                    cov = np.load("%s/analytic_cov_%s_%s.npy"%(cov_dir, combin, combin))
                    cov = so_cov.selectblock(cov,
                                            ["TT", "TE", "ET", "EE"],
                                            n_bins = len(lb),
                                            block=use_spec+use_spec)

                    std = np.sqrt(cov.diagonal())
                    #print(spec_name)
                    if ar1 == 'pa3_f090' and ar2=='pa3_f090':
                        specs90.append(Db[use_spec])
                        stds90.append(std)
                    elif ar1=='pa3_f090' or ar2=='pa3_f090':
                        specsx.append(Db[use_spec])
                        stdsx.append(std)
                    else:
                        specs150.append(Db[use_spec])
                        stds150.append(std)
    specs = [np.array(sp) for sp in [specs150, specsx, specs90]]
    stds = [np.array(sp) for sp in [stds150, stdsx, stds90]]
    for sp, sd, freq in zip(specs,stds,['150','150x90','90']):
        newspec = np.average(sp, weights=sd**-2, axis=0)
        newstds = (np.sum(sd**-2,axis=0))**-0.5
        out=np.transpose([lb,newspec,newstds])
        if not (freq != '150x90' and (spec == 'TE' or spec == 'ET')):
            np.savetxt('%s/meanspec_%s_%s_%s' % (specDir, type, spec, freq), out)
        if (freq != '150x90' and spec == 'ET'):
            np.savetxt('%s/meanspec_%s_TE_%s' % (specDir, type, freq), out) # Call it TE if not x-spectra

    
