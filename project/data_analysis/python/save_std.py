from pspy import so_dict, so_spectra, so_cov
import numpy as np
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

surveys = d["surveys"]
lmax = d["lmax"]
type = d["type"]
cov_dir = "covariances"
specDir = "spectra"

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
for id_sv1, sv1 in enumerate(surveys):
    arrays_1 = d["arrays_%s" % sv1]
    for id_ar1, ar1 in enumerate(arrays_1):
        for id_sv2, sv2 in enumerate(surveys):
            arrays_2 = d["arrays_%s" % sv2]
            for id_ar2, ar2 in enumerate(arrays_2):

                if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                if  (id_sv1 > id_sv2) : continue

                combin = "%s_%sx%s_%s" % (sv1, ar1, sv2, ar2)
                cov = np.load("%s/analytic_cov_%s_%s.npy"%(cov_dir, combin, combin))
                spec_name = "%s_%s_%s" % (type, combin, "cross")
                lb, _ = so_spectra.read_ps("%s/%s.dat" % (specDir, spec_name), spectra=spectra)
                stds = []
                for spec in ["TT", "TE", "ET", "EE"]:
                    sub_cov = so_cov.selectblock(cov,
                                            ["TT", "TE", "ET", "EE"],
                                            n_bins = len(lb),
                                            block=spec+spec)
                    std = np.sqrt(sub_cov.diagonal())
                    stds.append(std)
                stds = np.array(stds)
                stds=stds.transpose()
                np.savetxt('%s/analytic_errors_%s.txt' % (specDir, combin), stds, header="TT, TE, ET, EE")

