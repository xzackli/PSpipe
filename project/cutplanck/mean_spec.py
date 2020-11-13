from pspy import pspy_utils, so_dict, so_spectra, so_cov
import numpy as np
import sys
import data_analysis_utils

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

type = d["type"]
surveys = d["surveys"]
regions = d["regions"]
freqs = d["freqs"]
lmax = d["lmax"]

bestfit_dir = "best_fits"
cov_dir = "covariances"
specDir = "spectra"
mcm_dir = "mcms"

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
n_spectra = data_analysis_utils.get_nspec(d)

for spec in ["TT", "TE", "ET", "EE"]:
    s100x150 = []
    s143x150 = []
    s217x150 = []
    s100x090 = []
    s143x090 = []
    s217x090 = []

    for reg in regions:
        for freq in freqs:
            for id_sv2, sv2 in enumerate(surveys):
                arrays_2 = d["arrays_%s" % sv2]
                for id_ar2, ar2 in enumerate(arrays_2):

                    combin = "%s_%sx%s_%s" % (reg, freq, sv2, ar2)
                    spec_name = "%s_%s" % (type, combin)

                    lb, Db = so_spectra.read_ps("%s/%s.dat" % (specDir, spec_name), spectra=spectra)

                    #print(spec_name)
                    if ar2 != 'pa3_f090':
                        if freq == '100':
                            s100x150.append(Db[spec])
                        elif freq == '143':
                            s143x150.append(Db[spec])
                        elif freq == '217':
                            s217x150.append(Db[spec])
                    else:
                        if freq == '100':
                            s100x090.append(Db[spec])
                        elif freq == '143':
                            s143x090.append(Db[spec])
                        elif freq == '217':
                            s217x090.append(Db[spec])

    specs = [np.array(sp) for sp in [s100x150, s143x150, s217x150, s100x090, s143x090, s217x090]]
    for sp, freq in zip(specs,['100x150','143x150','217x150', '100x090', '143x090', '217x090']):
        newspec = np.mean(sp, axis=0)
        out=np.transpose([lb,newspec])
        np.savetxt('%s/meanspec_%s_%s' % (specDir, spec, freq), out)

    
