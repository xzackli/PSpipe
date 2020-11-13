## Stealing code from ACT DR4 notebook / nawrapper to do the preprocessing of the maps (pixwin, kfilt)
import sys
from pixell import enmap
import nawrapper as nw
from pspy import so_dict, so_map_preprocessing, so_map
import numpy as np

def kfilt_unpixwin(d):
    surveys = d["surveys"]
    vk_mask = d["vk_mask"]
    hk_mask = d["hk_mask"]
    unpixwin = d["remove_act_pixwin_preprocess"]
    act_kfilt = d["act_kfilt"]
    binning_file = d["binning_file"]
    lmax = d["lmax"]
    
    if act_kfilt:
        assert vk_mask[0] == -vk_mask[1]
        assert hk_mask[0] == -hk_mask[1]
        kx_cut = vk_mask[1]
        ky_cut = hk_mask[1]
    else:
        kx_cut, ky_cut = -1, -1
    for svi, sv in enumerate(surveys):
        tfname = d['tf_%s' % sv]
        print('survey %s / %s' % (svi+1, len(surveys)))
        arrays = d["arrays_%s" % sv]
        footprint = enmap.read_map(d["footprint_%s"%sv])
        shape, wcs = footprint.shape, footprint.wcs
        del footprint
        for ari, ar in enumerate(arrays):
            srcfree_maps = d["srcfree_maps_%s_%s" % (sv, ar)]
            src_maps = d["src_maps_%s_%s" % (sv, ar)]
            maps = d["maps_%s_%s" % (sv, ar)]
            for index, tup in enumerate(zip(srcfree_maps, src_maps)):
                srcfree, src = tup
                mm = enmap.read_map(srcfree)
                mm = enmap.extract(mm, shape, wcs)
                for ii in range(mm.shape[0]):
                    mm[ii] = nw.preprocess_fourier(mm[ii], kx_cut=kx_cut, ky_cut=ky_cut, unpixwin=unpixwin)
                src = enmap.read_map(src)
                src = enmap.extract(src, shape, wcs)
                outmap = mm + src
                if act_kfilt and ari==0:
                    som = so_map.from_enmap(outmap)
                    lb, tf = so_map_preprocessing.analytical_tf(som, binning_file, lmax, vk_mask=vk_mask, hk_mask=hk_mask)
                    savearr = np.array([lb,tf,tf,np.zeros_like(tf)]) # Put in same format as other transfer function
                    savearr = savearr.transpose()
                    np.savetxt(tfname, savearr)
                del mm, src
                print("writing map %s / %s" % (index+1 + ari*len(srcfree_maps), len(srcfree_maps)*len(arrays)))
                enmap.write_map(maps[index], outmap)
                del outmap

def make_window(d):
    surveys = d["surveys"]
    for svi, sv in enumerate(surveys):
        print('survey %s / %s' % (svi+1, len(surveys)))        
        footprint = d["footprint_%s" % sv]
        footprint = enmap.read_map(footprint)
        shape,wcs = footprint.shape, footprint.wcs    
        arrays = d["arrays_%s" % sv]
        for index, ar in enumerate(arrays):
            ivar_file = d["ivar_maps_%s_%s" % (sv, ar)]
            if ar[-3:] == "150":
                psmask_file = d["psmask_%s_150" % sv]
                apod = d["apod_150"]
            elif ar[-3:] == "090":
                psmask_file = d["psmask_%s_090" % sv]
                apod = d["apod_090"]
            else:
                raise ValueError("Array name should end in 150 or 090, not %s " % ar[-3:])

            # read in the point source mask, make sure it has the correct shape, and apodize
            psmask = enmap.extract(enmap.read_map(psmask_file), shape, wcs)
            psmask = nw.apod_C2(psmask, apod)
            # read in the coadd inverse variance map and make sure it has the correct shape
            ivar = enmap.extract(enmap.read_map(ivar_file), shape, wcs)
            mask = footprint*psmask*ivar
            outname = d["window_T_%s_%s" % (sv, ar)]
            print("writing window %s / %s" % (index+1, len(arrays)))
            enmap.write_map(outname, mask)

if __name__ == '__main__':
    d = so_dict.so_dict()
    d.read_from_file(sys.argv[1])
    print("Doing k-filter and unpixwin")
    kfilt_unpixwin(d)
    # print("making window functions")
    # make_window(d)
