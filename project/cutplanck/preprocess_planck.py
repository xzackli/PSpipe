import numpy as np
import sys
from pixell import enmap, reproject
import nawrapper as nw
from pspy import so_dict
import healpy as hp

def transform_planck(d):
    freqs = d['freqs']
    regions = d["regions"]
    kfilt = d["planck_kfilt"]
    pixwin = d["remove_planck_pixwin_preprocess"]
    
    splits = d["splits"]
    unit = 1e-6    
    for freq in freqs:
        for split in splits:
            mapf_hp = d["planck_hp_%s_%s" % (freq, split)]
            for sv in regions:
                footprint = d["footprint_%s" % sv]
                footprint = enmap.read_map(footprint)
                shape,wcs = footprint.shape, footprint.wcs
                del footprint
                map_hp = hp.read_map(mapf_hp, field=(0,1,2)) ## Warning will not work with 545

                covmap = hp.read_map(mapf_hp, field=4)
                misspix = (covmap == 0)
                del covmap
                misspix = np.ones((3,misspix.size),dtype=bool) * misspix
                map_hp[misspix] = 0 # Null the missing pixels since they can't be perfectly masked after remapping
                del misspix

                map_car = reproject.enmap_from_healpix(map_hp, shape, wcs, ncomp=3, unit=unit, lmax=6000, rot='gal,equ')
                del map_hp
                if kfilt or pixwin:
                    if kfilt:
                        kx_cut, ky_cut = 90, 50 # default values
                    else:
                        kx_cut, ky_cut = 0, 0
                    for ii in range(map_car.shape[0]):
                        map_car[ii] = nw.preprocess_fourier(map_car[ii], unpixwin=pixwin, kx_cut=kx_cut, ky_cut=ky_cut)
                outname = d["planck_car_%s_%s_%s" % (freq, split, sv)]
                enmap.write_map(outname, map_car)

def transform_planck_masks(d):
    freqs = d['freqs']
    regions = d["regions"]
    splits = d["splits"]
    unit = 1
    kinds = ['T', 'P']
    for kind in kinds:
        for freq in freqs:
            for split in splits:
                mapf_hp = d["planck_mask_hp_%s_%s" % (freq, split)]
                for sv in regions:
                    footprint = d["footprint_%s" % sv]
                    footprint = enmap.read_map(footprint)
                    shape,wcs = footprint.shape, footprint.wcs
                    if kind == 'T': field=0
                    if kind == 'P': field=1
                    map_hp = hp.read_map(mapf_hp, field=field) ## Warning will not work with 545
                    #map_hp = hp.read_map(mapf_hp)
                    map_car = reproject.enmap_from_healpix_interp(map_hp, shape, wcs, rot='gal,equ', interpolate=True)
                    del map_hp
                    map_car *= footprint
                    outname = d["planck_mask_car_%s_%s_%s_%s" % (freq, split, sv, kind)]
                    enmap.write_map(outname, map_car)

def make_planck_mask(d):
    for freq in d['freqs']:
        for split in d['splits']:
            psmask_T_hp = d['planck_psmask_hp_T_%s' % freq]
            psmask_pol_hp = d['planck_psmask_hp_pol_%s' % freq]
            galmask_T_hp = d['planck_galmask_hp_T_%s' % freq]
            galmask_pol_hp = d['planck_galmask_hp_pol_%s' % freq]
            mapf_hp = d["planck_hp_%s_%s" % (freq, split)]
            covmap = hp.read_map(mapf_hp, field=4)
            misspix = (covmap == 0)
            del covmap
            maskT = hp.read_map(psmask_T_hp) * hp.read_map(galmask_T_hp) * misspix
            maskP = hp.read_map(psmask_pol_hp) * hp.read_map(galmask_pol_hp) * misspix
            hp.write_map(d["planck_mask_hp_%s_%s" % (freq, split)], [maskT, maskP, maskP])


if __name__ == '__main__':
    d = so_dict.so_dict()
    d.read_from_file(sys.argv[1])
    # print("Making Planck CAR maps")
    # transform_planck(d)
    # print("Making Planck Masks")
    # make_planck_mask(d)
    #print("Making Planck CAR masks")
    transform_planck_masks(d)

