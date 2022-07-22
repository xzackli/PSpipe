'''
Some utility functions for processing the planck data
'''

import numpy as np
import healpy as hp
from pspy import so_mcm, so_spectra, pspy_utils, so_map
from pixell import curvedsky
from astropy.io import fits

def bin_planck_spectra_coupled(cl, binning_file, lmax, spectra, toDl=False):
    bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
    n_bins = len(bin_hi)
    l = np.arange(2, lmax)
    if toDl:
        fac = (l * (l + 1) / (2 * np.pi))
    else:
        fac = np.ones_like(l)

    vec = []
    for spec in spectra:
        binnedPower = np.zeros(len(bin_c))
        for ibin in range(n_bins):
            loc = np.where((l >= bin_lo[ibin]) & (l <= bin_hi[ibin]))
            binnedPower[ibin] = (cl[spec][loc] * fac[loc]).mean()
        vec = np.append(vec, binnedPower)
    Db = so_spectra.vec2spec_dict(n_bins, vec, spectra)
    return bin_c, Db

def process_planck_spectra(l, cl, binning_file, lmax, spectra, mcm_inv, cltype, wls=None):
    bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
    n_bins = len(bin_hi)
    unbin_vec = []
    for spec in spectra:
        unbin_vec = np.append(unbin_vec, cl[spec][2:lmax])
    if mcm_inv is not None:
        if type(mcm_inv) is dict:
            mcm_inv = so_mcm.coupling_dict_to_array(mcm_inv)
        cl = so_spectra.vec2spec_dict(lmax-2, np.dot(mcm_inv, unbin_vec), spectra)
    else:
        cl = so_spectra.vec2spec_dict(lmax-2, unbin_vec, spectra)

    # Do beams here, after mask deconvolution
    if wls is not None:
        for ii in range(len(spectra)):
            cl[spectra[ii]] = cl[spectra[ii]] / wls[ii][2:lmax]
        # cl['TT'] = cl['TT'] / wls[0][2:lmax]
        # if 'TE' in cl.keys(): # polarization included
        #     cl['TE'] = cl['TE'] / wls[1][2:lmax]
        #     cl['ET'] = cl['ET'] / wls[2][2:lmax]
        #     cl['EE'] = cl['EE'] / wls[3][2:lmax]
        #     sz = cl['TT'].size
        #     zero_keys = ['TB', 'BT', 'EB', 'BE', 'BB']
        #     for kk in zero_keys:
        #         cl[kk] = np.zeros(sz)

    l = np.arange(2, lmax)
    if cltype == 'Dl':
        fac = (l * (l + 1) / (2 * np.pi))
    elif cltype == 'Cl':
        fac = np.ones_like(l)
    else:
        raise ValueError("'cltype' is %s. Can only be Cl or Dl" % cltype)

    vec = []
    for spec in spectra:
        binnedPower = np.zeros(len(bin_c))
        for ibin in range(n_bins):
            loc = np.where((l >= bin_lo[ibin]) & (l <= bin_hi[ibin]))
            binnedPower[ibin] = (cl[spec][loc] * fac[loc]).mean()#/(fac[loc].mean())
        vec = np.append(vec, binnedPower)
    Db = so_spectra.vec2spec_dict(n_bins, vec, spectra)
    return l, cl, bin_c, Db

def subtract_mono_di(map_in, mask_in, nside):
    
    map_masked = hp.ma(map_in)
    map_masked.mask = (mask_in < 1)
    mono, dipole = hp.pixelfunc.fit_dipole(map_masked)
    print(mono, dipole)
    m = map_in.copy()
    npix = hp.nside2npix(nside)
    bunchsize = npix // 24
    bad = hp.UNSEEN
    for ibunch in range(npix // bunchsize):
        ipix = np.arange(ibunch * bunchsize, (ibunch + 1) * bunchsize)
        ipix = ipix[(np.isfinite(m.flat[ipix]))]
        x, y, z = hp.pix2vec(nside, ipix, False)
        m.flat[ipix] -= dipole[0] * x
        m.flat[ipix] -= dipole[1] * y
        m.flat[ipix] -= dipole[2] * z
        m.flat[ipix] -= mono
    return m

def binning(l, cl, lmax, binning_file=None, size=None):
    
    if binning_file is not None:
        bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
    else:
        bin_lo = np.arange(2, lmax, size)
        bin_hi = bin_lo + size - 1
        bin_c = (bin_lo + bin_hi) / 2
    
    fac = (l * (l + 1) / (2 * np.pi))
    n_bins = len(bin_hi)
    binnedPower = np.zeros(len(bin_c))
    for ibin in range(n_bins):
        loc = np.where((l >= bin_lo[ibin]) & (l <= bin_hi[ibin]))
        binnedPower[ibin] = (cl[loc]*fac[loc]).mean()/(fac[loc].mean())
    return bin_c, binnedPower

def noise_matrix(noise_dir, exp, freqs, lmax, nsplits):
    
    """This function uses the noise power spectra computed by 'planck_noise_model'
    and generate a three dimensional array of noise power spectra [nfreqs,nfreqs,lmax] for temperature
    and polarisation.
    The different entries ([i,j,:]) of the arrays contain the noise power spectra
    for the different frequency channel pairs.
    for example nl_array_t[0,0,:] =>  nl^{TT}_{f_{0},f_{0}),  nl_array_t[0,1,:] =>  nl^{TT}_{f_{0},f_{1})
    this allows to have correlated noise between different frequency channels.
        
    Parameters
    ----------
    noise_data_dir : string
      the folder containing the noise power spectra
    exp : string
      the experiment to consider ('Planck')
    freqs: 1d array of string
      the frequencies we consider
    lmax: integer
      the maximum multipole for the noise power spectra
    n_splits: integer
      the number of data splits we want to simulate
      nl_per_split= nl * n_{splits}
    """    
    nfreqs = len(freqs)
    nl_array_t = np.zeros((nfreqs, nfreqs, lmax))
    nl_array_pol = np.zeros((nfreqs, nfreqs, lmax))

    for count, freq in enumerate(freqs):

        l, nl_t = np.loadtxt("%s/noise_TT_mean_%s_%sx%s_%s.dat"%(noise_dir, exp, freq, exp, freq), unpack=True)
        l, nl_pol = np.loadtxt("%s/noise_EE_mean_%s_%sx%s_%s.dat"%(noise_dir, exp, freq, exp, freq), unpack=True)

        nl_array_t[count, count, :] = nl_t[:] * nsplits
        nl_array_pol[count, count, :] = nl_pol[:] * nsplits
    
    return l, nl_array_t, nl_array_pol

def generate_noise_alms(nl_array_t, nl_array_pol, lmax, n_splits):
    
    """This function generates the alms corresponding to the noise power spectra matrices
    nl_array_t, nl_array_pol. The function returns a dictionnary nlms["T", i].
    The entry of the dictionnary are for example nlms["T", i] where i is the index of the split.
    note that nlms["T", i] is a (nfreqs, size(alm)) array, it is the harmonic transform of
    the noise realisation for the different frequencies.
        
    Parameters
    ----------
    nl_array_t : 3d array [nfreq, nfreq, lmax]
      noise power spectra matrix for temperature data
    nl_array_pol : 3d array [nfreq, nfreq, lmax]
      noise power spectra matrix for polarisation data

    lmax : integer
      the maximum multipole for the noise power spectra
    n_splits: integer
      the number of data splits we want to simulate

    """
    
    nlms = {}
    for k in range(n_splits):
        nlms["T", k] = curvedsky.rand_alm(nl_array_t, lmax=lmax)
        nlms["E", k] = curvedsky.rand_alm(nl_array_pol, lmax=lmax)
        nlms["B", k] = curvedsky.rand_alm(nl_array_pol, lmax=lmax)
    
    return nlms


def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())


def zero_misspix(map, mapname, flag, wcov_dictname, dct, external_wcov_override=None, split_index=None, ncomp=3):
    """ Apply missing pixels mask. Either use internal wcov from fits ext 4, or external wcov file. Better to include this in the actual mask!
        map: 3xnpix array of t,q,u maps
        mapname: str, name of map for loading ext=4
        flag: str, either 'external', 'internal', or 'none'
        wcov_dictname: name of the external wcov in the dict
        dct:  dict object, for loading external wcov
        external_wcov_override: npix array, the external wcov map. To use if you want to load the wcov once outside the function instead of reloading here
        split_index: For External wcov map. int. index of list for the correct wcov, if dct[wcov_dictname] is a list. If None, assume it's a string
        Returns masked map
    """
    if flag == 'none' or flag is None:
        return map
    elif flag == 'external' or flag == 'internal':
        if flag == 'external':
            if external_wcov_override is not None:
                cov_map = external_wcov_override
            else:
                wcov_map = dct[wcov_dictname]
                if not (type(wcov_map) is str):
                    if split_index is None:
                        raise ValueError("Need to provide a split_index if dct[wcov_dictname] is a list")
                    wcov_map = wcov_map[split_index]
                cov_map = so_map.read_map(wcov_map, fields_healpix=0)
        elif flag == 'internal':
            if ncomp!=3: print("Warning: Checking misspix for T only; this will not work with Planck 545")
            cov_map = so_map.read_map(mapname, fields_healpix=4)
        goodpix = (cov_map.data != 0)
        if ncomp == 1:
            return map * goodpix # T only
        for i in range(3):
            map[i] = map[i] * goodpix
        return map
    else:
        raise ValueError("Missing pixels flag is %s. Should be one of 'external', 'internal', none'" % flag)

def zero_large_values(map, masks, flag, mapname='', threshold=1e20, ncomp=3):
    """Handle large values in the maps - likely missing pixels
       maps: 3xnpix array of t,q,u maps
       masks: (mask_t, mask_pol)
       flag: str, either 'none', 'zero', error'
       threshold: float, threshold above which to zero/error
       mapname: str, name of the map for error reporting
       Returns map (masked if 'zero')
    """

    if flag == "none":
        # Do nothing
        return map
    elif flag == "zero":
        # Zero pixels in the map greater than the threshold value
        absmap = np.abs(map)
        if ncomp == 1:
            good = absmap < threshold
            return map * good
        else:
            good = np.product(absmap<threshold, axis=0)
            for i in range(3):
                map[i] = map[i] * good
            return map

    elif flag == "error":
        # Raise an error if there are any pixels greater than the threshold value
        if not type(masks) is tuple:
            raise TypeError("masks has type %s; should be a 1 or 2-tuple" % type(masks))            
        if not (len(masks) > 1 or len(masks) == 2):
            raise ValueError("masks has len %s; should be a 1 or 2-tuple" % len(masks))
        if ncomp==1:        
            newmap = map * masks[0]
            amax = np.max(np.abs(newmap))
            if amax > threshold:
                raise ValueError("Maximum value of map %s greater than threshold %s" % (mapname, threshold))
        else:
            mask_t, mask_pol = masks
            mask3 = [mask_t, mask_pol, mask_pol]
            for i in range(3):
                newmap = map[i] * mask3[i]
                amax = np.max(np.abs(newmap))
                if amax > threshold:
                    raise ValueError("Maximum value of map %s greater than threshold %s" % (mapname, threshold))
        return map     
    else:
        raise ValueError("Large values flag is %s. Should be one of 'zero', 'none', error'" % flag)

def load_beam_planck_wl_old(wl_beamname, spec): # Works for npipe and official planck; superseded by below function that works for 545 also
    # Get Bl (since you take square root!) of Planck beams. Does not include crosstalk terms or pixel window.
    if wl_beamname is None:
        return np.ones(5000)
    wl = fits.open(wl_beamname)
    specdict = {'TT':1, 'EE':2, 'BB':3, 'TE':4}
    if spec in ['TT', 'EE', 'BB', 'TE']:
        return np.sqrt(wl[specdict[spec]].data[spec+'_2_'+spec][0])
    elif spec == 'ET':
        return np.sqrt(wl[4].data['TE_2_ET'][0])
    else:
        raise ValueError("spec must be one of 'TT' 'EE' 'BB' 'TE' 'ET' not %s" % spec)

def load_beam_planck_wl(beamname, spec):
    # Get Bl. Does not include crosstalk terms or pixel window
    # Works for planck Bl_TEB or Bl_ files
    if spec is None:
        spec = 'TT'
    if beamname is None:
        return np.ones(5000)
    bl = fits.open(beamname)[1].data
    if spec not in ['TT', 'TE', 'ET', 'EE', 'TB', 'BT', 'BB', 'EB', 'BE']:
        raise ValueError("unknown spec %s" % spec)
    if len(bl.names) == 3: #Bl_TEB file, names ['T', 'E', 'B']
        wl = bl[spec[0]] * bl[spec[1]]
    elif len(bl.names) == 1: # Bl_ file, names ['TEMPERATURE']
        assert spec == 'TT'
        wl = bl['TEMPERATURE'] * bl['TEMPERATURE']
    return np.sqrt(wl)
