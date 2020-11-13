from astropy.io import fits
import numpy as np
import healpy as hp
import re
from map_tools import read_map


def planck_misspix(mapname_list):
    if isinstance(mapname_list, str):
        mapname_list = [mapname_list]
    iicovmask_list = []
    for mapname in mapname_list:
        hdr = fits.getheader(mapname, 1)
        # Find the field containing the II covariance matrix
        ttypes = [key for key in hdr.keys() if re.match(r'TTYPE\d', key)]
        ttype_iicov = [ttype for ttype in ttypes if hdr[ttype].strip() == "II_COV"]
        if len(ttype_iicov) == 0:
            continue
        else:
            ttype_iicov = ttype_iicov[0]
        field = int(ttype_iicov[-1]) - 1  # -1 due to zero indexing
        iicovmask = hp.read_map(mapname, field=field, verbose=False)
        iicovmask_list.append(iicovmask)
    if len(iicovmask_list) == 0:
        misspix = np.ones_like(hp.read_map(mapname_list[0], verbose=False))
    else:
        misspix = np.ones_like(iicovmask_list[0], int)
        for iicovmask in iicovmask_list:
            misspix[iicovmask == 0] = 0
    return misspix


def mask_misspix(mapp):
    misspix = np.ones_like(mapp, int)
    misspix[mapp == hp.UNSEEN] = 0
    return misspix


def load_beam(beamname):
    try:
        beam= fits.getdata(beamname, ext=1)['TEMPERATURE']
        beam = beam/beam[0] # normalize
        return beam
    except:
        try:
            beam = np.loadtxt(beamname, unpack=True)[1]
            beam = beam/beam[0] # normalize
            return beam
        except:
            raise RuntimeError("Beam could not be loaded as fits ext 1 TEMPERATURE or as column 1 of a text file")
        
def apply_beam_planck_wl_xtalk(wl_beamname, clTT, clEE, clBB, clTE):
    wl = fits.open(wl_beamname)
    outbeam=[]
    for spec in ['TT', 'EE', 'BB', 'TE', 'ET']:
        xx = clTT*wl[1].data['TT_2_'+spec][0] + clEE*wl[2].data['EE_2_'+spec][0] + \
             clBB*wl[3].data['BB_2_'+spec][0] + clTE*wl[4].data['TE_2_'+spec][0]
        outbeam.append(xx)
    return outbeam

def load_beam_planck_wl(wl_beamname, spec): # Works for npipe and official planck
    # Get Bl (since you take square root!) of Planck beams. Does not include crosstalk terms or pixel window.
    wl = fits.open(wl_beamname)
    specdict = {'TT':1, 'EE':2, 'BB':3, 'TE':4}
    if spec in ['TT', 'EE', 'BB', 'TE']:
        return np.sqrt(wl[specdict[spec]].data[spec+'_2_'+spec][0])
    elif spec == 'ET':
        return np.sqrt(wl[4].data['TE_2_ET'][0])
    else:
        raise ValueError("spec must be one of 'TT' 'EE' 'BB' 'TE' 'ET' not %s" % spec)
    
def extend_array(arr, size, fill_val=None):
    if len(arr) >= size:
        return arr
    else:
        if fill_val is None:
            fill_val = arr[-1]
        narr = np.ones(size) * fill_val
        narr[:len(arr)] = arr
        return narr

def dl2cl(dl, ell):
    return dl * (2*np.pi) / (ell * (ell+1))

def load_map_misspix(mapname, mapcname, maptype, unit_factor, isPol):
    if isPol:
        field = (1,2)
    else:
        field=0
        
    mapp = read_map(mapname, maptype, field=field)
    if not isPol:
        mapp = np.array([mapp])
    if mapcname is None:
        planck_misspix_names = [mapname]
        mask_misspix_mapc = np.ones_like(mapp)
        mapc = None
    else:
        planck_misspix_names = [mapname, mapcname]
        mapc = read_map(mapcname, maptype, field=field)
        if not isPol:
            mapc = np.array([mapc])
        mask_misspix_mapc = mask_misspix(mapc)
    if maptype == 'HP':
        misspix = planck_misspix(planck_misspix_names)
        if misspix is None:
            misspix = np.ones_like(mapp)
    else:
        misspix = mask_misspix(mapp) * mask_misspix_mapc
    mapp = mapp * unit_factor
    if mapc is not None:
        mapc = mapc * unit_factor
    return mapp, mapc, misspix

alpha=dict()
alpha[('143','353')]=.0381#/(1-.038)
alpha[('217','353')]=.131#/(1-.131)
alpha[('143','545')]=.00223#/(1-.0022)
alpha[('217','545')]=.00767#/(1-.0077)
alpha[('143','857')]=2.91e-5#/(1.-2.91e-5)
alpha[('217','857')]=9.97e-5#/(1.-9.97e-5)
alpha[('143T','353')] = 0.0341 / (1 + 0.0341)
alpha[('143E', '353')] = 0.0392 / (1 + 0.0392)
alpha[('217T', '353')] = 0.143 / (1 + 0.143)
alpha[('217E', '353')] = 0.141 / (1 + 0.141)
alpha[('100T', '353')] = 0.0208 / (1 + 0.0208)
alpha[('100E', '353')] = 0.0192 / (1 + 0.0192)

def makecl2(al1,al2,cls):
    return (1.+al1)*(1.+al2)*cls[:,0]-(1.+al1)*al2*cls[:,1]-al1*(1.+al2)*cls[:,2]+al1*al2*cls[:,3]

def clean_external(cl1x2, cl1x2c, cl1cx2, cl1cx2c, freq1, freq2, clfreq1, clfreq2, alphadict=alpha):
    alpha1=alpha[(freq1,clfreq1)]
    alpha2=alpha[(freq2,clfreq2)]
    cls = np.array([cl1x2, cl1x2c, cl1cx2, cl1cx2c])
    cls = cls.transpose()
    clout = makecl2(alpha1, alpha2, cls)
    return clout


    
