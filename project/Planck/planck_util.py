import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

from astropy.io import fits
import scipy
import pymaster as nmt

class PlanckSpectra():

    def __init__(self, nside=2048, lmax=2508,
        binfile='planck_spectra/binused.dat',
        map_dir='maps/PR2/frequencyMaps',
        mask_dir='masks/'):
        # set up directories
        self.map_dir = map_dir
        self.mask_dir = mask_dir
        # maps should all be the same nside and (computed) lmax
        self.nside = nside
        self.lmax = lmax
        self.lmax_beam = 3*nside


        # set up binning
        binleft, binright = np.genfromtxt(binfile,
                                          unpack=True, usecols=(0,1), dtype=((int), (int)))
        # have to prepend the bins down to ell=2
        bonus_left, bonus_right = self.get_low_ell_bins(5, l_min=2)
        binleft, binright = np.hstack((bonus_left,binleft)), np.hstack((bonus_right,binright))

        # set up NaMaster bins (bins weighted as D_ell as specified by Planck)
        ells = np.arange(lmax+1)
        bpws = -1+np.zeros_like(ells) #Array of bandpower indices
        for i, (bl, br) in enumerate(zip(binleft, binright)):
            bpws[bl:br+1] = i
        self.weights = np.array([ 1 for l in ells])
        self.b = nmt.NmtBin(nside, bpws=bpws, ells=ells, weights=self.weights, lmax=lmax, is_Dell=True)
        self.lb = self.b.get_effective_ells()

        # compute and store the pixel window
        self.pixel_window = hp.sphtfunc.pixwin(nside, pol=False)[:self.lmax_beam]


    def subtract_mono_di(self, map_in, mask_in):
        map_masked = hp.ma(map_in)
        map_masked.mask = (mask_in<1)
        mono, dipole = hp.pixelfunc.fit_dipole(map_masked)
        m = map_in.copy()
        npix = hp.nside2npix(self.nside)
        bunchsize = npix // 24
        bad = hp.UNSEEN
        for ibunch in range(npix // bunchsize):
            ipix = np.arange(ibunch * bunchsize, (ibunch + 1) * bunchsize)
            ipix = ipix[(np.isfinite(m.flat[ipix]))]
            x, y, z = hp.pix2vec(self.nside, ipix, False)
            m.flat[ipix] -= dipole[0] * x
            m.flat[ipix] -= dipole[1] * y
            m.flat[ipix] -= dipole[2] * z
            m.flat[ipix] -= mono
        return m

    def load_mask(self, freq):
        """returns mask tuple, (temperature, polarization, polarization)"""
        maskT = hp.read_map(
                    f'{self.mask_dir}/COM_Mask_Likelihood-temperature-{freq}_2048_R2.00.fits',
                    verbose=False)
        maskP = hp.read_map(
            f'{self.mask_dir}/COM_Mask_Likelihood-polarization-{freq}_2048_R2.00.fits',
            verbose=False)
        return (maskT, maskP, maskP)

    def load_map(self, freq, split):
        mfile = f'{self.map_dir}/HFI_SkyMap_{freq}_2048_R2.02_halfmission-{split}.fits'
        return [
            hp.read_map(mfile, field=0, verbose=False), # I
            hp.read_map(mfile, field=1, verbose=False), # Q
            hp.read_map(mfile, field=2, verbose=False)] # U

    def load_bad_pix(self, freq, split):
        mfile = f'{self.map_dir}/HFI_SkyMap_{freq}_2048_R2.02_halfmission-{split}.fits'
        II_COV = hp.read_map(mfile, field=4, verbose=False)
        QQ_COV = hp.read_map(mfile, field=5, verbose=False)
        UU_COV = hp.read_map(mfile, field=6, verbose=False)
        
        badT = II_COV < -1e30
        badP = np.logical_or.reduce( 
            (II_COV < -1e30, QQ_COV < -1e30, UU_COV < -1e30) )
        return (badT, badP, badP)

#         m0_file = f'maps/PR3/frequencyMaps/HFI_SkyMap_{freq}_2048_R3.01_halfmission-{split}.fits'
#         map0 = [hp.read_map(m0_file, field=0, verbose=False), # I
#                 hp.read_map(m0_file, field=1, verbose=False), # Q
#                 hp.read_map(m0_file, field=2, verbose=False)] # U
#         badpix0 = [np.logical_or(m < -1e30, m > 0.002) for m in map0]
#         return badpix0
        
                            

    def load_beam(self, freq1, freq2):
        if float(freq1) > float(freq2):
            beam_filename = f'planck_beam/beam_likelihood_{freq2}hm1x{freq1}hm2.dat'
        else:
            beam_filename = f'planck_beam/beam_likelihood_{freq1}hm1x{freq2}hm2.dat'
        beam_ell, beam = np.genfromtxt(beam_filename, unpack=True) # beam file is ell, Bl
        Bl = np.zeros(self.lmax_beam)
        Bl[beam_ell.astype(int)] = beam
        return Bl

    def compute_master(self, f_a,f_b,wsp) :
        cl_coupled=nmt.compute_coupled_cell(f_a,f_b)
        cl_decoupled=wsp.decouple_cell(cl_coupled)
        return cl_decoupled

    def compute(self, freq1, freq2, split1, split2,
        subtract_mono_and_dipole=True, zero_bad_pixels=True, n_iter=0):

        # read masks from dict or disk
        mask1 = self.load_mask(freq1)
        mask2 = self.load_mask(freq2)
        map1 = self.load_map(freq1, split1)
        map2 = self.load_map(freq2, split2)

        # deal with missing pixels by using the pixel covariance maps
        if zero_bad_pixels:
            badpix1 = self.load_bad_pix(freq1, split1)
            badpix2 = self.load_bad_pix(freq2, split2)
            for i in range(3):
                map1[i][badpix1[i]] = 0.0
                mask1[i][badpix1[i]] = 0.0
                map2[i][badpix2[i]] = 0.0
                mask2[i][badpix2[i]] = 0.0

        # subtract monopole and dipole
        if subtract_mono_and_dipole:
#             map1[0] = self.subtract_mono_di(map1[0], mask1[0])
#             map2[0] = self.subtract_mono_di(map2[0], mask2[0])
            map1 = [self.subtract_mono_di(m, mask)
                for m, mask in zip(map1, mask1)]
            map2 = [self.subtract_mono_di(m, mask)
                for m, mask in zip(map2, mask2)]

        # obtain beam
        Bl = self.load_beam(freq1, freq2)

        # set up NaMaster objects
        f1t = nmt.NmtField(mask1[0], [map1[0]],
            beam=(Bl*self.pixel_window), n_iter=n_iter)
        f2t = nmt.NmtField(mask2[0], [map2[0]],
            beam=(Bl*self.pixel_window), n_iter=n_iter)
        f1p = nmt.NmtField(mask1[1],[map1[1], map1[2]],
            beam=(Bl*self.pixel_window), n_iter=n_iter)
        f2p = nmt.NmtField(mask2[1],[map2[1], map2[2]],
            beam=(Bl*self.pixel_window), n_iter=n_iter)

        w0=nmt.NmtWorkspace()
        w0.compute_coupling_matrix(f1t,f2t, self.b, n_iter=n_iter)
        w1=nmt.NmtWorkspace()
        w1.compute_coupling_matrix(f1t,f2p, self.b, n_iter=n_iter)
        w2=nmt.NmtWorkspace()
        w2.compute_coupling_matrix(f1p,f2p, self.b, n_iter=n_iter)

        self.last_workspace = (w0, w1, w2) # store last workspace for debugging

        # put it into a nice dictionary
        Cb={}
        Cb['TT'] = self.compute_master(f1t, f2t, w0)[0]
        spin1 = self.compute_master(f1t, f2p, w1)
        Cb['TE'] = spin1[0]
        Cb['TB'] = spin1[1]
        spin2 = self.compute_master(f1p, f2p, w2)
        Cb['EE'] = spin2[0]
        Cb['EB'] = spin2[1]
        Cb['BE'] = spin2[2]
        Cb['BB'] = spin2[3]

        # T1/P2 (already done), T2/P1
#         spin1 = self.compute_master(f2t,f1p,w1)
#         Cb['TE'] += spin1[0]
#         Cb['TB'] += spin1[1]
#         Cb['TE'] /= 2.0
#         Cb['TB'] /= 2.0

        Cb['ET']=Cb['TE']
        Cb['BT']=Cb['TB']

        return Cb


    def get_low_ell_bins(self, width=5, top_ell=30, l_min=2):
        bonus_bin_left, bonus_bin_right = [], []
        counter = 0
        for i in list(range(1,top_ell))[::-1]:
            counter += 1
            if i == l_min:
                bonus_bin_left.append(i)
                bonus_bin_right.append(i+width-1)
                break
            elif counter % width == 0:
                bonus_bin_left.append(i)
                bonus_bin_right.append(i+width-1)
        return bonus_bin_left[::-1], bonus_bin_right[::-1]
