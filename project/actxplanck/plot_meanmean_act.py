from matplotlib import pyplot as plt
import numpy as np
from glob import glob

specDir = "spectra"
plotDir = "plots/spectra"
freqs = ["150","150x90","90", "90x150"]
specs = ['EE','ET','TE','TT']
meanspecs = [sorted(glob(specDir+"/meanspec_%s*" % sp)) for sp in specs]

fig, ax = plt.subplots(figsize=(14,3.5))
for ii, sp in enumerate(meanspecs[3]): # TT
    ll, dl, dlerr = np.loadtxt(sp, unpack=True)
    plt.errorbar(ll, dl, yerr=dlerr, fmt=".", label=freqs[ii])
    
plt.xlabel("$\ell$")
plt.ylabel("$D_\ell$")
plt.legend()
plt.xlim(300,6000)
plt.ylim(-50, 6000)
plt.tick_params(axis='x',which='minor')
plt.tight_layout()
plt.savefig(plotDir+"/meanspecTT.pdf")
plt.savefig(plotDir+"/meanspecTT.png")

fig, ax = plt.subplots(figsize=(14, 3.5))
plt.yscale('log')
plt.xscale('log')
plt.tick_params(axis='y',which='minor')
plt.tick_params(axis='x',which='minor')
for ii, sp in enumerate(meanspecs[3]): # TT log
    ll, dl, dlerr = np.loadtxt(sp, unpack=True)
    plt.errorbar(ll, dl, yerr=dlerr, fmt=".", label=freqs[ii])
    
plt.xlabel("$\ell$")
plt.ylabel("$D_\ell$")
plt.legend()
plt.xlim(300,10000)
plt.ylim(20, 7000)
plt.tight_layout()
plt.savefig(plotDir+"/meanspecTT_log.pdf")
plt.savefig(plotDir+"/meanspecTT_log.png")


fig, ax = plt.subplots(figsize=(14,3.5))
plt.xscale('log')
plt.tick_params(axis='x',which='minor')
for ii, sp in enumerate(meanspecs[0]): # EE
    ll, dl, dlerr = np.loadtxt(sp, unpack=True)
    plt.errorbar(ll, dl, yerr=dlerr, fmt=".", label=freqs[ii])
plt.xlabel("$\ell$")
plt.ylabel("$D_\ell$")
plt.legend()
plt.xlim(300,10000)
plt.ylim(-30,50)
plt.tight_layout()
plt.savefig(plotDir+"/meanspecEE.pdf")
plt.savefig(plotDir+"/meanspecEE.png")

fig, ax = plt.subplots(figsize=(14,3.5))
plt.xscale('log')
plt.tick_params(axis='x',which='minor')

for ii, sp in enumerate(meanspecs[2] + meanspecs[1]): # TE
    ll, dl, dlerr = np.loadtxt(sp, unpack=True)
    plt.errorbar(ll, dl, yerr=dlerr, fmt=".", label=freqs[ii])
    
plt.xlabel("$\ell$")
plt.ylabel("$D_\ell$")
plt.legend()
plt.xlim(300,10000)
plt.ylim(-200,130)
plt.tight_layout()
plt.savefig(plotDir+"/meanspecTE.pdf")
plt.savefig(plotDir+"/meanspecTE.png")



    
                 
    
