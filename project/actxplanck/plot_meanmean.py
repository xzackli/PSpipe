from matplotlib import pyplot as plt
import numpy as np
from glob import glob

specDir = "spectra"
plotDir = "plots/spectra"
freqs = ["143x150", "100x090"]
th = np.loadtxt("/global/cscratch1/sd/rosenber/ACT_data/ACT_bestfit_theory_CAMB.txt", unpack=True)
th_ell, th_tt, th_te, th_ee = th[0], th[1], th[4], th[2]

specs = ['EE','ET','TE','TT']
meanspecs = []
for sp in specs:
    allfreqs = [glob(specDir+"/meanspec_%s_%s" % (sp,freq))[0] for freq in freqs]
    meanspecs.append(allfreqs)

fig, ax = plt.subplots(figsize=(10,6))
for ii, sp in enumerate(meanspecs[3]): # TT
    ll, dl = np.loadtxt(sp, unpack=True)
    plt.plot(ll, dl, ".", label=freqs[ii])
plt.plot(th_ell, th_tt, 'k',linewidth=1)    
plt.xlabel("$\ell$")
plt.ylabel("$D_\ell$")
plt.legend()
plt.xlim(0,2500)
plt.ylim(-50, 6000)
plt.tick_params(axis='x',which='minor')
plt.tight_layout()
plt.savefig(plotDir+"/meanspecTT.pdf")
plt.savefig(plotDir+"/meanspecTT.png")

fig, ax = plt.subplots(figsize=(10,6))
plt.yscale('log')
plt.xscale('log')
plt.tick_params(axis='y',which='minor')
plt.tick_params(axis='x',which='minor')
for ii, sp in enumerate(meanspecs[3]): # TT log
    ll, dl = np.loadtxt(sp, unpack=True)
    plt.plot(ll, dl, ".", label=freqs[ii])
plt.plot(th_ell, th_tt, 'k',linewidth=1)        
plt.xlabel("$\ell$")
plt.ylabel("$D_\ell$")
plt.legend()
plt.xlim(300,2500)
plt.ylim(20, 7000)
plt.tight_layout()
plt.savefig(plotDir+"/meanspecTT_log.pdf")
plt.savefig(plotDir+"/meanspecTT_log.png")


fig, ax = plt.subplots(figsize=(10,6))
plt.plot(th_ell, th_ee, 'k',linewidth=1)    
plt.xscale('log')
plt.tick_params(axis='x',which='minor')
for ii, sp in enumerate(meanspecs[0]): # EE
    ll, dl = np.loadtxt(sp, unpack=True)
    plt.plot(ll, dl, ".", label=freqs[ii])
plt.xlabel("$\ell$")
plt.ylabel("$D_\ell$")
plt.legend()
plt.xlim(300,2500)
plt.ylim(-30,50)
plt.tight_layout()
plt.savefig(plotDir+"/meanspecEE.pdf")
plt.savefig(plotDir+"/meanspecEE.png")

fig, ax = plt.subplots(figsize=(10,6))
plt.plot(th_ell, th_te, 'k',linewidth=1)    
plt.xscale('log')
plt.tick_params(axis='x',which='minor')

for ii, sp in enumerate(meanspecs[2]): # TE
    ll, dl = np.loadtxt(sp, unpack=True)
    plt.plot(ll, dl, ".", label=freqs[ii])
    
plt.xlabel("$\ell$")
plt.ylabel("$D_\ell$")
plt.legend()
plt.xlim(300,2500)
plt.ylim(-200,130)
plt.tight_layout()
plt.savefig(plotDir+"/meanspecTE.pdf")
plt.savefig(plotDir+"/meanspecTE.png")

fig, ax = plt.subplots(figsize=(10,6))
plt.plot(th_ell, th_te, 'k',linewidth=1)    
plt.xscale('log')
plt.tick_params(axis='x',which='minor')

for ii, sp in enumerate(meanspecs[1]): # ET
    ll, dl = np.loadtxt(sp, unpack=True)
    plt.plot(ll, dl, ".", label=freqs[ii])
    
plt.xlabel("$\ell$")
plt.ylabel("$D_\ell$")
plt.legend()
plt.xlim(300,2500)
plt.ylim(-200,130)
plt.tight_layout()
plt.savefig(plotDir+"/meanspecET.pdf")
plt.savefig(plotDir+"/meanspecET.png")



    
                 
    
