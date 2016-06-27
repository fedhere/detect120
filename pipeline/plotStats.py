import glob
import numpy as np
import optparse
import sys
import os
import pickle as pkl
import json
import scipy.optimize
import datetime
import itertools
import matplotlib

import pylab as pl

def plotstats(phases, fname, PC=None):
    #plots distribution of freq/phases over a run
    pcs = [16., 50., 68.]    
    fig = pl.figure(figsize=(10,8))
    ax = fig.add_subplot(211)
    pl.hist(phases[6], color="IndianRed", alpha=0.8)    
    ax.set_xlabel("frequency (Hz)")
    ax.set_title("Run %s"%fname.split("_")[-2].split(".")[0])
    pc =  np.percentile(phases[-1], pcs)
    for k in pc:
        ax.plot([k, k],[0, ax.get_ylim()[1]],
                 '-', color='black', alpha=0.5)        
    ax.set_xlim(0.18,0.33)
    ax = fig.add_subplot(212)
    pl.hist(phases[0], color="SteelBlue", alpha=0.8)
    ax.set_xlabel("phase")
    ax.set_xlabel(r"phase ($\pi$ rad)")
    if not PC is None:
        pl.hist((np.arctan2(PC[:,1][PC[:,0]>0.2],PC[:,2][PC[:,0]>0.2])+np.pi)/np.pi, color="Grey", alpha=0.8)               
    pc =  np.percentile(phases[0], pcs)
    for k in pc:
        ax.plot([k, k],[0, ax.get_ylim()[1]],
                 '-', color='black', alpha=0.5)
    
    pl.savefig(fname)
