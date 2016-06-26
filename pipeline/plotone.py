from __future__ import print_function
##CUSP UO 2016
__author__ = "fbb"

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
import subprocess

from images2gif import writeGif
from PIL import Image, ImageSequence
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import IPython.display as IPdisplay
import multiprocessing as mpc

from findImageSize import findsize
s = json.load( open("fbb_matplotlibrc.json") )
pl.rcParams.update(s)

### EXTRACT: if True extracts the lightcurves from imaging data
EXTRACT = True
EXTRACT = False

#enable parallel processing
NOPARALLEL = True
NOPARALLEL = False
#must set the backand to agg for parallel processing to work
if not NOPARALLEL: matplotlib.use('agg')

### READ: if True reads the results of the PCA/KM clustering
### stored in earlier runs, if they exist
READ = True
#READ = False
DETREND = 0
### nmax: maximum number of datapoints per lightcurve (overwritten by 
#nmax=100
### lmax: maximum number of windows to use 
#lmax = 10

FQOFFSET = 0.01

font = {'size'   : 18}

kelly_colors_hex = [
    '#FFB300', # Vivid Yellow
    '#803E75', # Strong Purple
    '#FF6800', # Vivid Orange
    '#A6BDD7', # Very Light Blue
    '#C10020', # Vivid Red
    '#CEA262', # Grayish Yellow
    '#817066', # Medium Gray
    '#007D34', # Vivid Green
    '#F6768E', # Strong Purplish Pink
    '#00538A', # Strong Blue
    '#FF7A5C', # Strong Yellowish Pink
    '#53377A', # Strong Violet
    '#FF8E00', # Vivid Orange Yellow
    '#B32851', # Strong Purplish Red
    '#F4C800', # Vivid Greenish Yellow
    '#7F180D', # Strong Reddish Brown
    '#93AA00', # Vivid Yellowish Green
    '#593315', # Deep Yellowish Brown
    '#F13A13', # Vivid Reddish Orange
    '#232C16', # Dark Olive Green
    ]

wave = lambda t, phi, fq: np.sin(2.*fq*np.pi*t + phi*np.pi)/0.7
resd = lambda phi, fq, x, t: np.sum((wave(t*fq, phi, fq) - x)**2)
resd_freq = lambda fq, phi, x, t: np.sum((wave(t*fq, phi, fq) - x)**2)



def plotit(arg, options):
    
    filepattern, x, y = arg
    print (x,y)
    x=int(float(x))
    y=int(float(y))
    impath = os.getenv("UIdata") + filepattern
    print ("\n\nUsing image path: %s\n\n"%impath)
    fnameroot = filepattern.split('/')[-1]
    nmax = options.nmax

    if  glob.glob(impath+"*0000.raw")>0:
        img = glob.glob(impath+"*0000.raw")[0]


        flist = sorted(glob.glob(impath+"*.raw"))

        print ("Total number of image files: %d"%len(flist))

        nmax = min(options.nmax, len(flist)-options.skipfiles)
    
        print ("Number of timestamps (files): %d"%(nmax))
        if nmax<30: return 0
        flist = flist[options.skipfiles:nmax + options.skipfiles]    
        
        

    print ("Max number of windows to use: 1")
    outdir0 = '/'.join(filepattern.split('/')[:-1])+'/N%04dS%04d'%(nmax,
                                                                   options.skipfiles)
    outdir = '/'.join(filepattern.split('/')[:-1])+'/N%04dW%04dS%04d'%(nmax,
                                                                       options.lmax,
                                                                       options.skipfiles)

    if not os.path.isdir(outdir0):
        subprocess.Popen('mkdir -p %s/%s'%(outdir0+'pickles'), shell=True)
        #os.system('mkdir -p %s'%outdir0+"/pickles")
        subprocess.Popen('mkdir -p %s/%s'%(outdir0+'gifs'), shell=True)
        #os.system('mkdir -p %s'%outdir0+"/gifs")
    print ("Output directories: ",
           '/'.join(filepattern.split('/')[:-1]), outdir0, outdir)

    if (os.path.getsize(filepattern+".log"))>100000:
        #print (os.path.getsize(filepattern+".log"), filepattern+".log")
        #print ('tail -1000 %s > tmplog | mv tmplog %s'%(filepattern+".log", filepattern+".log"))
        try: subprocess.Popen('tail -1000 %s > tmplog | mv tmplog %s'%(filepattern+".log", filepattern+".log"))
        except OSError: pass

    logfile = open(filepattern+".log", "a")
    print ("Logfile:", logfile)
    print ("\n\n\n\t\t%s"%str(datetime.datetime.now()), file=logfile)
    print ("options and arguments:", options,
           arg, file=logfile)

    bsoutfile = outdir + "/" + fnameroot + "_bs.npy"
    goodcoordsoutfile = outdir + "/" + fnameroot + "_goodcoords.npy"
    pcaresultfile = outdir + "/" + fnameroot + "_PCAresult.pkl"        
    
    if os.path.isfile(bsoutfile):
        print ("reading old data file", bsoutfile)
        bs = np.load(bsoutfile)
    else:
        print ("data files %s and %s must exist already"%(bsoutfile,
                                                          coordsoutfile))
        
    if os.path.isfile(pcaresultfile) and os.path.isfile(goodcoordsoutfile.replace("goodcoords.npy","PCAamplitudes.npy")):
        pca = pkl.load(open(pcaresultfile))
        timeseries = bs
        PCAr = np.load(goodcoordsoutfile.replace("goodcoords.npy","PCAamplitudes.npy"))

    else: 
        print ("\n### Need a PCA output file. Run the pipeline first! PCA")
        
    if   os.path.isfile(goodcoordsoutfile):
        print ("loading", goodcoordsoutfile)
        goodcoords = np.load(goodcoordsoutfile)
    else:
        print  ("\n### Need a goof coords output file. Run the pipeline first!")
    #fig = pl.figure(figsize=(15,10))
    fig, (ax1, ax2) = pl.subplots(2, sharex=True, sharey=True, figsize=(15,10))
    #ax1 = fig.add_subplot(211)
    print (goodcoords[2], goodcoords[3])
    lightindx = (goodcoords[2].astype(int)==x) *  (goodcoords[3].astype(int)==y)
    phase = goodcoords[0][lightindx]
    freq = goodcoords[6][lightindx]
    indx = int(goodcoords[1][lightindx])
    print (lightindx, phase, freq, indx)
    ax1.tick_params(axis='both', which='major', labelsize=18)

    ax1.plot(np.arange(len(bs[indx]))*options.sample_spacing,
            (bs[indx]-bs[indx].mean())/bs[indx].std(), 'k', lw=3, alpha=0.8,
            label=r"data (x=%d, y=%d)"%(x,y))
    ax1.plot(np.arange(len(bs[indx]))*options.sample_spacing,
            PCAr[indx,1] * pca.components_[0] +
            PCAr[indx,2] * pca.components_[1], color='IndianRed', lw=3,
            label=r"2 component PCA reconstruction")
    ax1.legend(loc=0, ncol=2)

    fig.subplots_adjust(hspace=0)
    ax1.set_yticks([-1.5,0,1.5])

    #ax2 = fig.add_subplot(212, sharex=ax1, )
    pl.tick_params(axis='both', which='major', labelsize=18)
    ax2.plot(np.arange(len(bs[indx]))*options.sample_spacing,
            (bs[indx]-bs[indx].mean())/bs[indx].std(), 'k', lw=3, alpha=0.8,
            label=r"data (x=%d, y=%d)"%(x,y))
    ax2.plot(np.arange(len(bs[indx]))*options.sample_spacing,
            wave(np.arange(len(bs[indx]))*options.sample_spacing,
                 phase, freq), color='SteelBlue', lw=3, label=r"sine wave, $phi$=%.2f $\nu$=%.2f"%(phase, freq))
    ax2.set_yticks([-1.5,0,1.5])
    
    pl.legend(loc=0, ncol=2)
    pl.xlabel("time (seconds)", fontsize=18)
    pl.savefig(bsoutfile.replace("bs.npy","%d_%d_lcv.pdf"%(x,y)))
    pl.show()
    return bs


if __name__=='__main__':

    parser = optparse.OptionParser(usage="getallcv.py 'filepattern' x y ",
                                   conflict_handler="resolve")
    parser.add_option('--nmax', default=100, type="int",
                      help='number of images to process (i.e. timestamps)')
    parser.add_option('--lmax', default=None, type="int",
                      help='number of lights')
    parser.add_option('--skipfiles', default=150, type="int",
                      help="number of files to skip at the beginning")
    parser.add_option('--sample_spacing', default=0.25, type="float",
                      help="camera sample spacing (inverse of sample rate)")

    parser.add_option('--coordfile', default=None, type="str",
                      help='coordinates python array (generated by windowFinder.py)')
    
    options,  args = parser.parse_args()
    #options.lmax=500
    options.coordfil="coordfile stacks/groundtest1/ESB_c0.7Hz_250ms_2016-05-24-230354_N20_coords.npy"
    
    print ("options", options)
    print ("args", args, args[0])
    
    if len(args) < 3:
        sys.argv.append('--help')
        options,  args = parser.parse_args()
           
        sys.exit(0)
    print (args, options)
    bs = plotit(args, options)
    
