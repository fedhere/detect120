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

GIF = True
GIF = False

#enable parallel processing
NOPARALLEL = True
#NOPARALLEL = False
#must set the backand to agg for parallel processing to work
#if not NOPARALLEL: matplotlib.use('agg')

### READ: if True reads the results of the PCA/KM clustering
### stored in earlier runs, if they exist
READ = True
#READ = False
DETREND = 0
### nmax: maximum number of datapoints per lightcurve (overwritten by 
#nmax=100
### lmax: maximum number of windows to use 
#lmax = 10

FQOFFSET = 0.02
TOL = 1e-10
font = {'size'   : 13}

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
resd = lambda phi, fq, x, t: np.sum((wave(t, phi, fq) - x)**2)
resd_freq = lambda fq, phi, x, t: np.sum((wave(t, phi, fq) - x)**2)

def resd_fold((p), data, time):
    phi, fq = p
    print (phi,fq)
    #pl.plot(time, data, 'o')
    #pl.plot(time, data, '-')    
    x,t = folding(data, time, fq)
    pl.plot(t, x, 'g')
    pl.plot(t, wave(t, phi, fq), 'r')
    pl.draw()
    raw_input()
    
    print ("tmp",phi)
    return np.sum((wave(t, phi, fq) - x)**2)

def resd_freq_fold(fq, phi, data, time):
    x,t = folding(data, time, fq)
    np.sum((wave(t, phi, fq) - x)**2)

def folding(flux, runtime, freq, cycles=2):
    #print (freq)
    newtime = np.mod(runtime, float(cycles) / freq)
    indx = np.argsort(newtime)
    return flux[indx], newtime[indx]#np.mod(runtime, float(cycles) / freq)

def lnlike(theta, x, y):
    phi, freq = theta
    #y, x = folding(y, x, freq, cycles=2)
    #pl.plot(x,y, 'k-')
    model = wave (x, phi, freq)
    #pl.plot(x,model, 'r--')
    return -0.5*(np.sum((y-model)**2))#*inv_sigma2 - np.log(inv_sigma2)))

def lnprior(theta):
    phi, freq = theta
    if 0 < phi < 2 and 0.18 < freq < 0.35 :
        return 0.0
    return -np.inf

def lnprob(theta, x, y):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y)

def plotstats(phases, fname, PC=None):

    pcs = [16., 50., 68.]    
    fig = pl.figure(figsize=(10,8))
    ax = fig.add_subplot(211)
    pl.hist(phases[-1], color="IndianRed", alpha=0.8)    
    ax.set_xlabel("frequency (Hz)")
    ax.set_title("Run %s"%fname.split("_")[-2].split(".")[0])
    pc =  np.percentile(phases[-1], pcs)
    for k in pc:
        ax.plot([k, k],[0, ax.get_ylim()[1]],
                 '-', color='black', alpha=0.5)        

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

def rsquared (data,model):
    datamean = data.mean()
    return np.sum((model - datamean)**2)/np.sum((data-datamean)**2)

def chisq(data, model):
    return np.sum((data-model)**2)/(data-model).std()/(len(data)-2)

def makeGifLcPlot(x, y, ind, ts, xfreq, aperture, fnameroot,
                  imshape, flist, showme=False, fft=False,
                  stack=None, outdir="./"):

    fig = callplotw(x, y, ind, ts, xfreq, flist, aperture,
                    imshape, fname = fnameroot,
                    giflen=160, stack=stack, outdir=outdir)

    if not fft: fig.savefig(fnameroot + "_%d_%d"%(x,y)+".png")
    else: fig.savefig(fnameroot + "_%d_%d"%(x,y)+"_fft.png")
    if showme:
        pl.show()
        print (fnameroot + "_%d_%d"%(x,y)+".GIF")
        IPdisplay.Image(url=fnameroot + "_%04d_%04d"%(x,y)+".GIF")
    pl.close(fig)
    
def callplotw(xc, yc, i, ts, xfreq, flist, aperture, imshape, fft=False,
              fname = None, giflen=40, stack=None, outdir='./'):

   fig = pl.figure(figsize=(10,10))
   ax1 = fig.add_subplot(221)
   if stack is None:
       print ("no stack")
       ind = min([len(flist),10])
       img = np.fromfile(flist[ind],
                        dtype=np.uint8).reshape(imshape['nrows'],
                                                imshape['ncols'],
                                                imshape['nbands'])
   else:
       img = stack

   plotwindows(xc-25, yc-25, xc+25, yc+25,
               img, imshape, axs=ax1, c='w',
               plotimg = 0)
   ax1.set_title("window coordinates: %d %d"%(xc, yc))
    
   ax2 = fig.add_subplot(222)
   
   plotwindows(xc-aperture, yc-aperture, xc+aperture, yc+aperture, 
               img, imshape, wsize=(30,30), axs=ax2, c='Lime',
               plotimg = 0)
   ax2.set_title("aperture %d, index %d"%(aperture, i))
   ax2.axis('off')
   
   if fname:
       try:
           images = [Image.fromarray(np.fromfile(f,
                             dtype=np.uint8).reshape(imshape['nrows'],
                                                     imshape['ncols'],
                                imshape['nbands'])[yc-22:yc+21,
                                                   xc-22:xc+21]) \
                 for f in flist[:giflen]]
           fullfname = outdir + "/gifs/" + fname.split('/')[-1]
           if GIF:
               try: 
                   writeGif(fullfname + "_%04d_%04d"%(xc,yc)+".GIF",
                        images, duration=0.01)
               except: 
                   print ("failed to write to disk")
                   pass
       #print ("Done writing GIF %s_%04d_%04d.GIF"%(fname, xc, yc))
       except ValueError:
           print ("Error in giffing") 

   ax3 = fig.add_subplot(223)
   ax3.plot(ts, label = "time series")
   ax3.legend()
   ax4 = fig.add_subplot(224)
   if fft:
       ax4.plot(np.abs(np.fft.irfft(ts)), 
                label = "ifft")
       #ax4.plot([0.25, 0.25], [ax4.get_ylim()[0], ax4.get_ylim()[1]])
   else:
       ax4.plot(np.abs(np.fft.rfft(ts))[2:], 
                label = "fft")
       ax4.plot([0.25, 0.25], [ax4.get_ylim()[0], ax4.get_ylim()[1]])
   ax4.legend()
   return fig


def sparklines(data, lab, ax, x=None, title=None,
               color='k', alpha=0.3,
               twomax = True, maxminloc=False, nolabel=False):
    if x is None :#and len(x)==0:
        x=np.arange(data.size)
    xmax, xmin = -99, -99
    ax.plot(x, data, color=color, alpha=alpha)
    ax.axis('off')
    ax.set_xlim(-(x.max()-x.min())*0.1, (x.max()-x.min())*1.1)
    ax.set_ylim(ax.get_ylim()[0]-(ax.get_ylim()[1]-ax.get_ylim()[0])/10, ax.get_ylim()[1])

    ax.text(-0.1, 0.97, lab, fontsize = 10, 
            transform = ax.transAxes)
    if title:
        ax.plot((0,ax.get_xlim()[1]), 
                (ax.get_ylim()[1], ax.get_ylim()[1]), 'k-',)

    if nolabel:
       return  ((np.nan, np.nan), np.nan)
    ymax = max(data[:-5])
    xmaxind = np.where(data[:-5] == ymax)[0]
    #print (xmaxind, x[xmaxind])
    try:
        xmax =  x[xmaxind]
#        print ("xmax", xmax)
        ax.plot(xmax, ymax, 'o', ms=5, color='SteelBlue')
        if twomax:
            if xmaxind-3>0:
                max2data = np.concatenate([data[:xmaxind-3],data[xmaxind+3:-5]])
                x2 = np.concatenate([x[:xmaxind-3],x[xmaxind+3:-5]])
            else:
                max2data = data[3:-5]
                x2 = x[3:-5]
            xmax1 =  x2[np.where(max2data == max(max2data))[0]]
            #        print ("xmax1", xmax1, max(max2data))
            ax.plot(xmax1, max(max2data), 'o', ms=5,
                    color='SteelBlue', alpha=0.5)
        else: xmax1=np.nan  
    except ValueError:
        try:
            xmax = x[xmaxind[0]]
            ax.plot(xmax, ymax, 'o', ms=5, color='SteelBlue')

            max2data = np.concatenate([data[:xmax-5],data[xmax+5:-5]])
            x2 = np.concatenate([x[:xmax-5],x[xmax+5:-5]])
            xmax1 =  x2[np.where(max2data == max(max2data))[0][0]]
            ax.plot(xmax1, max(max2data), 'o', ms=5,
                color='SteelBlue', alpha=0.5)
        except IndexError:
            xmax, xmax1 = np.nan, np.nan
    try:
        xmin =  x[2:-5][np.where(data[2:-5] == min(data[2:-5]))[0]]   
        ax.plot(xmin, min(data[2:-5]), 'o', ms=5, color='IndianRed')
    except ValueError:
        try:
            xmin =  x[2:-5][np.where(data[2:-5] == min(data[2:-5]))[0][0]]  
            ax.plot(xmin, min(data[2:-5]), 'o', ms=5, color='IndianRed')
            
        except IndexError: pass
        
    if not isinstance(xmax, (np.int64,float)) and len(xmax)==0: xmax=np.nan
    if not isinstance(xmax1, (np.int64,float)) and len(xmax1)==0: xmax1=np.nan
    if not isinstance(xmin, (np.int64,float)) and len(xmin)==0: xmin=np.nan    
    
    if maxminloc:
        ax.text(1, 0.4, "%.2f"%(xmin), fontsize = 10, 
                transform = ax.transAxes, ha = 'right', color='IndianRed')
        ax.text(1, 0.7, "%.2f"%(xmax), fontsize = 10, 
                transform = ax.transAxes, ha = 'right', color='SteelBlue')
    else:
        ax.text(1, 0.4, "%.1f"%(min(data)), fontsize = 10, 
                transform = ax.transAxes, ha = 'right', color='IndianRed')
        ax.text(1, 0.7, "%.1f"%(max(data)), fontsize = 10, 
                transform = ax.transAxes, ha = 'right', color='SteelBlue')  

    return ((xmax, xmax1), xmin)


def extraction(((coord), xfreq, lmax,
               flist, filepattern, imsize, outdir,
                options, fig, ax, figfft, axfft)):

    cc,i = coord
    bs, fs  = get_plot_lca ((int(cc[0]+0.5)-options.aperture,
                             int(cc[1]+0.5)-options.aperture, 
                             int(cc[0]+0.5)+options.aperture+1,
                             int(cc[1]+0.5)+options.aperture+1),
                            flist, 
                            filepattern+'_x%d_y%d_ap%d'%(int(cc[0]+0.5),
                                                         int(cc[1]+0.5), 
                                                         options.aperture),
                            imsize, fft=options.fft,
                            verbose=False, showme=options.showme,
                            outdir = outdir)

    #pl.plot(b1)
    #print (bs[i].size)
    
    #sparklines(bs, '%d %d:%d'%(i, int(cc[0]), int(cc[1])), ax[-1])
    #if options.fft:
        #axfft.append(figfft.add_subplot(lmax/2+1,2,i+1))                
        #sparklines(fs[2:-5], '%d %d:%d'%(i, int(cc[0]), int(cc[1])),
        #           axfft[-1], x=xfreq[2:-5],
        #           maxminloc=options.fft)
        
    return bs, fs

def get_plot_lca (coords, flist, fname, imshape, fft=False, c='w', verbose=True,
                   showme=False, outdir='./'):
    
    if not fft:
        a = read_to_lc(coords, flist, fname, imshape, fft=fft,
                       showme=showme, c=c, verbose=verbose, outdir=outdir)
        afft = np.ones(a.size/2+1) * np.nan

    else:
        afft, a = read_to_lc(coords, flist, fname, imshape, fft=fft,
                  showme=showme, c=c, verbose=verbose, outdir=outdir)
    #print (a)
    flux0=(a-np.nanmean(a))/np.nanstd(a)
    flux=flux0.copy()
    
    if showme:
        pl.rc('font', **font)

        fig = pl.figure(figsize=(15,5))
        pl.plot(flux, color=c)
        pl.xlabel("img number")
        pl.ylabel("standardized flux")
        pl.show()
        pl.close(fig)
    return flux0, afft


def read_to_lc(coords, flist, fname, imshape, fft=False, c='w',
               showme=False, verbose = False, outdir = './'):
    x1, y1, x2, y2 = coords
    fullfname = outdir + "/pickles/" + fname.split('/')[-1]
    nmax = len(flist)
    if showme :
        plotwindows(x1,y1,x2,y2, flist[0], c=c)
    rereadnow = EXTRACT

    if not rereadnow:
        try:
            if verbose : print (fullfname+".pkl", os.isfile(fullfname+".pkl"))
            if not fft:
                try:
                    a = pkl.load(open(fullfname+".pkl",'rb'))
                except ValueError:
                    rereadnow = True
            else:
                try:
                    a = pkl.load(open(fullfname+".pkl",'rb'))
                    #print ("printing now ",a.size)
                    #pl.plot(a)
                    #pl.show()
                    afft = pkl.load(open(fullfname+"_fft.pkl",'rb'))
                    afft[0] = min(a)
                except ValueError:
                    rereadnow = True                    
        except:
            rereadnow = True
            print ("missing files: changing reread to True")
        
    if rereadnow:
        a = np.ones(nmax)*np.nan
        for i,f in enumerate(flist):
            try:
                a[i] = np.fromfile(f,
                        dtype=np.uint8).reshape(imshape['nrows'],
                                                imshape['ncols'],
                                imshape['nbands'])[y1:y2,x1:x2].sum()
            except: 
                a[i]=np.nan
                continue
        if not fft:
            pkl.dump(a, open(fullfname+".pkl","wb"))
        else:
            afft = np.abs(np.fft.rfft(a))
            pkl.dump(afft,open(fullfname+"_fft.pkl","wb"))
            if not os.path.isfile(fullfname+"_fft.pkl"):
                pkl.dump(a, open(fullfname+".pkl","wb"))
            
    try:
        f0 = np.fromfile(flist[0],
                     dtype=np.uint8).reshape(imshape['nrows'],
                                             imshape['ncols'],
                                     imshape['nbands'])[y1:y2,x1:x2]
    except IndexError:
        print ("something is wrong with the file list ", flist)
    '''
    print (flist[0])
    print (np.fromfile(flist[0],
                       dtype=np.uint8).reshape(imshape['nrows'],
                                               imshape['ncols'],
                                               imshape['nbands']))
    print (np.fromfile(flist[0],
                      dtype=np.uint8).reshape(imshape['nrows'],
                                              imshape['ncols'],
                                              imshape['nbands'])[y1:y2,x1:x2])
        
        #print (glob.glob(impath+"*.raw"))
    '''
    area = (x2-x1)*(y2-y1)
    if verbose:
        print ('mean: {0:.2f}, stdev: {1:.2f}, area: {2:d} pixels, {3:d} images, average flux/pixel: {4:.1f}'.format(
                    np.nanmean(a), np.nanstd(a), area,
            a.size, np.nanmean(a)/float(area)))


    if verbose:
        R = float(f0[:,:,0].sum())
        G = float(f0[:,:,1].sum())
        B = float(f0[:,:,2].sum())
        mx = np.argmax([R,G,B])
        colors= ['Red','Yellow','Blue']
        print (colors[mx],
               ': R: {0:.2f} G: {1:.2f} B: {2:.2f}'.format(1, G/R, B/R))

    if not fft: return  a
    
    return afft, a



def plotwindows(x1,y1,x2,y2, img, imshape, wsize=None,
                axs=None, c='w', plotimg=0):

    if not axs :
        fig, axs = pl.subplots(1,1,figsize=(15,15))
    #pl.imshow(np.fromfile(flist[0],dtype=np.uint8)\.reshape(nrow,ncol,nband))
    #pl.imshow(np.fromfile(flist[0],dtype=np.uint8).reshape(nrow,ncol,nband)[x1:x2,y1:y2])
    #axs.imshow(np.fromfile(flist[0],dtype=np.uint8).reshape(nrow,ncol,nband)[y1:y2,x1:x2],  interpolation='nearest')

    xmin,xmax,ymin,ymax = 0,0,0,0
    if plotimg == 0:
        if wsize:
            xmin = int(max(0,(x1+x2)/2-wsize[0]))
            xmax = int(min(imshape['ncols'],(x1+x2)/2+wsize[0]))
            ymin = int(max(0,(y1+y2)/2-wsize[1]))
            ymax = int(min(imshape['nrows'],(y1+y2)/2+wsize[1]))
            #print (ymin, ymax, xmin, xmax)
            axs.imshow(img[ymin:ymax, xmin:xmax],
                       interpolation='nearest')
        else:
            axs.imshow(img,
                       interpolation='nearest')            
        axs.set_xlim(0, axs.get_xlim()[1])
        axs.set_ylim(axs.get_ylim()[0], 0)
        
    axs.plot([x1-xmin,x2-xmin],[y1-ymin,y1-ymin], '-', color='%s'%c)    
    axs.plot([x2-xmin,x2-xmin],[y1-ymin,y2-ymin], '-', color='%s'%c)    
    axs.plot([x1-xmin,x2-xmin],[y2-ymin,y2-ymin], '-', color='%s'%c)    
    axs.plot([x1-xmin,x1-xmin],[y2-ymin,y1-ymin], '-', color='%s'%c)    
    #pl.show()

def fit_freq(freq, ts, imgspacing, phi0=0.0, iteratit=False,
             fold=False, mcmc=False, fp=None):
    lts =  np.arange(ts.size)*imgspacing
    signal = ts
    #if fold:
    #    signal, lts = folding(ts, lts, freq)
    savephi = 0
    minres = 99e9
    for phi in (np.arange(0,1.1,0.25)+phi0)*np.pi:
        #print (phi)
        res = resd(phi, np.abs(freq), signal, lts)
        #print (res, minres)
        if res<minres:
            savephi = phi
            minres = res
        #print (savephi)
    phi0 = savephi

    

    #print ("phase 0", phi0, freq)        
    fits = scipy.optimize.minimize(resd, phi0,
                                   args=(np.abs(freq),
                                         signal, lts))            
    phase = fits.x%2
    #print ("phase 1", phase, freq)
    fits = scipy.optimize.minimize(resd_freq, freq,
                                   args=(phase,
                                         signal, lts))
    freq=fits.x[0]
    #print ("freq 2", phase, freq)
            
    fits = scipy.optimize.minimize(resd, phase,
                                   args=(np.abs(freq),
                                         signal, lts))
    phase = fits.x[0]%2
    #print ("phase 3", phase, freq)    
    #raw_input()
    if mcmc:
        
        ndim, nwalkers = 2, 100
        pos = [np.array([phase,np.abs(freq)]) +
               1e-4*np.random.randn(ndim) for i in range(nwalkers)]
        #print (pos)
        import emcee
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                        args=(lts, signal))
        sampler.run_mcmc(pos, 500)
        samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
        samples[:,0]  = samples[:,0]+2
        import corner
        
        phase_all, freq_all = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))
        phase,freq = phase_all[0]-2, freq_all[0]
        #print (phase_all, freq_all)
                         
        if fp:
            import time
            fname = '/'.join(fp.split('/')[:-1])+"/triangles/"+fp.split('/')[-1]+"_triangle"
            subprocess.Popen('mkdir -p %s '%('/'.join(fp.split('/')[:-1])+"/triangles"), shell=True)
            
            cfig = corner.corner(samples, labels=["$\phi$", "freq"],
                      truths=[phase,freq])
            tm = time.time()
            cfig.savefig(fname+"%d.png"%int((tm-int(tm))*10000))
        
        phase_err, freq_err = phase_all[1:], freq_all[1:]
    else:
        if iteratit:
            iteration=0
            better=99
            while better>TOL and iteration<100:
                freqold = freq
                phaseold = phase

                
                fits = scipy.optimize.minimize(resd_freq, freq,
                                               args=(phase,
                                                     signal, lts))
                freq=fits.x

                
                fits = scipy.optimize.minimize(resd, phase,
                                               args=(np.abs(freq),
                                                     signal, lts))
                phase = fits.x%2

                iteration +=1            
                better = np.abs((freqold-freq))+np.abs((phaseold-phase))
                phase_err, freq_err = [0,0],[0,0]
                #print (iteration, freq, phase, better)
    sinwave = wave(lts, phase, np.abs(freq))

    if len(sinwave) == 2: thiswave = sinwave[0]
    chi2 = (chisq(signal, sinwave) if np.abs(freq-0.25)<0.06 else 99)
    #print (chi2)
    return phase, freq, chi2, sinwave, phase_err, freq_err

    


def fit_waves(filepattern, lmax, nmax, timeseries, transformed,
              ntot, img, fnameroot, freqs, stack, bs,
              logfile, allights, imgspacing,
              aperture, imsize, flist, xfreq, chi2thr, phi0=0, 
              fft=False,   showme=False,  km = None, fold=False,
              goodlabels = None, iteratit=False, mcmc=False,
              outdir="./", outdir0="./"):

    newntot = 0
    j = 0
    fig = pl.figure(figsize=(10,(int(ntot/2)+1)))#figsize=(10,100))
    ax = []
    fig2 = pl.figure()

    axs0 = fig2.add_subplot(211)
    axs1 = fig2.add_subplot(212)    
    #fig2 = pl.figure()
    #axs2 = fig2.add_subplot(111)
    
    axs0.imshow(stack, interpolation='nearest')
    axs0.set_xlim(0, axs0.get_xlim()[1])
    axs0.set_ylim(axs0.get_ylim()[0], 0)
    axs0.axis('off')

    axs1.imshow(img,  interpolation='nearest')
    axs1.set_xlim(0, axs1.get_xlim()[1])
    axs1.set_ylim(axs1.get_ylim()[0], 0)
    axs1.axis('off')
    print ("")
    
    ###start here if you just want to do the fitting

    #for i,cc in enumerate(allights[:lmax]):
    #phases = np.zeros((6,len(bs)))*np.nan
    phases = {'index':np.zeros(len(bs))*np.nan,
              'x':np.zeros(len(bs))*np.nan,
              'y':np.zeros(len(bs))*np.nan,
              'freq':np.zeros(len(bs))*np.nan,
              'chi2':np.zeros(len(bs))*np.nan,
              'phase':np.zeros(len(bs))*np.nan,
              'km_cluster':np.zeros(len(bs))*np.nan,
              'phase_e':np.zeros((len(bs),2))*np.nan,
              'freq_e':np.zeros((len(bs),2))*np.nan
              }
    if km:
        try:
            freq = [freqs[goodlabels[0]], freqs[goodlabels[1]]]
            i=2
            freq[0] = [ 0.25]
            while (freq[0] == freq[1] and i<len(goodlabels)):
                #print (goodkmeanlabels)
                freq[1] = freqs[goodlabels[i]]
                i=i+1
            if freq[0] == freq[1]: freq[1] = [0.32]
        except IndexError:
            freq = (freqs[goodlabels[0]])
    else:
        freq = freqs
        if isinstance(phi0, int):
            phi0=np.zeros(timeseries.shape[0])+phi0
    
    if fft:
        outphasesfile = open(filepattern+"_fft_phases_N%04dW%04dS%04d.dat"%(nmax,lmax, options.skipfiles), "w")
    else:
        outphasesfile = open(filepattern+"_phases_N%04dW%04dS%04d.dat"%(nmax,lmax,
         options.skipfiles), "w")
        
    print ("#index,x,y,chi2,phase,freq", file=outphasesfile)
    figsp = pl.figure(figsize = (10,(int(lmax/4)+1)))
    figspfft = pl.figure(figsize = (10,(int(lmax/4)+1)))
    #figspgood = pl.figure(figsize = (10,(int(lmax/4)+1)))
    axsp = []
    axspfft = []    
    goodcounter = 0
    #print (timeseries.shape, goodlabels.shape, transformed.shape)
    ilast = 0
    for i, Xc in enumerate(timeseries):
        stdtimeseries = (Xc-Xc.mean())/Xc.std()
        stdbs = (bs[i]-bs[i].mean())/bs[i].std()
#        print ("here", min(stdbs),  max(stdbs), np.mean(stdbs), np.std(stdbs))
        color, alpha = 'k', 0.3
        axsp.append(figsp.add_subplot(lmax,2,(i*2+1)))
        sparklines(stdbs, '%d'%i,
                   axsp[-1], color=color, alpha=alpha, nolabel=True)
        
        # if this is not already done in fft space save an fft plot also
        if not fft:
            axspfft.append(figspfft.add_subplot(lmax,2,(i*2+1)))
            power = np.abs(np.fft.rfft(stdbs))[2:-5]
#            print (power)
            xmaxs, xmin = sparklines(power, '%d'%i,
                                     axspfft[-1], x=xfreq[2:-5], color=color,
                                     alpha=alpha)
            ymin, ymax = axspfft[-1].get_ylim()
            axspfft[-1].plot([0.25,0.25],[ymin, ymax],'k-')
#            print ("\r freq max",xmaxs, end='')
            sys.stdout.flush()


        axsp.append(figsp.add_subplot(lmax,2,(i*2+2)))        

        if km:
            if not fft:
                sparklines(km.cluster_centers_[km.labels_[i]],
                           "cluster %i"%km.labels_[i], axsp[-1],
                           color=color, alpha=alpha)
                axspfft.append(figspfft.add_subplot(lmax,2,(i*2+2)))
                xmaxs, xmin = sparklines(np.abs(np.fft.rfft(km.cluster_centers_[km.labels_[i]])[2:-5]),
                                         "cluster %i"%km.labels_[i], axspfft[-1],
                                         x=xfreq[2:-5], color=color, alpha=alpha)
                #=print (xmaxs)            
                ymin, ymax = axspfft[-1].get_ylim()
                axspfft[-1].plot([0.25,0.25],[ymin, ymax],'k-')
            else:
                sparklines(transformed[km.labels_[i]][2:-5],
                           "cluster %i"%km.labels_[i], axsp[-1],
                           color=color, alpha=alpha, maxminloc=True)
        else:
            if not fft:
                sparklines(transformed[i][2:-5],
                                       "", axsp[-1],
                           color=color, alpha=alpha, maxminloc=True)
                #print (len(xfreq[2:-5]))
                axspfft.append(figspfft.add_subplot(lmax,2,(i*2+2)))
                xmaxs, xmin = sparklines(transformed[i][1:-5],
                                         "", axspfft[-1],
                                         x=xfreq[2:-5], color=color, alpha=alpha)
                #=print (xmaxs)            
                ymin, ymax = axspfft[-1].get_ylim()
                axspfft[-1].plot([0.25,0.25],[ymin, ymax],'k-')
            else:
                sparklines(transformed[i][2:-5],
                           "", axsp[-1],
                           color=color, alpha=alpha, maxminloc=True)
        final_good_windows = []
        
        if km and  km.labels_[i] in goodlabels:
            goodcounter +=1
            color, alpha = 'Navy', 0.8
#            print ("min and max: ",min(stdbs), max(stdbs))
            sparklines(stdbs, "",
                       axsp[-2], color=color, alpha=alpha)
        
            if not options.fft:
                sparklines(km.cluster_centers_[km.labels_[i]],
                   "", axsp[-1],
                                        color=color, alpha=alpha)
            else:
                sparklines(transformed[km.labels_[i]],
                           "cluster %i"%km.labels_[i], axsp[-1],
                           color=color, alpha=alpha, maxminloc=True)
            if len(freq)==2:
                print ("\r Analyzing window {0:d} of {1:d} (testing periods {2}, {3}, accepted so far {4})".format(goodcounter, ntot, freq[0], freq[1], newntot),
                   end="")
                sys.stdout.flush()

                print (" Analyzing & plotting sine window {0:d} (testing periods {1}, {2})".format(i, freq[0], freq[1]),
                       file=logfile)
            elif len(freq)==1:
                print ("\r Analyzing window {0:d} of {1:d} (testing periods {2}, accepted so far {3}))".format(goodcounter, ntot, freq[0], newntot),
                   end="")
                sys.stdout.flush()

                print (" Analyzing & plotting sine window {0:d} (testing periods {1})".format(i, freq[0]),
                       file=logfile)                

            j = j+1
            
            
            phases['phase'][i], phases['freq'][i], phases['chi2'][i], thiswave, phases['phase_e'][i], phases['freq_e'][i] = fit_freq(freq[0], stdbs, imgspacing, mcmc=mcmc, fold=fold)

            sparklines(thiswave,  "          %.2f %.2f"%(freq[0], phases['chi2'][i]),
                           axsp[-2], color='r', alpha=0.3, nolabel=True)
            if phases['chi2'][i] > chi2thr and len(freq)>1:
                phases['phase'][i], phases['freq'][i], phases['chi2'][i], thiswave, phases['phase_e'][i], phases['freq_e'][i] = fit_freq(freq[1], stdbs, imgspacing, mcmc=mcmc, fold=fold)
                     
                sparklines(thiswave,  "                   %.2f %.2f"%(freq[1], phases['chi2'][i]),
                               axsp[-2], color='y', alpha=0.3, nolabel=True)
                if phases['chi2'][i] > chi2thr:
                    print ("\tbad chi square fit to sine wave: %.2f"\
                           %(phases['chi2'][i]),
                           file=logfile)
                    #print ("\tbad chi square fit to sine wave: %.2f"\
                    #       %(phases['chi2'][i]))
                    continue
            print ("\t accepted freq: {0:f}".format(phases['freq'][i]),
                   end="")
                
            phases['index'][i] = goodlabels[i]
            phases['x'][i] = allights[i][0]
            phases['y'][i] = allights[i][1]
            phases['km_cluster'][i] = km.labels_[i]
 
            
            ax.append(fig.add_subplot(max(ntot/2+1, newntot+1), 1, (newntot+1)))
            
            ax[-1].plot(np.arange(stdbs.size) * options.sample_spacing,
                        stdbs, label="x=%.2f y=%.2f"%(phases['x'][i], phases['y'][i]))
            ax[-1].plot(np.arange(stdbs.size) * options.sample_spacing,
                        thiswave,
                        label=r"$\phi=%.2f \chi^2=%.2f$"%\
                        (phases['phase'][i], phases['chi2'][i]))
        
            ax[-1].set_xticks([])
            ax[-1].set_yticks([])            
            newntot += 1
            ax[-1].legend(frameon=False, fontsize=10)
            
        
            if GIF: makeGifLcPlot(phases['x'][i], phases['y'][i], i,
                                  stdtimeseries, xfreq, 
                                  aperture, filepattern,
                                  imsize, flist,
                                  showme = showme,
                                  fft = fft,
                                  stack = stack, outdir=outdir0)
    
            x1,x2=int(phases['x'][i])-aperture*3,\
                   int(phases['x'][i])+aperture*3
            y1,y2=int(phases['y'][i])-aperture*3,\
                   int(phases['y'][i])+aperture*3
            axs1.plot([x1,x2],[y1,y1], '-',
                      color='%s'%kelly_colors_hex[km.labels_[i]])    
            axs1.plot([x2,x2],[y1,y2], '-',
                      color='%s'%kelly_colors_hex[km.labels_[i]])    
            axs1.plot([x1,x2],[y2,y2], '-',
                      color='%s'%kelly_colors_hex[km.labels_[i]])    
            axs1.plot([x1,x1],[y2,y1], '-',
                      color='%s'%kelly_colors_hex[km.labels_[i]])

            print ("%d,%.2f,%.2f,%.2f,%.2f,%2f"%(i, phases['x'][i], phases['y'][i],
                                                 phases['phase'][i], phases['chi2'][i],
                                                 phases['freq'][i]),
                   file=outphasesfile)
            ilast = i

        elif not km:
            goodcounter +=1

            if len(freq)==2:
                print ("\r Analyzing window {0:d} of {1:d} (testing frequencies {2}, {3}, accepted so far {4})".format(goodcounter, ntot, freq[0], freq[1], newntot),
                       end="")
                sys.stdout.flush()

                print (" Analyzing & plotting sine window {0:d} (testing frequencies {1}, {2}".format(i, freq[0], freq[1]), file=logfile)
            elif len(freq)==1:
                print ("\r Analyzing window {0:d} of {1:d} (testing frequencies {2}, accepted so far {3}))".format(goodcounter, ntot, freq[0], newntot),
                   end="")
                sys.stdout.flush()

                print (" Analyzing & plotting sine window {0:d} (testing frequencies {1})".format(i, freq[0]),
                       file=logfile)                
            j = j+1
            phases['phase'][i], phases['freq'][i], phases['chi2'][i], thiswave, phases['phase_e'][i], phases['freq_e'][i] = fit_freq(freq[0], stdbs, imgspacing, phi0[i], iteratit=iteratit, mcmc=mcmc, fold=fold)#, fp = filepattern)

            sparklines(thiswave,  "          %.2f"%(phases['chi2'][i]),
                           axsp[-2], color='r', alpha=0.3, nolabel=True)
            if phases['chi2'][i] > chi2thr and len(freq)>1:
                phases['phase'][i], phases['freq'][i], phases['chi2'][i], thiswave, phases['phase_e'][i], phases['freq_e'][i]  = fit_freq(freq[1], stdbs, imgspacing, phi0[i], iteratit=iteratit, mcmc=mcmc, fold=fold)#, fp = filepattern)
                     
                sparklines(thiswave,  "                   %.2f"%(phases['chi2'][i]),
                               axsp[-2], color='y', alpha=0.3, nolabel=True)
                if phases['chi2'][i] > chi2thr:
                    print ("\tbad chi square fit to sine wave: %.2f"\
                           %(phases['chi2'][i]),
                           file=logfile)
                    #print ("\tbad chi square fit to sine wave: %.2f"\
                    #       %(phases['chi2'][i]))
                    continue
                print ("accepted freq: {0:f}".format(phases['freq'][i]),
                       end="")
            sys.stdout.flush()
                
            phases['index'][i] = goodlabels[i]
            phases['x'][i] = allights[goodlabels[i]][0]
            phases['y'][i] = allights[goodlabels[i]][1]
            
            ax.append(fig.add_subplot(max(ntot/2+1, newntot+1), 1, (newntot+1)))
            
            ax[-1].plot(np.arange(stdbs.size) * options.sample_spacing,
                        stdbs, label="x=%.2f y=%.2f"%(phases['x'][i], phases['y'][i]))
            ax[-1].plot(np.arange(stdbs.size) * options.sample_spacing,
                        thiswave,
                        label=r"$\phi$=%.2f $\chi^2$=%.2f $\nu$=%.2f$"%\
                        (phases['phase'][i], phases['chi2'][i], phases['freq'][i]))
        
            ax[-1].set_xticks([])
            ax[-1].set_yticks([])            
            newntot += 1
            ax[-1].legend(frameon=False, fontsize=10)
            

            if GIF: makeGifLcPlot(phases['x'][i], phases['y'][i],
                                  goodlabels[i], stdtimeseries, xfreq, 
                                  aperture, filepattern, imsize, flist,
                                  showme = showme, fft = fft,
                                  stack = stack, outdir=outdir0)
    
            x1,x2=int(phases['x'][i])-aperture*3,\
                   int(phases['x'][i])+aperture*3
            y1,y2=int(phases['y'][i])-aperture*3,\
                   int(phases['y'][i])+aperture*3
            axs1.plot([x1,x2],[y1,y1], '-',
                      color='%s'%kelly_colors_hex[3])    
            axs1.plot([x2,x2],[y1,y2], '-',
                      color='%s'%kelly_colors_hex[3])    
            axs1.plot([x1,x2],[y2,y2], '-',
                      color='%s'%kelly_colors_hex[3])    
            axs1.plot([x1,x1],[y2,y1], '-',
                      color='%s'%kelly_colors_hex[3])

            print ("%d,%.2f,%.2f,%.2f,%.2f,%2f"%(i, phases['x'][i], phases['y'][i],
                                                 phases['phase'][i], phases['chi2'][i],
                                                 phases['freq'][i]),
                   file=outphasesfile)

            
            
            ilast = i

    goodphases = np.array([phases['phase'][phases['chi2']< chi2thr],
                           phases['index'][phases['chi2']< chi2thr],
                           phases['x'][phases['chi2']< chi2thr],
                           phases['y'][phases['chi2']< chi2thr],
                           phases['km_cluster'][phases['chi2']< chi2thr],
                           phases['chi2'][phases['chi2']< chi2thr],
                           phases['freq'][phases['chi2']< chi2thr],
                           phases['phase_e'][:,0][phases['chi2']< chi2thr],
                           phases['phase_e'][:,1][phases['chi2']< chi2thr],
                           phases['freq_e'][:,0][phases['chi2']< chi2thr],
                           phases['freq_e'][:,1][phases['chi2']< chi2thr]])
    #print (goodphases, phases['x'], phases['y'])
    
    
    xticks = [0,5,10,15,20,25]
    ax[-1].set_xticks(xticks)
    ax[-1].set_xticklabels(["%d"%xt for xt in xticks])
    if not fft: 
        fignamekm = outdir + "/" + fnameroot + "_km_assignments.pdf"
        fignamegw = outdir + "/" + fnameroot + "_goodwindows.pdf"
        fignamefft = outdir + "/" + fnameroot + "_transform.pdf" 
        fignamegwf = outdir + "/" + fnameroot + "_goodwindows_fits.pdf"
        fignamegwfpca = outdir + "/" + fnameroot + "_goodwindows_fits_pca.pdf"
                                         
    else: 
        fignamekm = outdir + "/" + fnameroot + "_km_assignments_fft.pdf"
        fignamegw = outdir + "/" + fnameroot + "_goodwindows_fft.pdf"
        fignamegwfpca = outdir + "/" + fnameroot + "_goodwindows_fits_fft_pca.pdf"
        
    try:
        if km:
            figsp.savefig(fignamekm)
            if not fft:
                figspfft.savefig(fignamefft)
    except ValueError: 
        print ("could not save figure %s or %s"%(figname, fignamefft))
    pl.close(figsp)
    pl.close(figspfft)
    
    print ("")
    if newntot == 0:
        print ("\n   !!!No windows with the right time behavior!!!")
        return [-1], [-1]
        #sys.exit()
    axs1.set_title("%d good windows"%newntot)

    try:
        fig2.savefig(fignamegw)
    except ValueError: 
        print ("could not save figure %s"%figname)    
    pl.close(fig2)

    ax[-1].set_xlabel("seconds")
    try:
        if km:
            fig.savefig(fignamegwf)
        else:
            fig.savefig(fignamegwfpca)
                        
    except ValueError: 
        print ("could not save figure %s"%figname)
    pl.close(fig)
    print ("\nActual number of windows well fit by a sine wave (chisq<%.2f): %d"%(chi2thr, newntot))
    #sys.exit()
  
    outphasesfile.close()

    return goodphases, phases

def KMAnalysis(evecs, timeseries, xfreq, allights, stack, img, fnameroot, options):

    print ("\n### Starting K-Means clustering")

    #Note: the first and second PCA components are the sine waves 180 deg phase shifted.
    #If I want to make a model for the saturated waves i can use those. does not really work unless there are several saturated lights

    kmcenters = evecs[:options.nkmc].copy()
    if not options.fft:
        kmcenters[-1] = evecs[0]*10<0.5
        
        kmcenters[-2] = evecs[1]*10<0.5

    km = KMeans(n_clusters=options.nkmc, init=evecs[:options.nkmc])
    vals = ((timeseries.T - timeseries.mean(1))/timeseries.std(1)).T
    km.fit(vals)
    

    print ("\n### Determining phase for selected windows")
    fig = pl.figure(figsize = (10,10))
    kmax = []
    goodkmeanlabels, km_freqs = [], {}
    transformed = []
    for i,Xc in enumerate(km.cluster_centers_):
        kmax.append(fig.add_subplot(20,2,(i*2+1)))
        if not options.fft:
            sparklines(Xc, "%i (%d)"%(i, len(km.labels_[km.labels_==i])),
                       kmax[-1])
            kmax.append(fig.add_subplot(20,2,(i*2+2)))
            transformed.append( np.abs(np.fft.rfft(Xc))[1:])
            transformed[-1][0] = 0
            km_xmax, km_xmin = sparklines(transformed[-1][:-5], "",
                                          kmax[-1], x=xfreq[1:-5])
        else:
            kmax.append(fig.add_subplot(20,2,(i*2+2)))
            #print (Xc[1:-5])
            km_xmax, km_xmin = sparklines(Xc[1:-5], "%i"%i, kmax[-1],
                                          x=xfreq[1:-5], maxminloc=True)
            transformed.append(np.abs(np.fft.irfft(Xc)))
            sparklines(transformed[-1], "", kmax[-2])

            accepted = ''

        if np.abs(km_xmax[0] - 0.25) < 0.08:
           goodkmeanlabels.append(i)
           km_freqs[i] = km_xmax[0]
           accepted = "ACCEPTED!"            
        elif np.abs(km_xmax[1] - 0.25) < 0.08 :
           goodkmeanlabels.append(i)
           km_freqs[i] = km_xmax[1]            
           accepted = "ACCEPTED!"
        
           print ("cluster {0}: strongest frequecies {1}, {2} - {3}".\
                  format(i,
                         km_xmax[0],
                         km_xmax[1], accepted))


        ymin, ymax = kmax[-1].get_ylim()
        kmax[-1].plot([0.25,0.25],[ymin, ymax],'k-')
       #kmax[-1].text(0.8,0.8,"cluster %i"%i,
       #              color=kelly_colors_hex[i],
       #              transform=ax[-1].transAxes)

    print (" All labels: ", km.labels_, file=logfile)
    print (" Good labels:", goodkmeanlabels, file=logfile)

    
    if not options.fft: figname = outdir + "/" + fnameroot + "_KM.pdf"
    else: figname = outdir + "/" + fnameroot + "_KM_fft.pdf"
    pl.savefig(figname)
    pl.close(fig)
    
    transformed = np.array(transformed)
    
    fig = pl.figure()
    axs = fig.add_subplot(311)
    axs0 = fig.add_subplot(312)
    axs1 = fig.add_subplot(313)    
    
    
    axs.imshow(stack,  interpolation='nearest')
    axs.set_xlim(0, axs.get_xlim()[1])
    axs.set_ylim(axs.get_ylim()[0], 0)
    axs0.imshow(img,  interpolation='nearest')
    axs0.set_xlim(0, axs0.get_xlim()[1])
    axs0.set_ylim(axs0.get_ylim()[0], 0)
    axs1.imshow(img,  interpolation='nearest')
    axs1.set_xlim(0, axs1.get_xlim()[1])
    axs1.set_ylim(axs1.get_ylim()[0], 0)
    
    for i,cc in enumerate(allights):
        x1,x2=int(cc[0])-options.aperture*3,\
               int(cc[0])+options.aperture*3
        y1,y2=int(cc[1])-options.aperture*3,\
               int(cc[1])+options.aperture*3
        #axs0.plot([x1,x2],[y1,y1], '-',
        #         color='%s'%kelly_colors_hex[km.labels_[i]])    
        #axs0.plot([x2,x2],[y1,y2], '-',
        #         color='%s'%kelly_colors_hex[km.labels_[i]])    
        #axs0.plot([x1,x2],[y2,y2], '-',
        #         color='%s'%kelly_colors_hex[km.labels_[i]])    
        #axs0.plot([x1,x1],[y2,y1], '-',
        #         color='%s'%kelly_colors_hex[km.labels_[i]])
        axs1.plot([x1,x2],[y1,y1], '-', lw=0.3,
                  color='%s'%kelly_colors_hex[km.labels_[i]])    
        axs1.plot([x2,x2],[y1,y2], '-', lw=0.3,
                  color='%s'%kelly_colors_hex[km.labels_[i]])    
        axs1.plot([x1,x2],[y2,y2], '-', lw=0.3,
                  color='%s'%kelly_colors_hex[km.labels_[i]])    
        axs1.plot([x1,x1],[y2,y1], '-', lw=0.3,
                  color='%s'%kelly_colors_hex[km.labels_[i]])
        R = np.sum(img[y1:y2,x1:x2,0].astype(float))
        G = np.sum(img[y1:y2,x1:x2,1].astype(float))
        B = np.sum(img[y1:y2,x1:x2,2].astype(float))
        norm = np.array([R,G,B]).max()
        if norm>0 and not np.isnan(norm):
            #print (norm, R, G, B, R/norm, G/norm, B/norm)
            axs0.plot(int(cc[0]),int(cc[1]), 'o',
                      color=(R/norm, G/norm, B/norm), ms=2, alpha=0.8)    
            
            #print (km.labels_[i], '{0:.2f} {1:.2f} {2:.2f}'.format(R/norm, G/norm, B/norm))
            #axs2.scatter(km.labels_[i], sum(img[y1:y2,x1:x2,:]), s=60, 
           #         color = (R/norm, G/norm, B/norm), alpha=0.8)  
           
    axs.set_title("stack", fontsize=10)
    axs0.set_title("windows by RBG color", fontsize=10) 
    axs1.set_title("windows by K-means cluster", fontsize=10)
    axs.axis('off')
    axs0.axis('off')
    axs1.axis('off')    

    figname = outdir + "/" + fnameroot + "_windows.pdf"
    #pl.show()
    fig.savefig(figname)
    pl.close(fig)
    #fig2.savefig(filepattern + "_kmclusters_brightness.png")
    
    phi0 = 0
    ntot = 0
    for i in goodkmeanlabels:#[1,3,4,6,8,9]:#[1,2,3,4,7]: 
       ntot += sum(km.labels_ == i)
    print ("Total number of lights with sine behavior: %d (fraction: %.2f)"%(ntot, np.float(ntot)/lmax))
    if ntot == 0:
       print ("   !!!No windows with the right time behavior!!!")
       return (-1)
    #sys.exit()

    
    #print (km_freqs)
    kmresult = {'timeseries':timeseries,
               'transformed':transformed,
            'km':km, 'km_freqs':km_freqs,
            'goodkmeanlabels':goodkmeanlabels,
            'ntot':ntot}

    pkl.dump(kmresult, open(kmresultfile, "wb")) 

    return kmresult, transformed

def plotPC12plane(PCAr, srtindx, color=None, htmlout=None, phase=None, freq=None):

    fig = pl.figure()
    #print (color)
    rmin = np.sqrt(PCAr[srtindx[-1]][0])
    print ("minimum radius on the PC1 PC2 plane: %.2f" %rmin)
    pl.axes().set_aspect('equal')
    circle1 = pl.Circle((0,0), rmin, color='k', fill=False)
    circle2 = pl.Circle((0,0), 1, color='k', fill=False)
    #pl.plot(PCAr[:,1]/np.sqrt(PCAr[:,3]), PCAr[:,2]/np.sqrt(PCAr[:,3]), 'o',
    #     alpha=0.3, color="IndianRed")
    
    fig.gca().add_artist(circle1)
    fig.gca().add_artist(circle2)
    #replotting the selected lcvs to reduce their transparency
    pl.plot((PCAr[:,1]/np.sqrt(PCAr[:,3]))[PCAr[:,0]<rmin**2],
         (PCAr[:,2]/np.sqrt(PCAr[:,3]))[PCAr[:,0]<rmin**2],
         'o', alpha=0.3, color="grey")
    if color is None:
        pl.plot((PCAr[:,1]/np.sqrt(PCAr[:,3]))[PCAr[:,0]>rmin**2],
                (PCAr[:,2]/np.sqrt(PCAr[:,3]))[PCAr[:,0]>rmin**2],
                'o', alpha=0.6, color="IndianRed")
    else:
        colornorm = (color.astype(float) - color.min())/color.max()    
        #print (pl.cm.jet((color[srtindx][PCAr[:,0][srtindx]>rmin**2])/color.max()))
        pl.scatter((PCAr[:,1][srtindx]/np.sqrt(PCAr[:,3][srtindx]))[PCAr[:,0][srtindx]>rmin**2],
                (PCAr[:,2][srtindx]/np.sqrt(PCAr[:,3][srtindx]))[PCAr[:,0][srtindx]>rmin**2],
                   alpha=0.6, color=pl.cm.jet((color[srtindx][PCAr[:,0][srtindx]>rmin**2])/color.max()))
        
    pl.plot([0,0], [-1,1],'k', lw=0.5)
    pl.plot([-1,1], [0,0],'k', lw=0.5)
    pl.xlim(-1.1,1.1)
    pl.ylim(-1.1,1.1)    
    pl.text(0.62, -0.85, r"$R_\mathrm{min}$=%.2f"%rmin)
    pl.ylabel(r"$\mathrm{PC}_2$")
    pl.xlabel(r"$\mathrm{PC}_1$")

    if not htmlout is None and not color is None:
        print ("making Bokeh plot")
        from bokeh.plotting import Figure as figure
        from bokeh.plotting import save as save
        from bokeh.plotting import show
        from bokeh.models import ColumnDataSource, HoverTool, HBox, VBoxForm, BoxSelectTool, TapTool
        from bokeh.models.widgets import Slider, Select, TextInput
        from bokeh.io import curdoc, output_notebook
        from bokeh.plotting import output_file

        output_file(htmlout)
        print (htmlout)
        TOOLS = [HoverTool()]
        colors = ["#%02x%02x%02x" % (int(r*255), int((1-r)*255), 150) \
                  for r in colornorm[srtindx][PCAr[:,0][srtindx]>rmin**2]]
        
        source = ColumnDataSource(
            data = dict(
                id = srtindx,
                x= (PCAr[:,1][srtindx]/np.sqrt(PCAr[:,3][srtindx]))[PCAr[:,0][srtindx]>rmin**2],
                y= (PCAr[:,2][srtindx]/np.sqrt(PCAr[:,3][srtindx]))[PCAr[:,0][srtindx]>rmin**2],
                color = colors, #(color[srtindx][PCAr[:,0][srtindx]>rmin**2]),
                phase = phase[srtindx][PCAr[:,0][srtindx]>rmin**2],
                freq = freq[srtindx][PCAr[:,0][srtindx]>rmin**2],
                label = color[srtindx][PCAr[:,0][srtindx]>rmin**2]
                ))
        hover = HoverTool(
            tooltips=[
                ("ID", "@id"),
                ("label", "@label"),
                ("phase", "@phase"),
                ("frequency", "@freq"),
            ])
        p = figure(plot_width=400, plot_height=400, title=None, tools=[hover])
        #p.circle(x=0,y=0,radius=1,fill_color="#ffffff",line_color="#000000")
        p.circle('x', 'y', size=7, source=source, color='color')
        p.xaxis.axis_label = "PC1"
        p.yaxis.axis_label = "PC2"
        save(p)
    return fig, rmin

def fftsmooth(timeseries, spacing):

    from scipy.signal import gaussian
    from scipy.ndimage import filters

    transform = np.fft.fft(timeseries)
    #print (transform.shape)
    xfreq = np.fft.fftfreq(transform.shape[0], d=spacing)  
    xtmp = np.abs( xfreq - 0.25)
    freq0 = np.where(xtmp==xtmp.min())
    g = np.exp(-((np.abs(xfreq) - xfreq[freq0[0]]) / xfreq[freq0[0]] ) ** 2.) 
    
    smoothedFFTs = transform * g
    smoothedTs = np.real(np.fft.ifft(smoothedFFTs))
    
    return [smoothedTs, transform, smoothedFFTs, xfreq]

def PCAnalysis(pca, ts, xfreq, fname,
               plotsparks = False, coords = None,
               smooth=False):
    PCAr = np.zeros((ts.shape[0],5))
    evecs = pca.components_
    eg = pca.transform(ts)
    for i in range(ts.shape[0]):
        norm = (eg[i]**2).sum()
        #print (evecs[i][0]**2+eg[i][1]**2)
        PCAr[i][0] = (eg[i][0]**2+eg[i][1]**2)/norm
        PCAr[i][1] = eg[i][0]
        PCAr[i][2] = eg[i][1]
        PCAr[i][3] = norm
    PCAr[:,4] = (np.arctan2(PCAr[:,1], PCAr[:,2])+np.pi)/np.pi
    nmax = ts.shape[1]
    np.save(fname.replace("result.pdf","amplitudes.npy"), PCAr)


    #plotting
    fig = pl.figure()
    ax1 = fig.add_subplot(212)
    pl.plot(evecs[0], label=r"PC_1")
    pl.plot(evecs[1], label=r"PC_2")
    pl.legend(loc = 6, ncol=2)
    ax2 = fig.add_subplot(211)
    pl.hist(np.log10(np.sqrt(PCAr[:,0])), bins=30, cumulative=True)
    ax2.set_xlabel(r"$r  = \sqrt{PC_1^2+PC_2^2}$")
    ax2.set_ylabel("Number of lightcurves \n(cumulative)")
    #ax2.set_yscale('log')
    #ax2.set_yticks([0,50,100,500,1000,1500])

    fig.savefig(fname)

    fig = pl.figure()
    ax1 = fig.add_subplot(212)
    pl.plot(evecs[0], label=r"PC_1")
    pl.plot(evecs[1], label=r"PC_2")
    pl.legend(loc = 6, ncol=2)
    ax2 = fig.add_subplot(211)
    hist, bins = pl.histogram(np.sqrt(PCAr[:,0]), bins=30)
    pl.bar(bins[:-1], hist/(bins[:-1]+np.diff(bins))**2,
           np.diff(bins), color=['SteelBlue'], alpha=1.0)    
    ax2.set_xlabel(r"$r  = \sqrt{PC_1^2+PC_2^2}$")
    ax2.set_ylabel(r"Number of lightcurves / $r^2$")
    #ax2.set_yscale('log')
    #ax2.set_yticks([0,50,100,500,1000,1500])
    fig.savefig(fname.replace(".pdf","2.pdf"))
    
    pcasklfig = pl.figure(figsize = (10,60))
    pcax = []
    transformed = []
    maxlcv = int(ts.shape[0]*0.1)
    print ("Exploring %d lightcurves ... "%maxlcv)
    srtindx = np.argsort(PCAr[:,0])[-1:-maxlcv:-1]
    if plotsparks:
        for i,ii in enumerate(srtindx):
        #print (ii)
            pcax.append(pcasklfig.add_subplot(maxlcv+1,2,(i*2+1)))
            gifpcaspkl = pl.figure(figsize=(10,3))
            gax1 = gifpcaspkl.add_subplot(121)
            sparklines(ts[ii],
                       "%d %.2f"%(i, np.sqrt(PCAr[ii][0])), gax1,
                       x=np.arange(nmax)*0.25, nolabel=True)
            
            sparklines(ts[ii],
                       "%d %.2f"%(i, np.sqrt(PCAr[ii][0])), pcax[-1],
                       x=np.arange(nmax)*0.25, nolabel=True)
                 
            pcax.append(pcasklfig.add_subplot(maxlcv+1,2,(i*2+2)))
            transformed.append(np.abs(np.fft.rfft(ts[ii]))[1:])
            
            transformed[-1][0] = 0
            pca_xmax, pca_xmin = sparklines(transformed[i][:-5], "",
                                            pcax[-1],
                                    x=xfreq[1:-5], maxminloc=True)
            sparklines(PCAr[ii,1] * pca.components_[0] +
                       PCAr[ii,2] * pca.components_[1],
                       "", pcax[-2], x=np.arange(nmax)*0.25,
                       color='red', nolabel=True)
            sparklines(wave(np.arange(options.nmax) * options.sample_spacing,
                            np.arctan(PCAr[ii,1]/PCAr[ii,2]),
                            pca_xmax[0]+FQOFFSET),
                       "", pcax[-2],
                       x=np.arange(nmax)*0.25,
                       color='blue', nolabel=True)
            gax2 = gifpcaspkl.add_subplot(122)
            pca_xmax, pca_xmin = sparklines(transformed[i][:-5]/\
                                            transformed[i][:-5].max(),
                                            "",
                                            gax2,
                                    x=xfreq[1:-5], maxminloc=True)
            sparklines(PCAr[ii,1] * pca.components_[0] +
                       PCAr[ii,2] * pca.components_[1],
                       "", gax1, x=np.arange(nmax)*0.25,
                       color='red', nolabel=True)
            gifpcaspkl.savefig(fname.replace(".pdf",
                                             "_sparklines_%04d.png"%i))
            pl.close()
        pcasklfig.savefig(fname.replace(".pdf","_sparklines.pdf"))
        pl.close(pcasklfig)
    else:
        transformed = np.array([np.abs(np.fft.rfft(tsi))[1:] for tsi in ts])
            
        transformed[:,0] = 0
            

    fig, rmin = plotPC12plane(PCAr, srtindx)
    
    fig.savefig(fname.replace(".pdf","_PC1PC2plane.pdf"))

    print ("saving ",fname, fname.replace(".pdf","_PC1PC2plane.pdf"))
    #sys.exit()

    return PCAr, srtindx, np.array(transformed)

def tryrunit((arg, options)):
    try: 
        res = runit((arg, options))
        return res
    except: 
        print ("One failed run...")
        return None

def runit((arg, options)):
    filepattern = arg
    impath = os.getenv("UIdata") + filepattern
    print ("\n\nUsing image path: %s\n\n"%impath)

    try:
        img = glob.glob(impath+"*0000.raw")[0]
    except IndexError:
        print ("no files in path %s: check your fileroot name"%(impath+"*0000.raw"))
    fnameroot = filepattern.split('/')[-1]


    if options.stack:
        print ("(There is a stack", options.stack,")")
        stack = np.load(options.stack)
        imsize  = findsize(stack,
                           filepattern=options.stack.replace('.npy','.txt'))
    else:
        imsize  = findsize(img, filepattern=filepattern)
        stack = np.fromfile(img,dtype=np.uint8).reshape(imsize['nrows'],
                                                       imsize['ncols'],
                                                       imsize['nbands'])
            
    
    print ("Image size: ", imsize)
        
    flist = sorted(glob.glob(impath+"*.raw"))

    print ("Total number of image files: %d"%len(flist))

    nmax = min(options.nmax, len(flist)-options.skipfiles)
    print ("Number of timestamps (files): %d"%(nmax))
    if nmax<30: return 0
    flist = flist[options.skipfiles:nmax+options.skipfiles]    

    if options.coordfile:
        print ("Using coordinates file", options.coordfile)
        try:
            allights = np.load(options.coordfile)
        except:
            print ("you need to create the window mask, you can use windowFinder.py")
            print (options.coordfile)
            return (-1)
    elif os.path.isfile(filepattern+"_allights.npy"):
        try:
            allights = np.load(filepattern+"_allights.npy")
        except:
            print ("you need to create the window mask, you can use windowFinder.py")
            print (filepattern+"_allights.npy")
            return (-1)
    else:
        print ("you need to create the window mask, you can use windowFinder.py")
        print (filepattern+"_allights.npy")
        return (-1)
        
        
    lmax = len(allights)
    if options.lmax: lmax = min([lmax, options.lmax])
    print ("Max number of windows to use: %d"%lmax)
    outdir0 = '/'.join(filepattern.split('/')[:-1])+'/N%04dS%04d'%(nmax,  options.skipfiles)
    outdir = '/'.join(filepattern.split('/')[:-1])+'/N%04dW%04dS%04d'%(nmax,
                                                                       lmax,
                                                                       options.skipfiles)
    if not os.path.isdir(outdir):
        subprocess.Popen('mkdir -p %s '%outdir, shell=True)
        #os.system('mkdir -p %s '%outdir)
    if not os.path.isdir(outdir0):
        subprocess.Popen('mkdir -p %s/%s'%(outdir0,'pickles'), shell=True)
        #os.system('mkdir -p %s'%outdir0+"/pickles")
        subprocess.Popen('mkdir -p %s/%s'%(outdir0,'gifs'), shell=True)
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
    xfreq = np.fft.rfftfreq(nmax, d=options.sample_spacing)    
    if options.fft:
        bsoutfile = outdir + "/" + fnameroot + "_bs_fft.npy"
        coordsoutfile = outdir + "/" + fnameroot + "_coords_fft.npy"
        goodcoordsoutfile = outdir + "/" + fnameroot + "_goodcoords_fft.npy"                
        kmresultfile = outdir + "/" + fnameroot + "_kmresult_fft.pkl"
        pcaresultfile = outdir + "/" + fnameroot + "_PCAresult_fft.pkl"        

        
    else:
        bsoutfile = outdir + "/" + fnameroot + "_bs.npy"
        coordsoutfile = outdir + "/" + fnameroot + "_coords.npy"
        goodcoordsoutfile = outdir + "/" + fnameroot + "_goodcoords.npy"
        if options.mcmc:
            goodcoordsoutfile = outdir + "/" + fnameroot + "_goodcoords_mcmc.npy"
        kmresultfile = outdir + "/" + fnameroot + "_kmresult.pkl"
        pcaresultfile = outdir + "/" + fnameroot + "_PCAresult.pkl"        
        if options.smooth:
            goodcoordsoutfile = outdir + "/" + fnameroot + "Smooth_goodcoords.npy"
            if options.mcmc:
                goodcoordsoutfile = outdir + "/" + fnameroot + "Smooth_goodcoords_mcmc.npy"
            pcaresultfile=pcaresultfile.replace("PCA","PCAsmooth")
    figspk = pl.figure(figsize = (10,(int(lmax/2)+1)))
    ax = []
    figspkfft = pl.figure(figsize = (10,(int(lmax/2)+1)))
    axfft = []
    
    bs = np.zeros((lmax, nmax))
    fs = np.zeros((lmax, int(nmax/2+1)))
    badindx = []


    print ("")

    if READ and os.path.isfile(bsoutfile) and \
       os.path.isfile(coordsoutfile):
        print ("reading old data file", bsoutfile)
        bs = np.load(bsoutfile)
        allights = np.load(coordsoutfile)
    else:
        #this is tricky: parallelize the extraction only if you are not
        #parallelizing the image sequencies (i.e. nbursts=1)
        if options.nps > 1 and options.nbursts == 1 and not NOPARALLEL:
            print ("\n\n\nrunning on %d threads\n\n\n"%options.nps)
            pool = mpc.Pool(processes=options.nps)
            primargs = np.array(zip(allights[:lmax], range(lmax)))
            #print (primargs)
            #sys.exit()
            secondargs = [xfreq, lmax, filepattern,
                          imsize, outdir0, options, figspk, ax, figspkfft, axfft]
            pool.map(extraction, (itertools.izip(primargs, itertools.repeat(secondargs))))
            return (-1)
        else:
            for i,cc in enumerate(allights[:lmax]):
                print (cc)
                print ("\r### Extracting window {0:d} of {1:d} (x={2:d}, y={3:d})"\
                       .format(i+1, lmax, int(cc[0]),  int(cc[1])), end="")
                sys.stdout.flush()
                
                bs[i], fs[i] = extraction(((cc, i), xfreq, lmax,
                                          flist, filepattern,
                                          imsize, outdir0,
                                          options, figspk, ax,
                                          figspkfft, axfft))
                
                ax.append(figspk.add_subplot(lmax/2+1,2,i+1))

                '''
                bs[i], fs[i]  = get_plot_lca ((int(cc[0]+0.5)-options.aperture,
                                          int(cc[1]+0.5)-options.aperture, 
                                          int(cc[0]+0.5)+options.aperture+1,
                                          int(cc[1]+0.5)+options.aperture+1),
                                          flist, 
                                          filepattern+'_x%d_y%d_ap%d'%(int(cc[0]+0.5),
                                                                       int(cc[1]+0.5), 
                                                                       options.aperture),
                                          imsize, fft=options.fft,
                                          verbose=False, showme=options.showme,
                                          outdir = outdir0)

                '''
                sparklines(bs[i], '%d %d:%d'%(i, int(cc[0]), int(cc[1])), ax[i])
                if options.fft:
                    axfft.append(figspkfft.add_subplot(lmax/2+1,2,i+1))                
                    sparklines(fs[i][2:-5], '%d %d:%d'%(i, int(cc[0]), int(cc[1])),
                               axfft[-1], x=xfreq[2:-5],
                               maxminloc=options.fft)

        print ("")

        ax[0].text (0.2, 1.2, '{0:1d} seconds           {1:2s}'\
                    .format(len(bs[0])*4,'min/max'), 
                    transform = ax[0].transAxes, fontsize=15)


        ax[1].text (0.2, 1.2, '{0:1d} seconds           {1:2s}'\
                    .format(len(bs[0])*4,'min/max'), 
                transform = ax[1].transAxes, fontsize=15)

        figname = outdir + "/" + fnameroot + "_sparklines_lcv.pdf"
        try: 
            figspk.savefig(figname)
        except ValueError:
            print ("could not save", figname)
        pl.close(figspk)   

        if options.fft:
            figname = outdir + "/" + fnameroot + "_sparklinesfft_lcv.pdf"
            try: 
                figspkfft.savefig(figname)
            except ValueError:
                print ("could not save", figname)
        pl.close(figspkfft)   
            
        if options.showme: pl.show()

        badindx = []
        for i,bsi in enumerate(bs):
            if np.isnan(bsi).any():
                badindx.append(i)

        print ("\n### Removing bad indices if any:", badindx, len(allights))            
        if len(badindx)>0:
            bsnew = np.delete(bs, badindx, 0)
            bs = bsnew

            allightsnew = np.delete(allights[:lmax],  badindx, 0)
            allights = allightsnew[:lmax-len(badindx)]
            fsnew = np.delete(fs, badindx, 0)
            fs = fsnew

        else: allights = allights[:lmax]
        np.save(coordsoutfile, allights)
        np.save(bsoutfile, bs)
        
    print ("Windows now:", len(bs))
    img = np.fromfile(flist[0],dtype=np.uint8).reshape(imsize['nrows'],
                                                       imsize['ncols'],
                                                       imsize['nbands'])
 
    if options.readKM and os.path.isfile(kmresultfile) and not options.PCAnalysis:
        try: 
            print ("reading KM result from saved file %s"%kmresultfile)
            kmresult = pkl.load(open(kmresultfile))
        except ValueError:
            print ("wrong pickle protocol. remove ", kmresultfile)
            return (-1)

    elif options.readPCA and os.path.isfile(pcaresultfile):
         pca = pkl.load(open(pcaresultfile))
         if not options.fft:
             timeseries = bs
             x = np.arange(timeseries.shape[0])
             if options.smooth:
                 timeseries_smooth = np.array([fftsmooth(trs, options.sample_spacing) for trs in timeseries])[:,0]

         else:
             timeseries = fs
             
    else: 
         print ("\n### Starting PCA")

         if not options.fft:
             timeseries = bs
             x = np.arange(timeseries.shape[0])
         else:
             timeseries = fs

         lts = len(timeseries)

         pca = PCA()

         if options.smooth:
             print ("smoothing lightcurves")
             timeseries_smooth = np.array([fftsmooth(trs, options.sample_spacing) for trs in timeseries])
             fftsmoothfig = pl.figure(figsize=(10,40))            
             for ii,jj in enumerate(np.random.randint(0, high=trs.shape[0], size=30)):
                 ax = fftsmoothfig.add_subplot(30,2,ii*2+1)
                 sparklines(timeseries[jj], "", ax,
                            x=np.arange(timeseries[0].shape[0]))
                 sparklines(timeseries_smooth[jj][0], "", ax,
                            x=np.arange(timeseries[0].shape[0]),
                            color="IndianRed", alpha=0.5)
                 ax = fftsmoothfig.add_subplot(30,2,ii*2+2)
                 sparklines(timeseries_smooth[jj][1]\
                                             [:timeseries[0].shape[0]/2],
                                             "", ax,
                                             x=timeseries_smooth[jj][3]\
                                             [:timeseries[0].shape[0]/2])
                 sparklines(timeseries_smooth[jj][2]\
                            [:timeseries[0].shape[0]/2],
                            "", ax,
                            x=timeseries_smooth[jj][3]\
                            [:timeseries[0].shape[0]/2],
                            color="IndianRed", alpha=0.5)                 
                 ax.plot([0.25,0.25],ax.get_ylim(), color='k')
             fftsmoothfig.savefig(outdir + "/" + fnameroot + "_sparklinesSmooth_lcv.pdf")

             timeseries_smooth = timeseries_smooth[:,0]
             
             X_pca = pca.fit_transform(timeseries_smooth)
             pca.fit(timeseries_smooth)
             
         else:             
             X_pca = pca.fit_transform(timeseries)
             pca.fit(timeseries)

         evals = pca.explained_variance_ratio_
         evals_cs = evals.cumsum()
         evecs = pca.components_

         figrows = min(60,(int(lts/2)+1))
         figpca = pl.figure(figsize = (10, figrows))
         spax = []
         
         if options.fft:
             for i,Xc in enumerate(evecs):
                 spax.append(figpca.add_subplot(figrows, 2, i+1))
                 sparklines(Xc, "%.3f"%evals_cs[i], spax[i],
                            x=xfreq, maxminloc=options.fft)
                 if evals_cs[i] > 0.9 or i+1 == figrows * 2:
                     break
         else:
             for i,Xc in enumerate(evecs):
                 spax.append(figpca.add_subplot(figrows, 2, i+1))
                 sparklines(Xc, "%.3f"%evals_cs[i], spax[i], twomax=False,
                            maxminloc=options.fft)
                 if evals_cs[i] > 0.9 or i+1 == figrows * 2:
                     break
                
             
         
         pkl.dump(pca, open(pcaresultfile, "wb"))
         

         print ("Number of PCA component to reconstruct 90% signal: {0}".format(i+1))
         
         if not options.fft:
             if not options.smooth:
                 figname = outdir + "/" + fnameroot + "_PCA.pdf"
             else : figname = outdir + "/" + fnameroot + "_PCAsmooth.pdf"
         else: figname = outdir + "/" + fnameroot + "_PCA_fft.pdf"

         figpca.savefig(figname)
         
    if options.PCAnalysis:
        if options.smooth:
            
            PCAr, pcaindx, tansformed = PCAnalysis(pca,
                                                   timeseries_smooth, xfreq,
                                                   pcaresultfile.replace(".pkl",".pdf"),
                                                   smooth=options.smooth)
        else: 
            PCAr, pcaindx, tansformed  = PCAnalysis(pca,
                                                    timeseries, xfreq,
                                                    pcaresultfile.replace(".pkl",".pdf"),
                                                    smooth=options.smooth)
            
        freqs = []
        fftpcacomp0 = np.abs(np.fft.rfft(pca.components_[0]))
        fftpcacomp1 = np.abs(np.fft.rfft(pca.components_[1]))
        ymax = max(fftpcacomp0[2:-5])
        xmaxind = np.where(fftpcacomp0[2:-5] == ymax)[0]
        freqs.append(xfreq[2:][xmaxind][0])
        ymax = max(fftpcacomp1[2:-5])
        xmaxind = np.where(fftpcacomp1[2:-5] == ymax)[0]
        freqs.append(xfreq[2:][xmaxind][0])
        #add an offset to the requency to have the chance to test 2 freqs
        if freqs[0] == freqs[1]:
            freqs[0] = (freqs[1] if freqs[0] == freqs[1] + FQOFFSET else  freqs[1] + FQOFFSET)
        #freqs[0]=0.3
            #if freqs[0] > 0.25:
            #    freqs[0] -= FQOFFSET
            #else:
            #freqs[0] += FQOFFSET
        if options.smooth:       
            goodphases, phases = fit_waves(filepattern, lmax, nmax,
                                           timeseries_smooth[pcaindx],
                                           tansformed, len(pcaindx),
                                           img, fnameroot, freqs, stack,
                                           bs[pcaindx], logfile, allights,
                                           options.sample_spacing,
                                           options.aperture, imsize, flist,
                                           xfreq, options.chi2,
                                           phi0=PCAr[pcaindx,4],
                                           iteratit=True,
                                           mcmc=options.mcmc,
                                           fold=options.folding,
                                           fft=False,   showme=False,
                                           km = None, goodlabels = pcaindx,
                                           outdir=outdir, outdir0=outdir0)
            plotstats(goodphases, outdir + "/" + fnameroot + "_statsSmooth.pdf", PCAr)
        else:
            #if options.mcmc:
                #subprocess.Popen('rm %s '%('/'.join(filepattern.split('/')[:-1])+"/triangles/*"), shell=True)
            goodphases, phases = fit_waves(filepattern, lmax, nmax,
                                           timeseries[pcaindx],
                                           tansformed, len(pcaindx),
                                           img, fnameroot, freqs, stack,
                                           bs[pcaindx], logfile, allights,
                                           options.sample_spacing,
                                           options.aperture, imsize, flist,
                                           xfreq, options.chi2,
                                           phi0=PCAr[pcaindx,4],
                                           iteratit=True,
                                           mcmc=options.mcmc,
                                           fold=options.folding,
                                           fft=False,   showme=False,
                                           km = None, goodlabels = pcaindx,
                                           outdir=outdir, outdir0=outdir0)


            plotstats(goodphases, outdir + "/" + fnameroot + "_stats.pdf", PCAr)
        
    else:
        kmresult = KMAnalysis(evecs, timeseries, xfreq, allights, stack, img, fnameroot, options)
        
        goodphases, phases, tansformed = fit_waves(filepattern, lmax, nmax,
                                                   kmresult['timeseries'],
                                                   kmresult['transformed'], 
                                                   kmresult['ntot'],
                                                   img, fnameroot,
                                                   kmresult['km_freqs'],
                                                   stack, bs, logfile,
                                                   allights,
                                                   options.sample_spacing,
                                                   options.aperture, imsize,
                                                   flist, xfreq, options.chi2,
                                                   fft=options.fft,
                                                   fold=options.folding,
                                                   mcmc=options.mcmc,
                                                   showme=options.showme,
                                                   km = kmresult['km'],
                                                   goodlabels = kmresult['goodkmeanlabels'],
                                                   outdir=outdir, outdir0=outdir0)

    try:
        if goodphases == [-1] and phases==[-1]:
            print ("the whole thing failed...")
            return -1
    except:
        pass

    np.save(goodcoordsoutfile, goodphases)
    
    ax = pl.figure().add_subplot(111)
    try:
        for i in set(phases['index'][~np.isnan(phases['phase'])]):
            pl.plot(i, np.mean(phases['phase'][phases['index']==i]/np.pi), 
                    'o', ms=sum(phases['index']==i)*5, alpha=0.3)
            pl.plot(phases['index'], phases['phase']/np.pi, 'o')
            ax.set_xlim(-0.5,9.5)
            ax.set_xlabel("K-means cluster")
            ax.set_ylabel(r"phase ($\pi$)")
    except TypeError: pass
    if not options.PCAnalysis:
        if options.fft: figname = outdir + "/" + fnameroot + "_km_phases_fft.png"
        else:  figname = outdir + "/" + fnameroot + "_km_phases.png"
    
        try:
            pl.savefig(figname)
        except ValueError: 
            print ("could not save figure %s"%figname)    
            pl.close()
    
    ax = pl.figure().add_subplot(111)
    cmap =  pl.cm.rainbow(phases['phase'][~np.isnan(phases['phase'])])
    #print (len(cmap), len(phases[0][~np.isnan(phases[0])]), cmap)
    
    #cmap = cm.rainbow
    #cs1 = [colors[i] for i in range(len(phases[0]))] 
    #for p in range(len(phases[0])):
    #    pl.scatter(phases[2][p], phases[3][p], color = cmap(len(phases[0]) / float(phases[0][p])))
    pl.imshow(stack,  interpolation='nearest')
    pl.scatter(phases['x'][~np.isnan(phases['phase'])], phases['y'][~np.isnan(phases['phase'])], color = cmap)
    ax.get_axes().set_aspect('equal', 'datalim')
    
    
    if not options.fft:
        pl.savefig(outdir + "/" + fnameroot + "_phases.png")
        if options.smooth: pl.savefig(outdir + "/" + fnameroot + "_phasesSmooth.png")
    else: pl.savefig(outdir + "/" + fnameroot + "_phases_fft.png")
    #ax.axis('off')
    #pl.show()
    pl.close()
    return bs


if __name__=='__main__':

    parser = optparse.OptionParser(usage="getallcv.py 'filepattern' ",
                                   conflict_handler="resolve")
    parser.add_option('--nmax', default=100, type="int",
                      help='number of images to process (i.e. timestamps)')
    parser.add_option('--lmax', default=None, type="int",
                      help='number of lights')
    parser.add_option('--showme', default=False, action="store_true",
                      help='show plots')
    parser.add_option('--aperture', default=2, type="int",
                      help="window extraction aperture (1/2 side)")
    parser.add_option('--chi2', default=1.0, type="float",
                  help="chi square threshold")    
    parser.add_option('--nkmc', default=10, type="int",
                      help='number of KMeans clusters')
    parser.add_option('--sample_spacing', default=0.25, type="float",
                      help="camera sample spacing (inverse of sample rate)")
    parser.add_option('--coordfile', default=None, type="str",
                      help='coordinates python array (generated by windowFinder.py)')
    parser.add_option('--stack', default=None, type="str",
                      help='stack python array')
    parser.add_option('--nps', default=None, type="int",
                      help='number of cores to use')
    parser.add_option('--nbursts', default=None, type="int",
                      help='number of bursts. leave this alone: it will be set t to the number of arguments passed')        
    parser.add_option('--skipfiles', default=150, type="int",
                      help="number of files to skip at the beginning")
    parser.add_option('--readKM', default=False, action="store_true",
                      help="rereading old KM result instead of redoing it")
    parser.add_option('--PCAnalysis', default=True, action="store_false",
                      help="PCA analysis only")  
    parser.add_option('--readPCA', default=False, action="store_true",
                      help="rereading old PCA result instead of redoing it")
    parser.add_option('--smooth', default=False, action="store_true",
                      help='smooth time series outsied of 0.2-0.31Hz')
    parser.add_option('--fft', default=False, action="store_true",
                      help='custer in fourier space')
    parser.add_option('--mcmc', default=False, action="store_true",
                      help='mcmc to get uncertainties')
    parser.add_option('--folding', default=False, action="store_true",
                      help='folding time series')
    
    options,  args = parser.parse_args()
    #options.lmax=500
    options.coordfil="coordfile stacks/groundtest1/ESB_c0.7Hz_250ms_2016-05-24-230354_N20_coords.npy"
    
    print ("options", options)
    print ("args", args, args[0])
    if options.fft and options.smooth:
        print ("smoothing is implemented only for natural space analysis, not fft")
        sys.exit(0)
    if not options.PCAnalysis :
        print ("km analysis is obsolete!")
        
    if len(args) < 1:
        sys.argv.append('--help')
        options,  args = parser.parse_args()
           
        sys.exit(0)
    if not options.nbursts:
        options.nbursts = len(args)
        
    if not options.nps:
        options.nps = mpc.cpu_count()-1 or 1
    
    if options.nps > 1 and options.nbursts > 1 and not NOPARALLEL:
        print ("\n\n\nrunning on %d threads\n\n\n"%options.nps)
        pool = mpc.Pool(processes=options.nps)
        print (type(itertools.izip(args, itertools.repeat(options))))
        print (pool.map(tryrunit, (itertools.izip(args, itertools.repeat(options)))))
    else:
        #import time
        #start_time = time.time()
        for arg in args:
            bs = runit((arg, options))
    
        #print("--- %s seconds ---" % (time.time() - start_time))
