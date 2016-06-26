__author__ = 'fbb CUSP2016'
import pylab as pl
import numpy as np
import os
import scipy
from scipy import optimize
from scipy import signal
from scipy.ndimage.filters import median_filter
from numpy import fft
import statsmodels.api as sm
from scipy.interpolate import interp1d


def lcvSmoothing(x, y, frac=0.25):
    lowess = sm.nonparametric.lowess
    return lowess(y, x, frac=frac,
                  is_sorted=False, return_sorted=False)
    
def mynorm(flux):
    flux -= flux.mean()
    return 2 * (flux - flux.min()) / (flux.max() - flux.min())


def plot_flux_by_runtime(flux, c_freq, runtime, fftax):
    pl.figure(figsize=(20, 5))

    color = 'dodgerblue'
    pl.title('Close-up Flux: Shutter 119.75Hz; Camera ' + str(c_freq) + 'Hz; Integration 1ms', fontsize=20)
    pl.plot(runtime, flux, '-', color=color, linewidth=0.75)
    pl.plot(runtime, flux, 'o', color=color)
    # pl.xlim(0,125)

    pl.xlabel("Seconds", fontsize=15)
    pl.ylabel("Almost normalized flux", fontsize=15)
    pl.ylim(-.1, 2.1)

    #pl.figure(figsize=(20, 5))

    #freq = np.fft.rfftfreq(len(flux), d=1.0 / c_freq)

    #fftax.plot(freq, np.abs(np.fft.rfft(flux)), alpha=0.5, label='%f' % c_freq)


def plot_flux_by_imgn(flux, ax, key):
    #pl.figure(figsize=(20, 5))
    x = range(len(flux))
    color = 'SteelBlue'
    ax.set_title('Flux: Shutter 119.75Hz; Camera 4Hz; Integration 13ms %s' % key, fontsize=20)
    ax.plot(x, flux, '-', color=color, linewidth=0.75)
    ax.plot(x, flux, 'o', color=color)
    # pl.xlim(0,125)
    
    ax.set_xlabel("image number", fontsize=15)
    ax.set_ylabel("Almost normalized flux", fontsize=15)
    #pl.ylim(-.1, 2.1)

def plot_fft(flux, fftax, key):
    #pl.figure(figsize=(20, 5))
    n = len(flux)
    freq = np.fft.rfftfreq(n)

    pwr = np.abs(np.fft.rfft(flux))
    
    color = 'IndianRed'
    #fftax.set_title('Flux FT: Shutter 119.75Hz; Camera 4Hz; Integration 13ms %s' % key, fontsize=20)
    fftax.plot(freq, pwr, '-', color=color, linewidth=0.75)
    fftax.plot(freq, pwr, 'o', color=color)
    # pl.xlim(0,125)
    
    fftax.set_xlabel("freq (1.0/image number)", fontsize=15)
    fftax.set_ylabel("Power", fontsize=15)
    #pl.ylim(-.1, 2.1)

    return freq, pwr
    

def folding(flux, runtime, freq, cycles=2):
    return flux, np.mod(runtime, float(cycles) / freq)


def makelcvmodel(fl, rt, fq, folding=False):
    if folding:
        foldedf, foldedt = folding(fl, rt, fq*4.0, cycles=1)
    else:
        foldedf, foldedt = fl, rt
    #pl.figure()
    #pl.plot(foldedt, foldedf, 'ro')
    #pl.plot(foldedt, foldedf, '-', alpha=0.1)
    smoothed = lcvSmoothing(foldedt, foldedf, frac=0.2)
    #print smoothed
    if np.isnan(smoothed).all(): pass
    pl.figure()
    pl.plot(foldedt, foldedf, 'ko')
    pl.plot(foldedt, smoothed, 'k')
    #pl.show()
    return interp1d(foldedt, smoothed, kind='cubic', bounds_error=False)

def model_wave(fl, rt, freq, mag=1.0, phase=0.0, offset=0, interpt=[]):
    foldedfl, foldedt = folding(fl, rt, freq, cycles=1)
    if len(interpt) == 0: interpt = foldedt
    else:
          tmp,  interpt = folding(np.zeros(len(interpt)), interpt, freq, cycles=1)
    return  makelcvmodel(foldedfl, foldedt, freq)(interpt)


def sine_wave(time, freq, mag=1.0, phase=0.0, offset=1):
    return mag * np.sin(2 * np.pi * freq * time + phase) + offset


def minimizer(params, data, time, freq, curve):
    #mag, phase = params[0], params[1]
    mag, phase = 1.0, params[0]
    if curve :
        #print curve(data, time, freq, mag, phase)
        return data - curve(data, time, freq, mag, phase)
    return data - sine_wave(time, freq, mag, phase)


def fitphase(sine, flux, fq, curve=None):
    if curve:
        res = scipy.optimize.leastsq(minimizer, (0), #(1,0)
                                     args=(flux, range(len(flux)),
                                           fq, curve))
        
    else:
        res = scipy.optimize.leastsq(minimizer, (0), #(1,0)
                           args=(flux, range(len(flux)), fq, None))
    return res


def sigmaclip(data, factor, replacement=None, median=False, maxiter = 100):
    std = np.std(data)
    iteration=0
    if median: center = np.nanmedian(data)
    else: center = np.nanmean(data)
    if not replacement: replacement = np.nan
    elif replacement == 'mean': replacement = center
    indx = (data>(center+std*factor))+(data<(center-std*factor))
    while np.sum(indx) > 0 and iteration < maxiter:
        #print indx, np.sum(indx)
        #pl.plot(data)
        #pl.plot([0,len(data)],[center+std*factor,center+std*factor])
        #pl.plot([0,len(data)],[center-std*factor,center-std*factor])        
        data[indx] = replacement
        std = np.std(data)
        if median: center = np.nanmedian(data)
        else: center = np.nanmean(data)
        if not replacement: replacement = np.nan
        elif replacement == 'mean': replacement = center
        indx = (data>(center+std*factor))+(data<(center-std*factor))
        #print indx, np.sum(indx)
        #pl.plot(data,'ko')
        
        #pl.show()
        iteration+=1
    return data
