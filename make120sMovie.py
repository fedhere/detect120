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

from findImageSize import findsize

giflen=100
#pl.ion()

if __name__=='__main__':
    filepattern = sys.argv[1]
    impath = os.getenv("UIdata") + filepattern
    fnameroot = filepattern.split('/')[-1]
    flist = sorted(glob.glob(impath+"*1??.raw"))#[:30]
    coords = np.load("groundtest1/N0100W1533S0150/ESB_s119.75Hz_c4.00Hz_100ms_2016-05-24-215440_goodcoords.npy")
    img = flist[0]
    imsize  = findsize(img, filepattern=filepattern)
    img = np.fromfile(img,dtype=np.uint8).reshape(imsize['nrows'],
                                                       imsize['ncols'],
                                                       imsize['nbands'])
            
    
    print ("Image size: ", imsize)
    
        #imshow(np.repeat((windows == l),3).reshape(stack.shape)*stack)

    mask = np.zeros((img.shape[0], img.shape[1]))*False
    families = np.load("stacks/ESB_c0.7Hz_250ms_2016-05-24-230354-0000_20_families.npy")
    windows = np.load("stacks/ESB_c0.7Hz_250ms_2016-05-24-230354-0000_20_labels.npy")

    for f in coords.T:
        #print (mask.sum())
        l = windows[int(f[3]),int(f[2])]
        #windows == l
        mask = mask + (windows==l)
    imgmask = np.repeat(mask, 3).reshape(img.shape)
    imgs = []
    for f in flist:
        '''
        print (imgmask * np\
                  .fromfile(f,dtype=np.uint8).reshape(imsize['nrows'],
                                                      imsize['ncols'],
                                                      imsize['nbands']))
        
        pl.imshow(imgmask * np\
                  .fromfile(f,dtype=np.uint8).reshape(imsize['nrows'],
                                                      imsize['ncols'],
                                                      imsize['nbands']))
        '''
        data =  np.fromfile(f,dtype=np.uint8).reshape(imsize['nrows'],
                           imsize['ncols'],
                           imsize['nbands'])#-50.0) * 3

        #data =  np.sqrt(np.fromfile(f,dtype=np.uint8).reshape(imsize['nrows'],
        #                   imsize['ncols'],
        #                   imsize['nbands'])/255.0) * 255.
        #pl.show()
        imgs.append((imgmask * data).clip(0,255).astype(np.uint8))
        
    images = [Image.fromarray(f) for f in imgs[:giflen]]
    writeGif(filepattern + ".GIF",
             images, duration=0.01)
