from __future__ import print_function
# CUSP UO 2016
__author__ = "fbb"

import numpy as np
import pylab as pl
import sys
import os
import json

""" Find an image size: nrows, ncols, for a raw image

    Args:
         img: a raw image as a numpy array
         nbands: needs to know the number of bands
         filepattern: the output file name for the json file. 
                      it will also check for an existing json file
         imsizefile: reads an image size file by another name 
                     (different than filepattern)
    Returns:
         image size dictionary upon success.
"""

OUTPUTDIR = "../outputs/"
pl.ion()


def divisorGenerator(n):
    """
    Finds  valid integer divisors of n:
    Args:
        N (number of pixels)
    Returns:
        the divisors.
    Comments:
        It also saves the image shape dictionary as a json file
        by the given name or by the name derived from the input file
    """

    large_divisors = []
    for i in range(1, int(np.sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i * i != n:
                large_divisors.append(n / i)
    for divisor in reversed(large_divisors):
        yield divisor


def findsize(imgfile, nbands = 3, filepattern = None, imsizefile = None):
    
    # if an nd.array is passed just return its shape
    if isinstance(imgfile, (np.ndarray)):
        imgsize = {}
        imgsize['nrows'], imgsize['ncols'], imgsize['nbands'] = imgfile.shape
        return imgsize

    # look for a saved file contianing the shape info. if found return that
    if imsizefile:
        print("looking for file ", OUTPUTDIR + imsizefile)
        if os.path.isfile(OUTPUTDIR + imsizefile):
            imgsize = json.load(open(OUTPUTDIR + imsizefile, 'r'))
            return imgsize

    if filepattern:
        print (OUTPUTDIR + filepattern + "_imgsize.txt")
        if os.path.isfile(OUTPUTDIR + filepattern + "_imgsize.txt"):
            imgsize = json.load(open(OUTPUTDIR + filepattern \
                                     + "_imgsize.txt", 'r'))
            return imgsize

    # if thr shape needs to be recinstructed
    # show the image with different shape assumptions and
    # wait for positive user response w raw_input

    imgsize = {}
    imgsize['nbands'] = nbands
    img = np.fromfile(imgfile, dtype = np.uint8)
    divlist = list(divisorGenerator(img.size / nbands))
    print (len(divlist))
    for i in range(int(len(divlist)/2), 0, -1):
        # print divlist[i], divlist[-i-1], divlist[i] *
        # divlist[-i-1], img.size
        pl.imshow(img.reshape(divlist[i], divlist[-i-1], nbands))
        pl.draw()
        if raw_input("is this the right size?").lower().startswith('y'):
            imgsize['nrows'] = divlist[i]
            imgsize['ncols'] = divlist[-i-1]
            if filepattern:
                json.dump(imgsize, open(OUTPUTDIR +
                                        filepattern +
                                        "_imgsize.txt", 'w'))
            return imgsize

if __name__ == '__main__':
    nbands = 3
    datapath = os.getenv("UIdata")
    img =  sys.argv[1]
    findsize(img, nbands)
