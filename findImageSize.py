import numpy as np
import pylab as pl
import sys
import os

import json


def divisorGenerator(n):
    large_divisors = []
    for i in range(1, int(np.sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i*i != n:
                large_divisors.append(n / i)
    for divisor in reversed(large_divisors):
        yield divisor


def findsize (imgfile, nbands=3, filepattern=None, imsizefile=None):
    
    ''' if an nd.array is passed just return its shape'''
    if isinstance(imgfile, (np.ndarray)):
        imgsize = {}
        imgsize['nrows'], imgsize['ncols'], imgsize['nbands'] = imgfile.shape
        return imgsize
    ''' look for a saved file contianing the shape info. if found return that'''
    if imsizefile:
        if os.path.isfile(imsizefile):
            imgsize = json.load(open(imsizefile,'r'))
            return imgsize
        
    if filepattern:
        print (filepattern+"_imgsize.txt")        
        if os.path.isfile(filepattern+"_imgsize.txt"):
            imgsize = json.load(open(filepattern+"_imgsize.txt",'r'))
            return imgsize
    
    ''' if thr shape needs to be recinstructed 
    show the image with different shape assumptions and 
    wait for positive user response w raw_input'''

    imgsize = {}
    imgsize['nbands'] = nbands
    divlist = list(divisorGenerator(100))
    img = np.fromfile(imgfile, dtype=np.uint8)
    divlist = list(divisorGenerator(img.size/nbands))
    for i in range(len(divlist)/2,0,-1):
        #print divlist[i], divlist[-i-1], divlist[i] * divlist[-i-1], img.size
        pl.imshow(img.reshape(divlist[i], divlist[-i-1], nbands))
        pl.show()
        if raw_input("is this the right size?").lower().startswith('y'):
           
            imgsize['nrows'] = divlist[i]
            imgsize['ncols'] = divlist[-i-1]
            if filepattern :
                json.dump(imgsize, open(filepattern+"_imgsize.txt",'w'))
            return imgsize






if __name__ == '__main__':
    nbands = 3
    datapath = os.getenv("UIdata")
    img =  sys.argv[1]
          #datapath = os.getenv("UIdata") + "/loads_test/fan_s119.75Hz_c4.00Hz_13.00ms_3_2016-05-22-114721-0388.raw"
    findsize(img, nbands)
