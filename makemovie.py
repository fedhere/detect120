from __future__ import print_function
import glob
import numpy as np
import pylab as pl
import sys
import os
import pickle as pkl
#from gatspy.periodic import LombScargleFast
from findImageSize import findsize
import json
s = json.load( open("fbb_matplotlibrc.json") )
pl.rcParams.update(s)


impath = os.getenv("UIdata")+"/groundtest1/ESB_s119.75*" 
img = glob.glob(impath+"*0000.raw")[0]


nrow, ncol = 1530, 2444
#= findsize(img)
nband = 3
print (nrow, ncol)


print (impath)
flist=glob.glob(impath+"*213949*.raw")
print(len(flist))
flist11975=np.array(flist)[np.argsort(flist)]
print(len(flist))
#print (flist11975)
for i,f in enumerate(flist11975[50:150]):

    fig, axs = pl.subplots(1,1,figsize=(15,15))
    axs.axis('off')

    axs.imshow(np.fromfile(f,dtype=np.uint8).reshape(nrow,ncol,nband),  interpolation='none')
    axs.set_xlim(0, axs.get_xlim()[1])
    axs.set_ylim(axs.get_ylim()[0], 0)
    pl.savefig("%s_%02d.png"%(impath[:-1],i),bbox_inches='tight')
 
