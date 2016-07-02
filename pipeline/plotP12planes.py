from __future__ import print_function
__author__ = 'fbb'
# CUSP 2016

import pylab as pl
import numpy as np
import sys
import json
import os
from getalllcvPCA import plotPC12plane


OUTPUTDIR = "../outputs/"

s = json.load( open(os.getenv ('PUI2015')+"/fbb_matplotlibrc.json") )
pl.rcParams.update(s)

'''
Calls the plotPC12plane function in getalllcvPCA.py to 
make an interactive bokeh version of the PC1PC2 plane plot. 
(http://cosmo.nyu.edu/~fb55/FBB_HTIGridDynamics/ESB_s119.75Hz_c4.00Hz_100ms_2016-05-24-PCAresult_PC1PC2plane_color.html)

Arguments: 
      runids (as many as you want): the name of the run.  
           e.g. $python plotP12plane.py 213949 215440 220941 222440 223940 
Comments: This code is designed for the runs with path 
          "groundtest1/ESB_s119.75Hz_c4.00Hz_100ms_2016-05-24-" at this time, 
           so pass just the run unique identifier. e.g.
'''

from pylab import rc
rc('axes', linewidth=1.2)
#from matplotlib.font_manager import FontProperties##
params = {'font.weight' : 'normal',
          'axes.linewidth' : 2,
          'figure.autolayout': True,
          'axes.grid' : False
}

pl.rcParams.update(params)

families = np.load(OUTPUTDIR + "stacks/groundtest1/ESB_c0.7Hz_250ms_2016-05-24-230354_N20_families.npy")
stack = np.load(OUTPUTDIR + "stacks/groundtest1/ESB_c0.7Hz_250ms_2016-05-24-230354_N20.npy")
fams = {}

# for every family of buildings
for i,f in enumerate(families):
    # for every member of the family
    for m in f:
        # set the family dictionary to the family number
        fams[(int(m[0]), int(m[1]))] = i+1

PCAr = []
goodphases = []
colors = []
srtindx = []
output_file = []
phases = []
freqs = []

for arg in sys.argv[1:]:        
    PCAr.append(np.load(OUTPUTDIR + "groundtest1/N0100W1533S0150/ESB_s119.75Hz_c4.00Hz_100ms_2016-05-24-%s_PCAamplitudes.npy"%arg))
    goodphases= np.load(OUTPUTDIR + "groundtest1/N0100W1533S0150/ESB_s119.75Hz_c4.00Hz_100ms_2016-05-24-%s_goodcoords.npy"%arg)
    print ("in file", goodphases)

    colors.append(np.zeros(PCAr[-1].shape[0], int))
    phases.append(np.ones(PCAr[-1].shape[0])*-1)
    freqs.append(np.ones(PCAr[-1].shape[0])*-1)
    
    for i, pca in enumerate(PCAr[-1]):
        #fam and goodphases are not in the same order
        if (int(pca[5]),int(pca[6])) in fams:
            colors[-1][i] = fams[(int(pca[5]),int(pca[6]))]

            xi = np.where((goodphases[2].astype(int) == int(pca[5])) *
                          (goodphases[3].astype(int) == int(pca[6])))
            if len(xi[0])>0:
                phases[-1][i] = goodphases[0][xi[0][0]]
                freqs[-1][i] = goodphases[6][xi[0][0]]

    maxlcv = 153
    srtindx.append(np.argsort(PCAr[-1][:,0])[-1:-maxlcv:-1])

output_file = OUTPUTDIR + "groundtest1/N0100W1533S0150/ESB_s119.75Hz_c4.00Hz_100ms_2016-05-24-PCAresult_PC1PC2plane_color.html"


fig, rmin = plotPC12plane(PCAr, srtindx, color=colors, htmlout=output_file, phase=phases, freq=freqs, multi=True, titles=sys.argv[1:], stack=stack)

    #fig.savefig(OUTPUTDIR + "groundtest1/N0100W1533S0150/ESB_s119.75Hz_c4.00Hz_100ms_2016-05-24-%s_PCAresult_PC1PC2plane_color.pdf"%sys.argv[1])
