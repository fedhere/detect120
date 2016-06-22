import pylab as pl
import numpy as np
import sys
import json
import os
from getalllcvPCA import plotPC12plane

import pylab as pl

s = json.load( open(os.getenv ('PUI2015')+"/fbb_matplotlibrc.json") )
pl.rcParams.update(s)


from pylab import rc
rc('axes', linewidth=1.2)
#from matplotlib.font_manager import FontProperties##
import matplotlib as mpl
params = {'font.weight' : 'normal',
          'axes.linewidth' : 2,
          'figure.autolayout': True,
          'axes.grid' : False
}

pl.rcParams.update(params)

families = np.load("stacks/ESB_c0.7Hz_250ms_2016-05-24-230354-0000_20_families.npy")
fams = {}
for i,f in enumerate(families):
    for m in f:
        fams[(int(m[0]),int(m[1]))] = i+1
PCAr = np.load("groundtest1/N0100W1533S0150/ESB_s119.75Hz_c4.00Hz_100ms_2016-05-24-%s_PCAamplitudes.npy"%sys.argv[1])
coords = np.load("groundtest1/N0100W1533S0150/ESB_s119.75Hz_c4.00Hz_100ms_2016-05-24-%s_coords.npy"%sys.argv[1])
goodphases = np.load("groundtest1/N0100W1533S0150/ESB_s119.75Hz_c4.00Hz_100ms_2016-05-24-%s_goodcoords.npy"%sys.argv[1])

colors = np.zeros(PCAr.shape[0], int)
phases = np.ones(PCAr.shape[0])*-1
freqs = np.ones(PCAr.shape[0])*-1
gxgy = np.array([(gx,gy) for gx,gy in zip(goodphases[2].astype(int),\
                                 goodphases[3].astype(int))])
#print (goodphases[2], goodphases[3])
tmp = []
for i,g in enumerate(gxgy):
    tmp.append([PCAr[i][0], g[0],g[1], PCAr[i][1]/PCAr[i][2]])
tmp=np.array(tmp)
for i in  (tmp[tmp.T.argsort(axis=1)[3]]):
    print i[0], i[1], i[2], i[3]

for i, pca in enumerate(PCAr):
    #pca and coords are in the same order, fan and goodphases are not
    if (int(coords[i][0]),int(coords[i][1])) in fams:
        colors[i] = fams[(int(coords[i][0]),int(coords[i][1]))]
        print (int(coords[i][0]),int(coords[i][1]), fams[(int(coords[i][0]),int(coords[i][1]))])
    xi = np.where(gxgy \
                  == (int(coords[i][0]),int(coords[i][1])))
    
    if len(xi[0])>1 and xi[0][0] == xi[0][1]:
        #print (i,colors[i])
        phases[i] = goodphases[0][xi[0][0]]
        freqs[i] = goodphases[6][xi[0][0]]

print (phases[phases>-1],freqs[phases>-1])
output_file = "groundtest1/N0100W1533S0150/ESB_s119.75Hz_c4.00Hz_100ms_2016-05-24-%s_PCAresult_PC1PC2plane_color.html"%sys.argv[1]


maxlcv = 153
srtindx = np.argsort(PCAr[:,0])[-1:-maxlcv:-1]

fig, rmin = plotPC12plane(PCAr, srtindx, color=colors, htmlout=output_file, phase=phases, freq=freqs)

fig.savefig("groundtest1/N0100W1533S0150/ESB_s119.75Hz_c4.00Hz_100ms_2016-05-24-%s_PCAresult_PC1PC2plane_color.pdf"%sys.argv[1])
