##this is a modification of greg dobler's code https://github.com/gdobler/grippy/blob/master/grippy/gen_cmap.py
###

import numpy as np
from scipy import interpolate
import matplotlib.colors as colors
import matplotlib.pyplot as plt


def hyss_gen_cmap(clrs=None,
                  gam=1.0, piv=None, see=False):

    if clrs is None:
        clrs = ['orange','darkorange','magenta', 'purple','lime','olive','dodgerblue','lightblue']

    # -- utilities
    lam = np.arange(256)/255.0


    # -- set up rgb for interpolation
    rgb = zip(*[colors.colorConverter.to_rgb(clr) for clr in clrs])
    piv = piv if piv else np.linspace(0.0,1.0,len(clrs))**gam


    # -- interpolate
    r,g,b = [np.interp(lam,piv,i) for i in rgb]


    # -- generate color map
    cdict   = {'red':zip(lam,r,r),'green':zip(lam,g,g),'blue':zip(lam,b,b)}
    my_cmap = colors.LinearSegmentedColormap('dum',cdict,256)


    # -- plot if desired
    if see:
        img = np.arange(10000).reshape(100,100)/10000.
        fig = plt.figure(0)
        fig.imshow(img,cmap=my_cmap,interpolation='nearest')
        plt.colorbar()
        plt.show()

    return my_cmap
