import pylab as pl
import numpy as np
from matplotlib.colors import rgb2hex

D2RAD = np.pi/180.
sin60 = 0.5*np.sqrt(3)
def RGBtoCartCoordXY(rgb):
    r,g,b = rgb
    rgb = r+g+b
    return 0.5/rgb*np.array([(2.0*g+b), np.sqrt(3.0)*b])


def plotrbg (c, ax = None):
    if ax is None:
        ax = pl.figure(figsize=(5,5)).add_subplot(111)

    c=c/c.sum()
    print (c)
    ax.set_xlim(0,1)
    ax.set_ylim(0,sin60)
    ax.set_aspect('equal')
    pl.plot((0,1),(0,0),'k-')
    pl.plot((0,0.5),(0,sin60),'k-')
    pl.plot((1,0.5),(0,sin60),'k-')
    x,y = RGBtoCartCoordXY(c)
    pl.plot(x, y, '.', color=rgb2hex((c[0], c[1], c[2])), alpha=0.5)

    print (x,y)

ax = pl.figure(figsize=(5,5)).add_subplot(111)

for i in np.arange(0.0,1,0.1):
    for j in np.arange(0.0,1,0.1):
        for k in np.arange(0.0,1,0.1):
            c = np.array([i,j,k])
            if c.sum() == 0:
                continue
            plotrbg(c, ax=ax)
pl.show()
