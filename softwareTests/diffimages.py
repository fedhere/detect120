from __future__ import print_function
import glob
import numpy as np
import matplotlib.pyplot as pl
import json
s = json.load( open("fbb_matplotlibrc.json") )
pl.rcParams.update(s)

import configstack as cfg



nmax = cfg.srcpars['nmax']
nstack = cfg.srcpars['nstack']
sample_rate = cfg.imgpars['sample_rate']
font = {'size'   : 23}





def scrollimg(data):

    def scroll(event):
        if event.key=='right':
            ind[0] += 1
        elif event.key=='left':
            ind[0] -= 1
        else:
            return
    
        ind[0] = ind[0] % data.shape[0]
        print (ind[0])
        lin.set_data(data[ind[0]])
        #lin.set_data(xx,data[ind[0]])
    
        fig.canvas.draw()
    
        return
    
    # -- set up the plot
    ind = [0]
    fig = pl.figure()
    ax = fig.add_subplot(111)

    #lin, = ax.plot(xx, data[0])
    print (data[0].median(axis=2).shape)
    lin = ax.imshow(data[0], vmin=0, vmax=10, cmap='bone',
                    aspect='auto',
                    interpolation='nearest')
    
    fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', scroll)

    return


def diffim(flist, fname, imgeom):
    nrow, ncol, nband = imgeom
    diffims = np.zeros((len(flist),nrow,ncol,nband))
    print ("here")
    i = 0
    for i in range(len(flist)-1):
        #print ("image %d"%i)
        diffims[i] = (np.fromfile(flist[0],
                                  dtype=np.uint8) -
                      np.fromfile(flist[i+1],
                      dtype=np.uint8))\
                      .reshape(nrow,ncol,nband)
    #data = np.random.rand(1000).reshape(10,100)
    #scrollimg(data)
    scrollimg(diffims)

    pl.savefig('stacks/'+fname+"_%04d_diff.png"%(i))
    pl.close()

print (cfg.imgpars['froot'])
flist = glob.glob(cfg.imgpars['froot'])


diffim(flist[:nmax], flist[0].replace('0000.raw','').replace('.raw',''),
       (cfg.imgpars['nrow'],
        cfg.imgpars['ncol'],
        cfg.imgpars['nband']))




