from __future__ import print_function

import numpy as np
import sys
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

try:
    raw_input
except NameError:
    # Python 3
    raw_input = input


class SelectFromCollection(object):
    """Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool highlights
    selected points by fading them out (i.e., reducing their alpha values).
    If your collection has alpha < 1, this tool will permanently alter them.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.

    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to `alpha_other`.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, self.Npts).reshape(self.Npts, -1)

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero([path.contains_point(xy) for xy in self.xys])[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    families = []
    data  = np.load(sys.argv[1])
    coords = np.array([np.load(sys.argv[2])[2,:], np.load(sys.argv[2])[3,:]])
    #datasum = data[:,:,:3].sum(axis=2)
    #tmp = [(j,i, datasum[i][j].astype(float)/255/3.0) for i in range(datasum.shape[0]) \
    #       for j in range(datasum.shape[1]) if datasum[i][j] > 0]
    #tmp = np.array(tmp)
    plt.ion()
    done = np.zeros_like(coords.T)
    ii = 0
    fams = 0
    #subplot_kw = dict(xlim=(0, 1), ylim=(0, 1), autoscale_on=False)
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)#subplot_kw=subplot_kw)
    while 1:
        ax.imshow(data[:,:,:3].sum(axis=2), cmap="bone")
        #subplot_kw=subplot_kw)
        ax.set_xlim(0,data.shape[1])
        ax.set_ylim(data.shape[0], 0)        
        try:
            toplot =np.array([c for c in coords.T if not c in done]).T
            if len(toplot)==0:
                break
            #print (toplot)
            ax.scatter(coords[0], coords[1], s=5, color='g')
            #pts = ax.scatter(tmp.T[0],tmp.T[1], color = tmp.T[2], alpha=0.1)
            pts = ax.scatter(toplot[0], toplot[1], s=5, color='r')
            #ax.imshow(data, cm="Grays")
            
            #pts = ax.scatter(data[:, 0], data[:, 1], s=80)
            #print (pts)
            selector = SelectFromCollection(ax, pts)
            plt.draw()
            raw_input('Press any key to accept selected points')
            print("Selected points:")
            xys = selector.xys[selector.ind]
            for xy in xys:
                done[ii]=xy
                ii+=1
            families.append(xys)
            fams+= len(families[-1])
            print(xys, len(families), fams)
            selector.disconnect()

            # Block end of script so you can check that the lasso is disconnected.
            ri = raw_input('Press q key to quit')
            if ri == 'q': break
        except IndexError: pass
        
    print (families)
    families = np.array(families)
    np.save(sys.argv[1].replace("labelledwindows.npy",
                                "families.npy"),
            families)
