#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def multiplot(xx=None,data=None):
    """
    Plot the second dimension of 2D as a line plot and scan through the 
    indices of the first dimension with the left/right buttons.
    """

    def scroll(event):
        if event.key=='right':
            ind[0] += 1
        elif event.key=='left':
            ind[0] -= 1
        else:
            return

        ind[0] = ind[0] % data.shape[0]

        lin.set_data(xx,data[ind[0]])
        mx = data[ind[0]].max()
        mn = data[ind[0]].min()
        dy = mx - mn
        yr = [mn-0.1*dy,mx+0.1*dy]
        ax.set_ylim(yr)
        txt.set_position((xr[1],yr[1]+0.02*(yr[1]-yr[0])))
        txt.set_text('{0} of {1}'.format(ind[0],data.shape[1]-1))
        fig.canvas.draw()

        return

    if data is None:
        data = xx
        xx   = np.arange(data.shape[1])


    ind = [0]

    # -- set up the plot
    fig, ax = plt.subplots()
    lin, = ax.plot(xx,data[0])
    yr   = ax.get_ylim()
    xr   = ax.get_xlim()
    txt  = ax.text(xr[1],yr[1]+0.02*(yr[1]-yr[0]),
                   '{0} of {1}'.format(ind[0],data.shape[1]-1),ha='right')
    fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', scroll)

    return


data = np.random.rand(2000).reshape(10,200)

multiplot(data)
