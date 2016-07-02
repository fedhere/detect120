from __future__ import print_function
# # CUSP UO 2016
__author__ = "fbb"

'''
Fetches windows (light sources) from an image stack of a dense urban landscape.
'''

import sys
import numpy as np
import scipy as sp
import pylab as pl
import optparse
import matplotlib.gridspec as gridspec
from scipy import ndimage
from scipy import optimize
from scipy.special import gammaln

try:
    raw_input
except NameError:
    #  Python 3
    raw_input = input


def getknuth(m, data, N=None):
    '''Calculates the likelihood of the proposed bin size according to Knuth's rule

    Arguments:
        the proposed bin size, data array (1D numpy array), data siz
    Returns:
        Knuth's likelihood
    '''
    m = int(m)
    if N is None:
        N = data.size

    if m > N:
        return -1
    bins = np.linspace(min(data), max(data), int(m) + 1)
    try:
        nk, bins = np.histogram(data, bins)
        return - (N * np.log(m) + gammaln(0.5 * m) - m * gammaln(0.5) -
                  gammaln(N + 0.5 * m) + np.sum(gammaln(nk + 0.5)))
    except:
        return -1


def knuthn(data):
    '''
    Finds the right bin size according to Knuth's rule (Bayesian inference)

    Arguments:
        data array. must be a 1D numpy array
    Returns:
        optimized value for bin size according to Knuth's rule.
    Calls:
        getknuth()
        scipy.optimize.fmin()
    '''
    assert data.ndim == 1, "data must be 1D array to calculate \
                            Knuth's number of bins"
    N = data.size
    maxM = 5 * np.sqrt(N)
    m0 = 2.0 * N**(1./3.)
    gk = getknuth
    if gk == -1:
        return m0
    mkall = optimize.fmin(gk, m0, args = (data, N), maxiter = 3)
    # , maxfun=1000)# [0]
    mk = mkall[0]
    if mk > maxM or mk < 0.3 * np.sqrt(N):
        return m0
    return mk

GS = 10  # gaussian standard deviation (x and y)

if __name__ == '__main__':
    """
    Fetches windows from an image stack.

    High-pass filters the image and thresholds the residuals to
              selects windowslight cources.

    Args:
        imge in in put. should be a stack of images, with reasonable SNR,
        as a npy array.

        Optional Args:
        thr: the threshold guess to be verified
        noauto: if True the first attempted guess is thr=110,
                iif False instead  the first attempted guess is
                thr=90th percent limit of the pixel distribution
        nocheck: does not plot the image and just goes for 90th percentile.
                 otherwise you re asked to confirm the threshold
        showme: if True will show the labelled and selected window
                images before saving
    Returns:
        Nothing: saves a npy image with pixels 0'ed away from windows.
           Saves:
                all sources above threshold and the histogram in a PDF image
                (_allwindows.pdf)
                all good (>10 pixel) labelled windows
                    in a npy array _labels.npy
                an image of the labelled windows in a npy _labelledwindows.npy
                an image of the labelled windows in PDF _labelledwindows.pdf
                a npy of the coordinates (center of brightness) of
                    all selected windows
                an image of the selected windows away from edges,
                    a dot per wondows in coords PDF _selectedwindows.pdf
                a npy mask that can be ised with the orignal image
                    to make it a masked array.
                All files stored in the past of the image, from the local dir
    Comments:
        Its a bit slow, so be patient...
    """

    parser = optparse.OptionParser(usage="windowFinder 'stackpath' ",
                                   conflict_handler="resolve")

    parser.add_option('--thr', default=110, type="float",
                      help='threshold for window')

    parser.add_option('--noauto', default=True, action="store_false",
                      help='''automatically setting the threshold
                      to 98% of pixel value distribution''')

    parser.add_option('--nocheck', default=False, action="store_true",
                      help='dont check if threshold is right')

    parser.add_option('--showme', default=False, action="store_true",
                      help='show plots')

    options,  args = parser.parse_args()
    print(len(args))
    options,  args = parser.parse_args()
    print("options and arguments:", options, args)
    if len(args) < 1:
        sys.argv.append('--help')
        options,  args = parser.parse_args()

        sys.exit(0)

    '''
    thr=200
    args = [0]
    args[0] = \
    "../outputs/stacks/ESB_c0.7Hz_250ms_2016-05-24-230354-0000_20.npy"
    '''
    stack = np.load(args[0])

    # gaussian filter first
    overgaussstack = (stack.astype(float) -
                      sp.ndimage.gaussian_filter(stack.astype(float),
                                                 (GS, GS, 0)))

    # reset the limits
    foo = overgaussstack - overgaussstack.min()
    foo /= foo.max()

    foo *= 255
    flatfoo = foo.flatten()
    pcs = np.array([16, 50, 84, 90, 95, 99.5])
    pc = np.percentile(flatfoo, pcs)
    if options.noauto: thr = pc[pcs == 90]
    else: thr = options.thr

    numbins = int(knuthn(flatfoo) + 0.5)
    counts, bins = np.histogram(flatfoo, bins = numbins,
                                density = True)
    widths = np.diff(bins)
    countsnorm = counts / counts.max()
    # show the stacks
    done = options.nocheck
    if not done:
        pl.ion()
        pl.clf()
        stackfig = pl.figure()
        gs = gridspec.GridSpec(2, 2)
        ax0 = stackfig.add_subplot(gs[0, 0])
        ax0.imshow(foo.clip(thr/3, 255).astype(np.uint8),
                   interpolation='nearest')
        ax0.set_title("gaussian filter residuals, gaussian size: %d"%GS,
                      fontsize = 10)
        ax0.axis('off')
        ax1 = stackfig.add_subplot(gs[0, 1])
        ax1.set_title("thresholded image, threshold: %.2f"%thr,
                      fontsize = 10)
        ax1.imshow(foo.mean(-1) > thr, interpolation = 'nearest',
                   cmap = 'gist_gray')
        ax1.axis('off')
        ax3 = stackfig.add_subplot(gs[1, :])
        ax3.bar(bins[:-1], countsnorm,  widths,
                        color = 'gray')
        for i, k in enumerate(pc[1:-1]):
            ax3.plot([k, k], [0, ax3.get_ylim()[1]],
                 '-', color = 'SteelBlue')
            ax3.text(k, ax3.get_ylim()[1] * (1.0 - i * 0.1),
                     "%.1f"%(pcs[i + 1]), ha = 'right')
            ax3.text(k, ax3.get_ylim()[1] * (1.0 - i * 0.1 - 0.08),
                     "%.1f"%(k), ha = 'right')

        ax3.plot([thr, thr], [0, ax3.get_ylim()[1]],
                 '-', color = 'IndianRed')
        ax3.text(thr, ax3.get_ylim()[1] * 0.8, "%d"%(thr),
                 ha = 'left', color = 'red')

        ax3.set_xlim(pc[0], max(pc[-2], thr))
        # pl.show()
    while not done:
        pl.draw()

        if raw_input("is this the right threshold?").lower().startswith('y'):
            done = True
            pl.savefig(args[0].replace(".npy", "_allwindows.pdf"))
            pl.close()
        else:
            thr = np.float(raw_input("try another threshold"))
            ax0 = stackfig.add_subplot(gs[0, 0])
            ax0.imshow(foo.clip(thr / 3, 255).astype(np.uint8),
                       interpolation = 'nearest')
            ax1 = stackfig.add_subplot(gs[0, 1])
            ax1.imshow(foo.mean(-1) > thr,
                       interpolation = 'nearest',
                       cmap = 'gist_gray')
            ax3 = stackfig.add_subplot(gs[1, :])
            ax3.bar(bins[:-1], countsnorm,  widths,
                        color = 'gray')
            for i, k in enumerate(pc[1:-1]):
                ax3.plot([k, k], [0, ax3.get_ylim()[1]],
                         '-', color = 'SteelBlue')
                ax3.text(k, ax3.get_ylim()[1] * (1.0 - i * 0.1),
                         "%.1f"%(pcs[i + 1]), ha = 'right')
                ax3.text(k, ax3.get_ylim()[1] * (1.0 - i * 0.1 - 0.08),
                         "%.1f"%(k), ha = 'right')
        pl.ioff()
        ax3.plot([thr, thr], [0, ax3.get_ylim()[1]],
                 '-', color = 'IndianRed')
        ax3.text(thr, ax3.get_ylim()[1]*0.8, "%d"%(thr),
                 ha = 'right')
        ax3.plot([thr, thr], [0, ax3.get_ylim()[1]],
                         '-', color = 'IndianRed')
        ax3.set_xlim(pc[0], max(pc[-1], thr))
    pl.ioff()
    newdata = sp.ndimage.filters.median_filter(\
            (foo.mean(-1) > thr).astype(float), 4).astype(np.uint8)

    labels, nlabels = sp.ndimage.measurements.label(newdata)

    print("Found %d individual labels"%nlabels)

    # only choose windows larger than 10 pixels to remove noise speks
    goodwindows = [(labels == i).sum() > 10 for i in range(nlabels)]

    print("Found %d individual labels with > 10 pixels"%np.sum(goodwindows))

    # remove bad windows
    for i in range(nlabels):
        if ~goodwindows[i]:
            labels[labels == i] = 0

    np.save(args[0].replace(".npy", "_labels.npy"), labels)
    # resetting label numbers to get  better color map

    tmp = np.random.choice(range(labels.max() + 1), labels.max() + 1,
                           replace = False)

    def mapcolor(i):
        return tmp[i].astype(float)

    bar = np.array(map(mapcolor, labels))
    print(labels, labels == 0)
    bar[labels == 0] = 0.
    bar = bar / bar.max()
    clrs = (pl.cm.jet((bar)) * 255).astype(np.uint8)

    # set background to black
    clrs[bar == 0] = [0, 0, 0, 255]

    fig = pl.figure()

    fig.imshow(clrs, interpolation = 'nearest')
    np.save(args[0].replace(".npy", "_labelledwindows.npy"), clrs)
    pl.xlim(0, pl.xlim()[1])
    pl.ylim(pl.ylim()[0], 0)
    # pl.draw()
    pl.title("gaussian std %d, \threshold %.1f, labels %d"%(GS,
                                                           thr,
                                            np.sum(goodwindows)))
    pl.savefig(args[0].replace(".npy", "_labelledwindows.pdf"))
    if options.showme: pl.show()
    pl.close()
    coords = np.array(
        sp.ndimage.measurements.center_of_mass(newdata, labels,
                                               np.where(
                                                   goodwindows))).squeeze()
    mask = np.array([labels == i for i in range(1, labels.max() + 1)])
    fig = pl.figure()

    fig.imshow(clrs, interpolation = 'nearest')
    pl.xlim(0, pl.xlim()[1])
    pl.ylim(pl.ylim()[0], 0)

    coords = (coords[(coords[:, 0] > 5) * \
                     (coords[:, 0] < stack.shape[0] - 5) *\
                     (coords[:, 1] > 5) * \
                     (coords[:, 1] < stack.shape[1] - 5)])

    print("Found %d good windows away from the edges"%len(coords))

    for c in coords:
        pl.plot(c[1], c[0], 'wo', alpha = 0.3)
    pl.title("gaussian std %d, threshold %.1f, labels %d"%(GS,
                                                           thr,
                                            np.sum(goodwindows)))
    pl.savefig(args[0].replace(".npy", "_selectedwindows.pdf"))
    if options.showme: pl.show()
    pl.close()
    # coords = np.array([coords[:, 1], coords[:, 0]])
    pl.save(args[0].replace(".npy", "_coords.npy"), coords[:, 1::-1])
    pl.save(args[0].replace(".npy", "_mask.npy"), mask)
    # print (mask)

    # for i in range(labels.max()+1):
    #    if i in labels:
    #        pl.figure()
    #        pl.imshow(mask[:i].sum(axis=0),  interpolation='nearest',
    #               cmap='gist_gray')
    #        pl.savefig("mask.%04d.png"%i)
    #        pl.close('all')
