from __future__ import print_function
# CUSP UO 2016
__author__ = "fbb"

import os
import sys
import glob
import numpy as np
import optparse
import pylab as pl
import matplotlib
import json

from images2gif import writeGif
from PIL import Image, ImageSequence

from findImageSize import findsize
s = json.load(open("fbb_matplotlibrc.json"))
pl.rcParams.update(s)

try:
    raw_input
except NameError:
    # Python 3
    raw_input = input

# if you have the stack parameters, for example from stacks already processed
# you can read this in
# import configstack as cfg

# save the images that go in the stack as png files
SAVEALL = False
# SAVEALL = True

# nstack = cfg.srcpars['nstack']
font = {'size': 23}


def stackem(imgesize, stackarray, filename, gifit=False):

    """ stacks a series of np arrays in a stack
    Args:
        imgesize :
            a dictionary containing the number of rows,
            columns and bands in the image
            stackarray : a numpuy ndarray dontaining  the data
            filename : the name of the file output, no path, no extension
    Returns: the stack in an nD array
    """

    stack = np.zeros_like(stackarray.shape[0])
    if SAVEALL:
        for i, f in enumerate(stackarray):
            print("\r  working on image {0} of {1}".format(i+1,\
                stackarray.shape[0]),
                end="")
            sys.stdout.flush()
            fig = pl.figure()
            pl.imshow(f, interpolation='nearest')
            pl.savefig('../outputs/stacks/'+filename+'_%03d.png'%i)
            pl.close(fig)
    # save the images of the stack in a movie
    if gifit:
        print("\nGIFfing...")
        writeGif('../outputs/stacks/' + \
                 filename + "_N%d"%stackarray.shape[0] + ".GIF",
                 [Image.fromarray(np.uint8(np.array(f) / \
                                           f.max() * 255))
                  for f in stackarray],
                 duration = 0.01)

    stack = np.median(stackarray, axis = 0).astype(np.uint8)
    return stack


if __name__ == '__main__':

    """
        Args:
            filename root: no extention path starting with $UIdata

        Uses:
            findImageSize.findsize():  finds image size by trial and attempt,
                               if not provided or stored in a file.
            also: glob, optparse
       Comments:
            This creates a directory **stacks** and stores the aa file
            recording the image size in it (under the assumptinon that
            the image size for science images is the same as that of the
            images used to make the stack.  If the image input has a path
            it will also create a directory corresponding to the full image
            path, up to the name (**groundtest1** in this case)
    """
    parser = optparse.OptionParser(usage="stackImages.py 'filepattern'",
                                   conflict_handler="resolve")
    parser.add_option('--nstack', default=20, type="int",
                      help='number of images to stack')
    parser.add_option('--gif', default=False, action="store_true",
                      help='make a gif')
    parser.add_option('--imsize', default='', type="str",
                      help='image size file')
    parser.add_option('--showme', default=False, action="store_true",
                      help='showing the stacked image')

    options,  args = parser.parse_args()
    print("options and arguments:", options, args)
    if len(args) != 1:
        sys.argv.append('--help')
        options,  args = parser.parse_args()

        sys.exit(0)

    try:
        os.environ['UIdata']
    except KeyError:
        print("must set environmental variable UIdata ")
        print("to the raw data directory")
        sys.exit(

    filepattern = args[0]
    impath = "%s/%s*.raw"%(os.getenv("UIdata"), filepattern)
    flist = glob.glob(impath)
    img = flist[0]

    # extract the filename from the path
    fnameroot = filepattern.split('/')[-1]

    # creates the directory to store the stacks
    os.system('mkdir -p ../outputs/stacks/' + '/'.\
              join(filepattern.split('/')[:-1]))

    # creates the directory to store the other produts
    os.system('mkdir -p ../outputs/'+'/'.join(filepattern.split('/')[:-1]))

    # find image size
    if not options.imsize == '':
        imsize = findsize(img, imsizefile=options.imsize)
    else:
        imsize = findsize(img, filepattern=filepattern)

    rgb = np.zeros((len(flist[:options.nstack]), imsize['nrows'],
                    imsize['ncols'], imsize['nbands']), np.uint8)

    for i, f in enumerate(flist[:options.nstack]):
        try:
            # reads in raw images
            rgb[i] = np.fromfile(f, dtype=np.uint8).clip(0, 255).\
                           reshape(imsize['nrows'],
                                   imsize['ncols'],
                                   imsize['nbands']).astype(float)
        except: pass



    # stack the images
    stack = stackem(imsize, rgb, filepattern, options.gif)
    print("")

    # saves the stack as a npy array file
    np.save('../outputs/stacks/%s_N%d.npy'%(filepattern, options.nstack),
            stack)

    # saves the stack in a png
    stackfig = pl.figure()
    ax0 = stackfig.add_subplot(211)
    ax0.imshow(stack.clip(0, 255).astype(np.uint8), interpolation='nearest')
    ax0.axis('off')
    if options.showme: pl.show()
    stackfig.savefig('../outputs/stacks/%s_N%d.png'%(filepattern,
                                                     options.nstack))
    print('../outputs/stacks/%s_N%d.png'%(filepattern, options.nstack)
