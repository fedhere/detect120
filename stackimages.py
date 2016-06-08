from __future__ import print_function
##CUSP UO 2016
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
s = json.load( open("fbb_matplotlibrc.json") )
pl.rcParams.update(s)

import configstack as cfg

GIFALL = False
#GIFALL = True
SAVEALL = False
#SAVEALL = True

nstack = cfg.srcpars['nstack']
font = {'size'   : 23}

def stackem(imgesize, stackarray, filename):
    stack = np.zeros((imsize['nrows'],
                      imsize['ncols'],
                      imsize['nbands']))
    if SAVEALL:
          for i,f in enumerate(stackarray):
              print ("\r  working on image {0} of {1}".format(i+1, stackarray.shape[0]),
                     end="")
              sys.stdout.flush()
              fig = pl.figure()
              pl.imshow(f, interpolation='nearest')
              pl.savefig('stacks/'+filename+'_%03d.png'%i)
              pl.close(fig)
    if GIFALL:
        print ("\nGIFfing...")
        writeGif('stacks/'+ filename + \
                 "_N%d"%stackarray.shape[0]+".GIF",
                 stackarray, duration=0.01)

    stack = np.median(stackarray, axis=0).astype(np.uint8)
    return stack


if __name__ == '__main__':
    parser = optparse.OptionParser(usage="stackImages.py 'filepattern'",
                                   conflict_handler="resolve")
    parser.add_option('--nstack', default=20, type="int",
                      help='number of images to stack')
    parser.add_option('--imsize', default='', type="str",
                      help='image size file')    
    parser.add_option('--showme', default=False, action="store_true",
                      help='show plots')


    options,  args = parser.parse_args()
    print ("options and arguments:", options, args)
    if len(args) != 1:
        sys.argv.append('--help')
        options,  args = parser.parse_args()

        sys.exit(0)
    
    filepattern = args[0]
    impath = os.getenv("UIdata") + '/' + filepattern + '*.raw'
    flist = glob.glob(impath)
    img = flist[0]

    fnameroot = filepattern.split('/')[-1]

    if not options.imsize == '':
        imsize = findsize(img, imsizefile=options.imsize)
    else:
        imsize = findsize(img, filepattern=filepattern)
        
    rgb = np.zeros((len(flist[:options.nstack]), imsize['nrows'],
                    imsize['ncols'], imsize['nbands']), np.uint8) 

    for i,f in enumerate(flist[:options.nstack]):
        try:
            rgb[i] = np.fromfile(f, dtype=np.uint8).clip(0,255).\
                   reshape(imsize['nrows'],
                           imsize['ncols'],
                           imsize['nbands']).astype(float)
        except: pass

    os.system('mkdir -p stacks/'+'/'.join(filepattern.split('/')[:-1]))
    stack = stackem(imsize, rgb, filepattern)
    print ("")
    stackfig = pl.figure()
    ax0 = stackfig.add_subplot(211)
    ax0.imshow(stack.clip(0,255).astype(np.uint8), interpolation='nearest')
    ax0.axis('off')
    if options.showme: pl.show()
    stackfig.savefig('stacks/%s_N%d.png'%(filepattern, options.nstack))

    np.save('stacks/%s_N%d.npy'%(filepattern, options.nstack), stack)

'''
class getWval:

    def __init__(self, data, clist):
        figWH = (10,10) # in
        self.data = data
        self.figure, self.ax = pl.subplots(1, 1, figsize=figWH)
        self.ax.imshow(self.data, interpolation='nearest', cmap='bone')
        self.clist = [c for c in clist]
        self.connect = self.figure.canvas.mpl_connect
        self.disconnect = self.ax.figure.canvas.mpl_disconnect

        self.clickCid = self.connect("button_press_event", self.onClick)
        pl.show(self.figure)

    def onClick(self, event):
        #print (event)
        if event.inaxes:
            print ('button=%d, xdata=%f, ydata=%f, x=%d, y=%d, '%(event.button,
                                                                  event.xdata,
                                                                  event.ydata,
                                                                  event.x,
                                                                  event.y))
            self.clist.append(np.array([event.xdata, event.ydata]))
            print (self.clist)
            return (event.xdata, event.ydata)

    def centerem(self):
        #print ("centering")
        self.newlist = np.array(self.clist).copy()
        for i,c in enumerate(self.clist):
            box = (self.data[c[1]-ww:c[1]+ww+1,c[0]-ww:c[0]+ww+1]).mean(axis=2)

            mx = np.unravel_index(box.argmax(), box.shape)

            newc = c - np.array([ww,ww]) + np.array(mx[1::-1])
            newbox = (self.data[newc[1]-ww:newc[1]+ww+1,
                                newc[0]-ww:newc[0]+ww+1]).mean(axis=2)
            
            cm = ndimage.measurements.center_of_mass(newbox)

            self.newlist[i] = (np.array(newc)+0.5).astype(int) - np.array([ww,ww]) + \
                           (np.array(cm[1::-1])+0.5).astype(int)
            
            

def getem(stack, clist):
    wvals = getWval(stack, clist)


    wvals.disconnect(wvals.clickCid)
    wvals.centerem()

    return wvals



flist = glob.glob(cfg.imgpars['froot'])
fname = flist[0].split("/")[-1]    

stack = stackem((cfg.imgpars['nrow'], cfg.imgpars['ncol'],
                 cfg.imgpars['nband']), flist[:nstack], fname)

fig, axs = pl.subplots(1,1,figsize=(10,10))

pl.imshow(stack, interpolation='nearest', cmap='bone')
pl.savefig('stacks/%s_%d.png'%(fname.replace('.raw','_N'),
                               nstack))

np.save('stacks/%s_%d.npy'%(fname.replace('.raw',''),
                               nstack), stack)

'''
