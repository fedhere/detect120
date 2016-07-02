from __future__ import print_function
##CUSP UO 2016
__author__ = "fbb"

import glob
import sys
import os
import optparse
import pylab as pl
import numpy as np

from scipy.ndimage.filters import median_filter
from images2gif import writeGif
from PIL import Image
from findImageSize import findsize

#giflen=100
#pl.ion()
pl.ioff()
OUTPUTDIR = "../outputs/"
SAVEORIGINAL = True
SAVEORIGINAL = False

if __name__=='__main__':

    '''
    Makes a movie of ONLY the selected windows (read from _family.npy file)

    Arguments:
           file pattern (from UIdata dir, e.g. 
           >python make120sMovie.py groundtest1/ESB_s119.75Hz_c4.00Hz_100ms_2016-05-24-215440)
    '''
    

    parser = optparse.OptionParser(usage="make120sMovie.py 'filepattern' ",   
                                       conflict_handler="resolve")
    parser.add_option('--families', default=None, type="str",
                      help='building families file (from lassoselect.py)')
    parser.add_option('--nmax', default=100, type="int",
                      help='number of images to process (i.e. timestamps)')
    parser.add_option('--lmax', default=None, type="int",
                      help='number of lights')
    parser.add_option('--skipfiles', default=150, type="int",
                      help="number of files to skip at the beginning")      
    parser.add_option('--coords', default=None, type="str",
                      help='coord file')  
    parser.add_option('--stack', default=None, type="str",
                      help='stack file')  

    options,  args = parser.parse_args()
    filepattern = sys.argv[1]
    impath = os.getenv("UIdata") + filepattern
    fnameroot = filepattern.split('/')[-1]
    dirstring = "/N%04dW%04dS%04d/"%(options.nmax,
                                   options.lmax,
                                   options.skipfiles)
    #options.nmax = 10
    flist = sorted(glob.glob(impath+"*.raw"))[options.skipfiles:options.nmax + options.skipfiles]#[:30]

    # load coordinates
    if options.coords:
        coordfile =  OUTPUTDIR + options.coords
    else:
        coordfile = OUTPUTDIR + '/'.join(filepattern.split('/')[:-1]) +\
                    dirstring + fnameroot +\
                    "_goodcoords.npy"
    if not os.path.isfile(coordfile):
        print("need coordinate file ", coordfile)
        sys.exit()
    coords = np.load(coordfile)
    
    # load one image to get size
    if  os.path.isfile(OUTPUTDIR + options.stack):
        img = np.load(OUTPUTDIR + options.stack)
        print("there is a stack", OUTPUTDIR + options.stack)
    else: img = flist[0]
    imsize = findsize(img, filepattern = filepattern)
    img = flist[0]
    img = np.fromfile(img, dtype=np.uint8).reshape(imsize['nrows'],
                                                       imsize['ncols'],
                                                       imsize['nbands']) 
    print("Image size: ", imsize)

    # imshow(np.repeat((windows == l),3).reshape(stack.shape)*stack)

    mask = np.zeros((img.shape[0], img.shape[1]))*False

    # load windows families
    if not os.path.isfile(OUTPUTDIR + options.families):
        print("need coordinate file ", OUTPUTDIR + options.families)
        sys.exit()

    families = np.load(OUTPUTDIR + options.families)

    if not os.path.isfile(OUTPUTDIR + options.families):
        print("need labels file ",
               OUTPUTDIR + options.families.replace("families", "labels"))
        sys.exit()
    windows = np.load(OUTPUTDIR + options.families.replace("families", "labels"))

    for f in coords.T:
        #print (mask.sum())
        l = windows[int(f[3]),int(f[2])]
        #windows == l
        mask = mask + (windows == l)
    imgmask = np.repeat(mask, 3).reshape(img.shape)
    imgs = []
    imgsnomask = []
    for f in flist:
        data =  np.fromfile(f, dtype=np.uint8).reshape(imsize['nrows'],
                           imsize['ncols'],
                           imsize['nbands'])#-50.0) * 3

        #data = np.sqrt(np.fromfile(f,dtype=np.uint8).reshape(imsize['nrows'],
        #                   imsize['ncols'],
        #                   imsize['nbands'])/255.0) * 255.
        #pl.show()

        # gamma correction

        # print (stats.mode(data, axis=None)
        data_new = (median_filter((255* (data.clip(20, img.max())/255.)**1), [3, 3, 1])).clip(0, 255).astype(np.uint8)
        # data_new=(data.clip(20, img.max())).clip(0, 255).astype(np.uint8)
        
        if SAVEORIGINAL :
            imgsnomask.append(data_new)
            pl.imshow(imgsnomask[-1])
            pl.axis('off')
            pl.savefig(f.split('/')[-1].replace('.raw','.png'))

        imgs.append((imgmask * data_new).clip(0, 255).astype(np.uint8))
        pl.imshow(imgs[-1])
        pl.axis('off')
        pl.savefig(f.split('/')[-1].replace('.raw','_masked.png'))
        #pl.show()
    images = [Image.fromarray(f) for f in \
              imgs]
    writeGif(OUTPUTDIR + filepattern + dirstring.replace('/','') + "_120only.GIF",
             images, duration=0.01)
    if SAVEORIGINAL :
        images = [Image.fromarray(f) for f in \
                  imgsnomask]
        writeGif(OUTPUTDIR + filepattern + dirstring.replace('/','') + ".GIF",
                 images, duration=0.01)
