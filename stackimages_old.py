from __future__ import print_function
##CUSP UO 2016
__author__ = "fbb"

import glob
import numpy as np
import matplotlib.pyplot as pl
import sys
import pickle as pkl
from scipy import misc
from scipy import ndimage
import matplotlib
import json
s = json.load( open("fbb_matplotlibrc.json") )
pl.rcParams.update(s)

import configstack as cfg

SAVEALL = True
SAVEALL = False
REREAD = False
REREAD = True
REWINDOW = False
REWINDOW = True
REGIF = False
REGIF = True

ANALIZE = False
DETREND =0

ww = cfg.srcpars['search_aperture']
ew = cfg.srcpars['extract_aperture']
nmax = cfg.srcpars['nmax']
nstack = cfg.srcpars['nstack']
sample_rate = cfg.imgpars['sample_rate']
font = {'size'   : 23}

def read_to_lc(c, aperture, flist, fname, imgeom, col='w'):
    nrow, ncol, nband = imgeom
    fig, axs = pl.subplots(1,1,figsize=(10,10))
    #pl.imshow(np.fromfile(flist[0],dtype=np.uint8).reshape(nrow,ncol,nband))
    #pl.imshow(np.fromfile(flist[0],dtype=np.uint8).reshape(nrow,ncol,nband)[x1:x2,y1:y2])
    #pl.figure()
    #axs.imshow(np.fromfile(flist[0],dtype=np.uint8).reshape(nrow,ncol,nband)[y1:y2,x1:x2],  interpolation='none')
    print ("c",c,"aperture",aperture)
    try:
        axs.imshow(np.fromfile(flist[0],
                           dtype=np.uint8).\
               reshape(nrow,ncol,nband)[c[1]-aperture*2:
                                        c[1]+aperture*2+1,
                                        c[0]-aperture*2:
                                        c[0]+aperture*2+1],
               interpolation='nearest', cmap='bone')
    except IndexError: pass
    x1, x2 = c[0]-aperture, c[0]+aperture+1
    y1, y2 = c[1]-aperture, c[1]+aperture+1
    #axs.plot([x1,x2],[y1,y1], '-', color='%s'%col)    
    #axs.plot([x2,x2],[y1,y2], '-', color='%s'%col)    
    #axs.plot([x1,x2],[y2,y2], '-', color='%s'%col)    
    #axs.plot([x1,x1],[y2,y1], '-', color='%s'%col)    
    #pl.show()
    pl.savefig(fname+"_thumb_%d_%d.png"%(c[0],c[1]))
    pl.close()
    if REREAD:
        if REGIF: images = []
        
        a = np.zeros(len(flist[:nmax]))
        rgb = np.zeros((len(flist[:nmax]),3))
        for i,f in enumerate(flist[:nmax]):
            try:
                data = np.fromfile(f,dtype=np.uint8).\
                        reshape(nrow, ncol, nband)[y1:y2, x1:x2].astype(float)
                rgb[i] = np.array([data[:,:,0].sum(),
                                   data[:,:,1].sum(),
                                   data[:,:,2].sum()])/data[:,:,0].sum()
                a[i] = (data.sum())
            except: 
                a[i]=float('NaN')
                rgb[i] = (float('NaN'), float('NaN'), float('NaN'))
                continue                
            if REGIF:
                import PIL
                matplotlib.use('TkAgg')
                from images2gif import writeGif
                #fig = pl.figure()
                #axs   = fig.add_subplot (111)
                #axs.imshow(np.fromfile(f,
                #                       dtype=np.uint8).\
                #           reshape(nrow,ncol,nband)[c[1]-aperture*2:
                #                                    c[1]+aperture*2+1,
                #                                    c[0]-aperture*2:
                #                                    c[0]+aperture*2+1],
                #           interpolation='none', cmap='bone')


                #fig.canvas.draw ( )
                #w, h = fig.canvas.get_width_height()
                #buf = np.fromstring ( fig.canvas.tostring_rgb(),
                      #                dtype=numpy.uint8 )

                #buf.shape = ( w, h, 4 )
 
                #buf = np.roll ( buf, 3, axis = 2 )
                buf = np.fromfile(f,
                                       dtype=np.uint8).\
                           reshape(nrow,ncol,nband)[c[1]-aperture*2:
                                                    c[1]+aperture*2+1,
                                                    c[0]-aperture*2:
                                                    c[0]+aperture*2+1]
                im=PIL.Image.fromstring( "RGB", ( aperture*2 ,aperture*2 ),
                                         buf.tostring())
                #im = im.convert(mode="RGB")
                images.append(im)
                #print (a[i])
                

            #print (np.fromfile(f,dtype=np.uint8).reshape(nrow,ncol,nband)[y1:y2,x1:x2,0])
            #a[i] = (np.fromfile(f,dtype=np.uint8).reshape(nrow,ncol,nband)[y1:y2,x1:x2]).sum()
    else:
        a = pkl.load(open('lcvs/'+fname+"_%d_%d.pkl"%(c[0],c[1]),'rb'))
        rgb = pkl.load(open('lcvs/'+fname+"_rgb_%d_%d.pkl"%(c[0],c[1]),'rb'))        
    f0 = ((np.fromfile(flist[0],dtype=np.uint8).reshape(nrow,ncol,nband)[y1:y2,x1:x2]))
    area = int((x2-x1)*(y2-y1))
    print ('mean: {0:.2f}, stdev: {1:.2f}, area: {2:d} pixels, {3:d} images, average flux/pixel: {4:.1f}'.format(
                    np.nanmean(a), np.nanstd(a), area, len(a), np.nanmean(a)/float(area)))
    R = float(f0[:,:,0].sum())
    G = float(f0[:,:,1].sum())
    B = float(f0[:,:,2].sum())
    mx = np.argmax([R,G,B])
    colors= ['Red','Yellow','Blue']
    print (colors[mx], ': R: {0:.2f} G: {1:.2f} B: {2:.2f}'.format(1, G/R, B/R))
    '''
    if REREAD:
        pkl.dump(a,open('lcvs/'+fname+"_%d_%d.pkl"%(c[0],c[1]),"wb"),
                 protocol=2)
        pkl.dump(rgb,open('lcvs/'+fname+"_rgb_%d_%d.pkl"%(c[0],c[1]),"wb"),
                 protocol=2)
    '''
    if REGIF:  writeGif('lcvs/'+fname+"_%d_%d.gif"%(c[0],c[1]),
                        images, dither=0)
    
    return a, rgb

def get_plot_lca (c, aperture, flist, fname, imgeom, col='k'):
    nrow, ncol, nband = imgeom
    a, rgb = read_to_lc(c, aperture, flist, fname, imgeom, col=col)
    n = len(a)
    flux0 = (a - np.nanmean(a))/np.nanstd(a)
    #flux = flux0.copy()
    pl.rc('font', **font)
    fig = pl.figure(figsize=(15, 10))
    axs = pl.subplot2grid((4,3), (0,0), colspan=2, rowspan=2)#(211)
    axs.plot(np.arange(0, n*sample_rate, sample_rate), flux0, '.')
    axs.plot(np.arange(0, n*sample_rate, sample_rate), flux0, 'k-')
    axs.yaxis.set_major_formatter(pl.NullFormatter())
    axs.set_xlabel("sec")
    axs.set_ylabel("standardized flux")

    axs.set_title(c)

    axs = pl.subplot2grid((4,3), (2,0), colspan=2)
    axs.plot(np.arange(0, n*sample_rate, sample_rate), rgb[:,0], 'r-')
    axs.plot(np.arange(0, n*sample_rate, sample_rate), rgb[:,1], 'g-')
    axs.plot(np.arange(0, n*sample_rate, sample_rate), rgb[:,2], 'b-')
    axs.set_xlabel("sec")
    axs.set_ylabel("G/R B/R color")

    axs = pl.subplot2grid((4,3), (3,0), colspan=2)
    freq = np.fft.rfftfreq(n, d=sample_rate)[1:]
    myfft = np.abs(np.fft.rfft(a))[1:]
    axs.plot(freq, myfft, alpha=0.5, color='IndianRed')
    axs.plot([0.25,0.25],[myfft.min(), myfft.max()],'k-')
    axs.yaxis.set_major_formatter(pl.NullFormatter())
    axs.set_xlabel("period (seconds)")
    axs.set_ylabel("power")

    
    #axs.set_xlim(0, 200)
    axs = pl.subplot2grid((4,3), (0,2), rowspan=2)#(211)
    axs.imshow(np.fromfile(flist[0],
                           dtype=np.uint8).\
               reshape(nrow,ncol,nband)[c[1]-aperture*4:
                                        c[1]+aperture*4+1,
                                        c[0]-aperture*4:
                                        c[0]+aperture*4+1],
               interpolation='nearest')
    x1 = aperture*3
    x2 = aperture*5
    y1,y2 = x1,x2
    axs.plot([x1,x1], [y1,y2], '-', color='DarkOrange')
    axs.plot([x1,x2], [y1,y1], '-', color='DarkOrange')  
    axs.plot([x2,x2], [y1,y2], '-', color='DarkOrange')   
    axs.plot([x1,x2], [y2,y2], '-', color='DarkOrange')
    axs.axis('off')
    axs.set_xlim(0,aperture*8)
    axs.set_ylim(0,aperture*8)

    axs = pl.subplot2grid((4,3), (2,2), rowspan=2)#(211)
    axs.imshow(np.fromfile(flist[0],
                           dtype=np.uint8).\
               reshape(nrow,ncol,nband),
               interpolation='nearest')
    ew = aperture*4
    axs.plot([c[0]-ew, c[0]-ew], [c[1]-ew, c[1]+ew], '-', color='DarkOrange')
    axs.plot([c[0]+ew, c[0]+ew], [c[1]-ew, c[1]+ew], '-', color='DarkOrange')   
    axs.plot([c[0]+ew, c[0]-ew], [c[1]+ew, c[1]+ew], '-', color='DarkOrange')   
    axs.plot([c[0]-ew, c[0]+ew], [c[1]-ew, c[1]-ew], '-', color='DarkOrange')   
    #axs.set_xlim(0,nrow)
    #axs.set_ylim(0,ncol)
    axs.axis('off')
    
    
    pl.savefig("lcvs/"+fname+"_lc_%d_%d.png"%(c[0],c[1]))
    pl.close()
    return flux0

def stackem(imgeom, flistforstack, fname):
    nrow, ncol, nband = imgeom
    stack = np.zeros((len(flistforstack), nrow, ncol, nband), np.uint8)
    for i,f in enumerate(flistforstack):
        print (f)
        data = np.fromfile(f,dtype=np.uint8).reshape(nrow,ncol,nband)
        if SAVEALL:
            pl.imshow(data)
            pl.savefig('stacks/'+f.split('/')[-1]+'.png')
        sharpened  =  data
        #np.dstack(
            #[1.0*im + 0.5*(1.0*im - ndimage.filters.gaussian_filter(1.0*im,2))
            # for im in data.transpose(2, 0, 1)]).clip(0, 255).astype(np.uint8)
        #pl.imshow(sharpened,vmin=0, vmax=255)
        #pl.show()
        
        stack[i] = sharpened

    #print ("here", stack[:])
    stack = np.median(stack, axis=0).astype(np.uint8)
    return stack

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

