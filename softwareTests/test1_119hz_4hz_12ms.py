import pylab as pl
import numpy as np
import glob


nrow = 2048
ncol = 2448
nband = 3

flist=glob.glob("/Users/fbianco/UIdata/test_119hz_4hz_12ms/*113841*raw")
flist119=np.array(flist)[np.argsort(flist)]
imgs=np.zeros((len(flist),nrow,ncol),float)


#pl.ion()

for i,img in enumerate(flist):
    imgs[i]=np.fromfile(img,dtype=np.uint8).reshape(nrow,ncol,nband)[:,:,1]


fig,ax=pl.subplots()
im=ax.imshow(imgs[0],cmap='gist_gray')
ax.set_title("119Hz 0.4Hz 12ms")

fig.canvas.draw()
for img in imgs:
    im.set_data(img)
    fig.canvas.draw()
