import numpy as np
import scipy as sp
import pylab as pl
'''
stack = np.load("stacks/ESB_c0.7Hz_250ms_2016-05-24-230354-0000_20.npy")
#pl.imshow(stack, interpolation='nearest')
overgaussstack = (stack.astype(float) - sp.ndimage.gaussian_filter(stack.astype(float), (10,10,0)))
#pl.imshow(overgaussstack.astype(np.uint8), interpolation='nearest')
foo = overgaussstack - overgaussstack.min()
foo/=foo.max()
foo.dtype
foo *= 255
#pl.imshow(foo.clip(0,255).astype(np.uint8),interpolation='nearest')
#pl.imshow(foo.mean(-1)>112,interpolation='nearest',cmap='gist_gray')
newdata = sp.ndimage.filters.median_filter(\
                (foo.mean(-1)>110).astype(float), 4).astype(uint8)

labels, nlabels = scipy.ndimage.measurements.label(newdata)
#windowlabels = [sum(labels==i)>10


goodwindows = [sum(labels==i)>10 for i in range (nlabels)]

bar = labels.astype(float)/labels.max()*1000%100
bar = bar/bar.max()

print bar.astype(float)

clrs2 = (cm.jet((bar))*255).astype(np.uint8)

#clrs2 = (bar*1000%100*25.5).astype(uint8)

#pl.imshow(clrs2, interpolation='nearest')

clrs2[bar==0] = [0,0,0,255]

#pl.figure()
#pl.imshow(clrs2, interpolation='nearest')


labels2 = np.zeros_like(labels)
for i in range (nlabels):
     if goodwindows[i]:
         labels2[labels==i] = labels[labels==i]


bar2 = labels2.astype(float)/labels2.max()*1000%100
bar2 = bar2/bar2.max()


clrs3 = (cm.jet((bar))*255).astype(np.uint8)

#clrs2 = (bar*1000%100*25.5).astype(uint8)

#pl.imshow(clrs2, interpolation='nearest')

clrs3[bar2==0] = [0,0,0,255]
'''
pl.figure()

pl.imshow(clrs3, interpolation='nearest')

coords = sp.ndimage.measurements.center_of_mass(newdata, labels, np.where(goodwindows))

for c in coords:
    pl.plot(c[0][1],c[0][0],'wo')
    
'''
overgaussstack = (stack - scipy.ndimage.gaussian_filter(stack, 20))
grad = overgaussstack[1:,1:,:].mean(axis=2) - overgaussstack[:-1,:-1,:].mean(axis=2)
imshow(grad, interpolation='nearest')
overgaussstack = (stack - scipy.ndimage.gaussian_filter(stack, 30))
imshow(overgaussstack)
pl.figure()
imshow(overgaussstack)
pl.figure()
imshow(stack, interpolation='nearest')
pl.figure()
imshow(overgaussstack, interpolation='nearest')
ndimage?
median_filter(stack)
scipy.ndimage.median_filter(stack)
scipy.ndimage.median_filter(stack, 10)
pl.close('all')
pl.figure()
imshow(stack,interpolation='nearest')
pl.figure()
overgaussstack = (stack - scipy.ndimage.gaussian_filter(stack, 50))
imshow(overgaussstack,interpolation='nearest')
overgaussstack = (stack.astype(float) - scipy.ndimage.gaussian_filter(stack.astype(float), (50,50,0)))
imshow(overgaussstack,interpolation='nearest')
imshow(overgaussstack.astype(np.uint8),interpolation='nearest')

stack.dtype
overgaussstack.min()
overgaussstack.mad()
overgaussstack.max()
imshow(overgaussstack.clip(0,255).astype(np.uint8),interpolation='nearest')

imshow(foo.clip(0,255).astype(np.uint8),interpolation='nearest')
imshow(foo.mean(-1),interpolation='nearest')
imshow(foo.mean(-1),interpolation='nearest',cmap='gist_gray')
imshow(foo.mean(-1),interpolation='nearest',cmap='gist_gray',clim=[100,255])
imshow(foo.mean(-1),interpolation='nearest',cmap='gist_gray',clim=[100,200])
imshow(foo.mean(-1),interpolation='nearest',cmap='gist_gray',clim=[50,200])
imshow(foo.mean(-1),interpolation='nearest',cmap='gist_gray',clim=[50,100])
imshow(foo.mean(-1)>50,interpolation='nearest',cmap='gist_gray')
imshow(foo.mean(-1)>100,interpolation='nearest',cmap='gist_gray')
imshow(foo.mean(-1)>150,interpolation='nearest',cmap='gist_gray')
imshow(foo.mean(-1)>130,interpolation='nearest',cmap='gist_gray')
imshow(foo.mean(-1)>120,interpolation='nearest',cmap='gist_gray')
imshow(foo.mean(-1)>110,interpolation='nearest',cmap='gist_gray')
imshow(foo.mean(-1)>120,interpolation='nearest',cmap='gist_gray')
imshow(foo.mean(-1)>120,interpolation='nearest',cmap='gist_gray')
imshow(foo.mean(-1)>115,interpolation='nearest',cmap='gist_gray')
imshow(foo.mean(-1)>112,interpolation='nearest',cmap='gist_gray')
imshow(scipy.ndimage.filters.median_filter(foo.mean(-1)>110), 2),interpolation='nearest',cmap='gist_gray')
imshow(scipy.ndimage.filters.median_filter(foo.mean(-1)>110, 2),interpolation='nearest',cmap='gist_gray')
pl.figure()
imshow(scipy.ndimage.filters.median_filter((foo.mean(-1)>110).astype(float), 2),interpolation='nearest',cmap='gist_gray')
imshow(foo.mean(-1)>110,interpolation='nearest',cmap='gist_gray')
imshow(scipy.ndimage.filters.median_filter((foo.mean(-1)>110).astype(float), 2),interpolation='nearest',cmap='gist_gray')
imshow(foo.mean(-1)>110,interpolation='nearest',cmap='gist_gray')
imshow(scipy.ndimage.filters.median_filter((foo.mean(-1)>110).astype(float), 3),interpolation='nearest',cmap='gist_gray')
imshow(scipy.ndimage.filters.median_filter((foo.mean(-1)>110).astype(float), 3)>0.1,interpolation='nearest',cmap='gist_gray')
imshow(scipy.ndimage.filters.median_filter((foo.mean(-1)>110).astype(float), 3)>0.3,interpolation='nearest',cmap='gist_gray')
pl.figure()
imshow(scipy.ndimage.filters.median_filter((foo.mean(-1)>110).astype(float), 3),interpolation='nearest',cmap='gist_gray')
imshow(scipy.ndimage.filters.median_filter((foo.mean(-1)>110).astype(float), 3),interpolation='nearest',cmap='gist_gray')
imshow(foo.mean(-1)>110,interpolation='nearest',cmap='gist_gray')
imshow(foo.mean(-1)>110,interpolation='nearest',cmap='gist_gray')
imshow(scipy.ndimage.filters.median_filter((foo.mean(-1)>110).astype(float), 3)>0.5,interpolation='nearest',cmap='gist_gray')
pl.figure()
imshow(scipy.ndimage.filters.median_filter((foo.mean(-1)>110).astype(float), 3)>0.8,interpolation='nearest',cmap='gist_gray')
imshow(scipy.ndimage.filters.median_filter((foo.mean(-1)>110).astype(float), 4),interpolation='nearest',cmap='gist_gray')

imshow(scipy.ndimage.measurements.label(scipy.ndimage.filters.median_filter((foo.mean(-1)>110).astype(float), 4)),interpolation='nearest',cmap='gist_gray')
imshow(scipy.ndimage.measurements.label(scipy.ndimage.filters.median_filter((foo.mean(-1)>110).astype(float), 4)).astype(np.unit8),interpolation='nearest',cmap='gist_gray')
imshow(scipy.ndimage.measurements.label(scipy.ndimage.filters.median_filter((foo.mean(-1)>110).astype(float), 4).astype(np.unit8)),interpolation='nearest',cmap='gist_gray')
imshow(scipy.ndimage.measurements.label(scipy.ndimage.filters.median_filter((foo.mean(-1)>110).astype(float), 4).astype(np.uint8)),interpolation='nearest',cmap='gist_gray')
imshow(scipy.ndimage.measurements.label(scipy.ndimage.filters.median_filter((foo.mean(-1)>110).astype(float), 4).astype(np.uint8)),interpolation='nearest',cmap='gist_gray')

bar = scipy.ndimage.filters.median_filter((foo.mean(-1)>110).astype(float), 4)
bar
bar = scipy.ndimage.filters.median_filter((foo.mean(-1)>110).astype(float), 4).astype(uint8)
bar
scipy.ndimage.measurements.label(bar)
scipy.ndimage.measurements.label(bar).max()
scipy.ndimage.measurements.label(bar)
scipy.ndimage.measurements.label(bar)[1].astype(float)
scipy.ndimage.measurements.label(bar)[1]
scipy.ndimage.measurements.label(bar)[0]
scipy.ndimage.measurements.label(bar)[0].astype(float)
bar = scipy.ndimage.measurements.label(bar)[0].astype(float)
bar /= bar.ma()
bar /= bar.max()
bar
max(bar)
bar.max()
bar.max()
imshow(scipy.ndimage.measurements.label(scipy.ndimage.filters.median_filter((foo.mean(-1)>110).astype(float), 4).astype(np.unit8)),interpolation='nearest',cmap='gist_gray')
cm.jet(bar)
cm.jet(bar).shape
cm.jet(bar)*255
imshow((cm.jet(bar)*255).astype(uint8), interpolation='nearest')
pl.figure()
imshow((cm.jet(bar)*255).astype(uint8), interpolation='nearest')
clrs = (cm.jet(bar)*255).astype(uint8)
clrs[bar==0] = [0,0,0,0]
imshow(clrs, interpolation='nearest')
clrs
clf()
imshow(clrs, interpolation='nearest')
clrs[bar==0] = [0,0,0,255]
imshow(clrs, interpolation='nearest')
bar.max()
np.unique(bar)
len(np.unique(bar))
bar*100%100
len(np.unique(bar*100%100))
bar*100%100
imshow(clrs, interpolation='nearest')
clrs2 = (bar*100%100*255).astype(uint8)
clrs2[bar==0] = [0,0,0,255]
clrs2
clrs2[bar==0] = [0,0,0,255]
clrs2 = (cm.jet(bar*100%100)*255).astype(uint8)
clrs2[bar==0] = [0,0,0,255]
pl.figure()
imshow(clrs2, interpolation='nearest')
clrs2
bar
bar.shape
cm.jet(bar)
cm.jet(bar).shape
cm.jet(bar*100%100).shape
clrs2 = (cm.jet(bar*100%100)*255).astype(uint8)
clrs2.shape
clrs2
clrs2[bar==0] = [0,0,0,255]
imshow(clrs2, interpolation='nearest')
clf()
clrs2 = (cm.jet(bar*100%100)*255).astype(uint8)
clrs2
imshow(clrs2, interpolation='nearest')
clrs2 = (cm.jet(bar*100.%100.)*255).astype(uint8)
imshow(clrs2, interpolation='nearest')
clrs2
clrs2 = (cm.jet(bar*100.%100.)*255).astype(uint8)
bar
bar*100%100
bar
imshow(bar)
imshow(foo)
clrs2 = (cm.jet(bar*1000%100.)*255).astype(uint8)

In [413]: imshow(clrs2, interpolation='nearest')
Out[413]: <matplotlib.image.AxesImage at 0x136872390>

In [414]: clf()

In [415]: imshow(clrs2, interpolation='nearest')
Out[415]: <matplotlib.image.AxesImage at 0x11752b410>

In [416]: imshow(bar*10000%100)
Out[416]: <matplotlib.image.AxesImage at 0x11752b510>

In [417]: imshow(bar*10000%100)[bar==0]^C

KeyboardInterrupt

In [417]: 

In [417]: clrs2 = (bar*1000%100*25.5).astype(uint8)

In [418]: imshow(clrs2, interpolation='nearest')
Out[418]: <matplotlib.image.AxesImage at 0x18d4ec190>

In [419]: clf()

In [420]: imshow(clrs2, interpolation='nearest')
Out[420]: <matplotlib.image.AxesImage at 0x11704e5d0>

In [421]: clrs2[bar==0] = [0,0,0,255]
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-421-5213c3de1d02> in <module>()
----> 1 clrs2[bar==0] = [0,0,0,255]

ValueError: NumPy boolean array indexing assignment cannot assign 4 input values to the 3456229 output values where the mask is true

In [422]: clr
clrs   clrs2  

In [422]: clr
clrs   clrs2  

In [422]: clrs2
Out[422]: 
array([[  6,   6,   6, ...,   0,   0,   0],
       [  6,   6,   6, ...,   0,   0,   0],
       [  6,   6,   6, ...,   0,   0,   0],
       ..., 
       [  0,   0,   0, ..., 110, 110, 110],
       [  0,   0,   0, ..., 110, 110, 110],
       [  0,   0,   0, ..., 110, 110, 110]], dtype=uint8)

In [423]: clrs
clrs   clrs2  

In [423]: clrs2.max()
Out[423]: 255

In [424]: clrs2
Out[424]: 
array([[  6,   6,   6, ...,   0,   0,   0],
       [  6,   6,   6, ...,   0,   0,   0],
       [  6,   6,   6, ...,   0,   0,   0],
       ..., 
       [  0,   0,   0, ..., 110, 110, 110],
       [  0,   0,   0, ..., 110, 110, 110],
       [  0,   0,   0, ..., 110, 110, 110]], dtype=uint8)

In [425]: clrs2.size
Out[425]: 3739320

In [426]: clrs2.shape
Out[426]: (1530, 2444)

In [427]: clrs2 = (cm.jet(clrs2.astype(float)/255)*255).astype(uint8)

In [428]: clrs2[bar==0] = [0,0,0,255]
'''
