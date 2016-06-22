import glob
flist=glob.glob("U*raw")
flistnew=np.array(flist)[argsort(flist)]
flistgood=flistnew[285:]
for f in flistnew[285:286]:
    figure();imshow(np.fromfile(f,dtype=np.uint8).reshape(nrow,ncol,nband))
    show()
figure()
figure()
imshow(np.fromfile(f,dtype=np.uint8).reshape(nrow,ncol,nband))
imshow(np.fromfile(flistnew[285],dtype=np.uint8).reshape(nrow,ncol,nband))
imshow(np.fromfile(flistgood[285],dtype=np.uint8).reshape(nrow,ncol,nband))
flist
flist=glob.glob("U*raw")
flist
!pwd
!cd ../UIdata/
flist=glob.glob("U*raw")
flist
ls
!pwd
%cd ../UIdata/
ls
flist=glob.glob("U*raw")
flistnew=np.array(flist)[argsort(flist)]
flistgood=flistnew[285:]
imshow(np.fromfile(flistgood[285],dtype=np.uint8).reshape(nrow,ncol,nband))
import numpy as np
fname = 'foo.raw'
nrow = 2048
ncol = 2448
nband = 3
imshow(np.fromfile(flistgood[285],dtype=np.uint8).reshape(nrow,ncol,nband))
imshow(np.fromfile(flistgood[285],dtype=np.uint8).reshape(nrow,ncol,nband))[1100:1500,1400:1600]
imshow(np.fromfile(flistgood[285],dtype=np.uint8).reshape(nrow,ncol,nband)[1100:1500,1400:1600])
for f in flistgood[::50]:
    imshow(np.fromfile(f,dtype=np.uint8).reshape(nrow,ncol,nband)[1100:1500,1400:1600])
    show()
for f in flistgood[::50]:
    figure(); imshow(np.fromfile(f,dtype=np.uint8).reshape(nrow,ncol,nband)[1100:1500,1400:1600])
    show()
len(flistgood)
a=np.zeros(len(flistgood))
for i,f in enumerate(flistgood[::50]):
    a[i]=sum(np.fromfile(f,dtype=np.uint8).reshape(nrow,ncol,nband)[1100:1500,1400:1600])
a.mean()
a.mean()
a=a-mean(a)
max(a)
min(a)
mean(a)
plot(a)
a=a[20:]
plot(a)
mean(a)
a=a-mean(a)
mean(a)
a.std()
a.mean()
for i,f in enumerate(flistgood):
    a[i]=sum(np.fromfile(f,dtype=np.uint8).reshape(nrow,ncol,nband)[1100:1500,1400:1600])
len(flistgood)
a=np.zeros(len(flistgood))
len(a)
for i,f in enumerate(flistgood):
    a[i]=sum(np.fromfile(f,dtype=np.uint8).reshape(nrow,ncol,nband)[1100:1500,1400:1600])
a.mean()
a.std()
a=a/a.mean()
a
a=a-1.0
a
min(a)
max(a)
a=a/0.009
a
plot(a)
plot(np.arange(a),a)
plot(np.arange(a.len()),a)
plot(np.arange(len(a)),a)
plot(np.arange(len(a))*1.0/80.,a)
clf()
plot(np.arange(len(a))*1.0/80.,a)
len(flist)
plot(np.arange(len(a))*1.0/0.8.,a)
plot(np.arange(len(a))*1.0/0.8,a)
clf()
plot(np.arange(len(a))*1.0/0.8,a)
plot(np.arange(len(a))*1.0/0.8/60,a)
clf()
plot(np.arange(len(a))*1.0/0.8/60,a)
plot(np.arange(len(a))*1.0/0.8/60,a)
pl.plot(np.arange(20*60*120), np.sin(2.*120*np.pi*np.arange(20*60*120)))
plot(np.arange(20*60*120), np.sin(2.*120*np.pi*np.arange(20*60*120)))
clf()
plot(np.arange(20*60*120), np.sin(2.*120*np.pi*np.arange(20*60*120)))
plot(np.arange(20*60*120), np.sin(2.*120*np.pi*np.arange(20*60*120)))
wave = lambda t: np.sin(2.*freq*np.pi*t+phi)
phi=0
freq=120
t=np.arange(20*60)
wave(t)
plot(t,wave(t))
clf()
plot(t,wave(t))
wave = lambda t: np.sin(2.*freq*np.pi*t+phi)
wave(10)
pl.plot(wave(range(10)))
plot(wave(range(10)))
clf()
plot(wave(np.arange(10)))
t=np.linspace(0,20*60,1000)
plot(wave(t))
clf()
plot(wave(t))
t=np.linspace(0,20*60,10000)
plot(wave(t))[:100]
clf()
plot(wave(t))[:100]
clf()
plot(wave(t)[:100])
t=np.linspace(0,20*60,100000)
plot(wave(t)[:100])
t=np.linspace(0,20*60,1000000)
plot(wave(t)[:100])
clf()
t=np.linspace(0,20*60,10000000)
plot(wave(t)[:100])
plot(wave(t)[:1000])
plot(np.arange(len(a))*1.0/8*10,a)
plot(np.arange(len(a))*1.0/8*100,a)
clf()
plot(np.arange(len(a))*1.0/8*100,a)
plot(wave(t)[:1000])
clf()
plot(wave(t)[:1000])
plot(np.arange(len(a))*1.0/8*10,a)
plot(np.arange(len(a))/8,a)
plot(wave(t)[:1000])
clf()
plot(wave(t)[:1000])
plot(np.arange(len(a))*1.0/8*10/60,a)
plot(np.arange(len(a))*1.0/8*10*60,a)
clf()
plot(np.arange(len(a))*1.0/8*10*60,a)
plot(np.arange(len(a))*1.0/8*10,a)
clf()
plot(np.arange(len(a))*1.0/8*10,a)
plot(np.arange(len(a))*1.0/8*10,a,'o')
print 1.0/8*10
t=np.linspace(0,200,1000000)
plot(t,wave(t))
plot(t,wave(t))
clf()
clf()
history


