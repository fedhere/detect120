import multiprocessing as mpc
import os
import itertools
import matplotlib
matplotlib.use('agg')

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def f(name):
    info('function f')
    pl.figure()
    for i in range(100000000):
        i
    print('hello', name)

#if __name__ == '__main__':
    info('main line')
    for i in range (10):
        p = Process(target=f, args=('bob',))
        p.start()
    p.join()

from multiprocessing import Pool
import numpy as np
import pylab as pl
def f((x, strhere)):
    pl.figure()
    for i in range(100**x):
        i
    print (strhere, i, x)
    return np.random.randn(x*x),np.random.randn(x+1)

if __name__ == '__main__':
    #print (f((1,"here")))
    p = Pool(5)
    #for i in itertools.izip(range(3), itertools.repeat("here")): print i
    x =  (p.map(f, (itertools.izip(range(3,0, -1), itertools.repeat("here")))))
    x  = np.array(x)
    print (x.shape)
    print x[0][0]
    
