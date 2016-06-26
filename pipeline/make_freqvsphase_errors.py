from __future__ import print_function
import glob
import numpy as np
import pylab as pl

for f in glob.glob("groundtest1/N0100W1533S0150/*goodcoords*mcmc*.npy"):
    print (f)
    gc = np.load(f)
    #print f,"+",gc[7][i],"-",gc[8][i],"  " ,gc[6][i],+gc[9][i],"-",gc[10][i]
    pl.errorbar(gc[0], gc[6], xerr=[gc[7],gc[8]], yerr=[gc[9],gc[10]],
                fmt='.', color='k')
    #pl.xticks(np.arange(2.0,4.0,0.5),
    #          ["%.1f"%i for i in np.arange(2.0,4.0,0.5)])#-2.0])
    
    pl.title (f.split('/')[-1].replace(".npy",""), fontsize=13)
    pl.ylabel("frequency", fontsize=15)
    pl.xlabel("phase", fontsize=15)
    pl.xlim(-0.1,2.1)
    pl.savefig(f.replace("goodcoords_mcmc.npy", "frequencyVSphaseMCMC.pdf"))
    #pl.show()
    print (gc[8]+gc[7])
    print ("phase error mean %.2f"%(gc[8]+gc[7]).mean())
    print ("phase error std %.2f"%(gc[8]+gc[7]).std())
    print ("frequency error mean %.4f"%(gc[10]+gc[9]).mean())
    print ("frequency error std %.4f"%(gc[10]+gc[9]).std())
        
        
        
