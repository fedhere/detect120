__author__ = 'Shi Fan'
import numpy as np
from scipy import signal
import pylab as pl

# default input: 120 Hz sine wave from 0 to 150 seconds (stepsize=1000) with magnitude of 1 and no phase
# output: a sine wave or square wave shifted up by its magnitude
def sim_wave(w_type='sine', 
	time=np.linspace(0, 150, 150*1000, endpoint=False), 
	s_freq=120, mag=1, phase=0):

	if w_type=='sine':
		wave = mag*np.sin(2*np.pi*s_freq*time+phase)+mag
	elif w_type=='square':
		wave = mag*signal.square(2*np.pi*s_freq*time+phase)+mag
	else:
		print 'Please enter sine or square for simulated wave type'
		return
	return wave

# default input: same as the sim_wave function
def folding_test(w_type='sine', 
	time=np.linspace(0, 150, 150*1000, endpoint=False), 
	s_freq=120, mag=1, phase=0):

	# simulate the wave first
	flux = sim_wave(w_type=w_type, time=time, s_freq=s_freq, mag=mag, phase=phase)

	# figure out where each point should be positioned at
	time_mod = (time)%(2/float(s_freq))
	sorted_idx = np.asarray([i[0] for i in sorted(enumerate(time_mod), key=lambda x:x[1])])

	# get the folded flux and time
	sim_time = 1000*time_mod[sorted_idx] # return time in microseconds
	sim_flux = flux[sorted_idx]

	# plot
	pl.figure(figsize=(20,5))
	pl.plot(sim_time, sim_flux, '-', color='deepskyblue', linewidth=0.75)
	pl.plot(sim_time, sim_flux, 'o', color='deepskyblue')
	pl.xlabel("Microseconds", fontsize=15)
	pl.ylabel("Almost Normalized Flux (Folded)", fontsize=15)
	pl.title('Folded Flux of Simulated Wave: Shutter '+str(s_freq)+'Hz', fontsize=20)
	pl.xlim(0,2000/float(s_freq))
	pl.ylim(-.1,mag*2+.1)
	pl.show()
	return

### TO BE FINISHED
# this will be a more customized function to fold our actual fluxes
# default input: empty flux (must be specified) with shutter frequency 120 Hz and camera frequency 4 Hz
def folding(flux={}, s_freq=120, c_freq=4):
	if flux=={}:
		print 'Flux must be specified (as a dict)'
		return
	time = np.linspace(0, len(flux[s_freq])/c_freq, len(flux[s_freq]), endpoint=True)
	time_mod = (1000*time)%(2/float(s_freq))
	sorted_idx = np.asarray([i[0] for i in sorted(enumerate(time_mod), key=lambda x:x[1])])
	return flux[sorted_idx], 1000*time_mod[sorted_idx] # return time in microseconds