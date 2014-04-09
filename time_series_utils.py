import os
import hash_utils 
import scipy
import joblib_utils as ju
import numpy as np



@ju.func_cache
def create_time_series_constant(rate, samplerate, num_channels, start = 0, end = None, acceptor = None, buffersamples = None):
    '''Will create time series for constant rate,
       this will be cached and stored for future reference 
       when creating hybrid datasets and for analysis.
       e.g. acceptor = '/chandelierhome/skadir/hybrid_analysis/mariano/n6mab041109_60sec.dat'
    '''
    if (not(acceptor is None) and end is None) : 
        totalsamples = os.stat(acceptor).st_size/(2*num_channels)
        end = totalsamples
        
    end = end - buffersamples    
    betweensamps = round(samplerate/rate)    
    numspikes = round((end-start)/betweensamps)+1
    donorspike_timeseries = np.linspace(start, end, num = numspikes)
    return donorspike_timeseries

@ju.func_cache
def make_uniform_amplitudes(NumSpikes2Add, lower_bound, upper_bound):
    ''' returns an array called 
    amplitude_array = [0.3, 1.2, 0.4,..]
    whose shape is:
    amplitude_array.shape = (NumSpikes2Add,)
    
    '''
    amplitude_array = np.random.uniform(lower_bound, upper_bound, NumSpikes2Add)
    return amplitude_array
