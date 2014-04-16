# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>
import matplotlib.pyplot as plt
import os
import hash_utils 
import scipy
import joblib_utils as ju
import numpy as np
import collections as col
import tables as tb
import time_series_utils as tsu
from spikedetekt2.dataio import (DatRawDataReader, create_kwd, convert_dat_to_kwd, add_recording_in_kwd,read_raw)
from kwiklib.dataio.tools import (load_binary, normalize)
from kwiklib.dataio import klustersloader
from scipy.interpolate import interp1d

@ju.func_cache
def merge_input_dicts(dict1,dict2):
    '''Utility function for merging dictionaries'''
    merged_dict = dict(dict1.items() + dict2.items())
    return merged_dict

@ju.func_cache
def interpolated_waveshift(nChannels,waveform,littleshift,knownsample_times):
    '''Utility function for interpolated a wave shifted by a fraction of 
    a sample
    littleshift is a scalar'''
    outputwaveform = np.zeros((waveform.shape[0]-1,waveform.shape[1]))
    #print knownsample_times.shape
    #print littleshift.shape
    shifted = knownsample_times+littleshift
    newshifted = shifted[0:(knownsample_times.shape[0]-1)]
    #print newshifted
    #print newshifted.shape
    #knownsample_timesplus = 
    for i in np.arange(nChannels):
        #plt.plot(knownsample_times,waveform[:,i],'o')
        #plt.show()
        interpfunc =interp1d(knownsample_times,waveform[:,i],kind = 'cubic')
        outputwaveform[:,i] = interpfunc(newshifted)
       
        
    return outputwaveform

@ju.func_cache
def convert_tuple_to_dict(keytuple, valuetuple):
    '''Will convert my list of tuples into dictionaries of the same length
        adding the appropriate keys, e.g.
        donorkeytuple = (donor,donorcluid,donorcluster)
        donorvalue = ('n6mab031109', 'MKKdistfloat',54)
        will turn into a dictionary like:
        converted_donordict = {'donor': 'n6mab031109', 'donorcluid': 'MKKdistfloat', 'donorcluster': 54}
        (will append output to listofcreationtuples.py)
        converted_dict = dict(zip(keylist, valuelist))'''
    converted_dict = dict(zip(keytuple,valuetuple))
    return converted_dict

#@ju.func_cache
#def create_time_series_constant(rate, samplerate, num_channels, start = 0, end = None, acceptor = None, buffersamples = None):
#    '''Will create time series for constant rate,
#       this will be cached and stored for future reference 
#       when creating hybrid datasets and for analysis.
#       e.g. acceptor = '/chandelierhome/skadir/hybrid_analysis/mariano/n6mab041109_60sec.dat'
#    '''
#    if (not(acceptor is None) and end is None) : 
#        totalsamples = os.stat(acceptor).st_size/(2*num_channels)
#        end = totalsamples
#        
#    end = end - buffersamples    
#    betweensamps = round(samplerate/rate)    
#    numspikes = round((end-start)/betweensamps)+1
#    donorspike_timeseries = np.linspace(start, end, num = numspikes)
#    return donorspike_timeseries

#@ju.func_cache
#def make_uniform_amplitudes(NumSpikes2Add, lower_bound, upper_bound):
#    ''' returns an array called 
#    amplitude_array = [0.3, 1.2, 0.4,..]
#    whose shape is:
#    amplitude_array.shape = (NumSpikes2Add,)
#    
#    '''
#    amplitude_array = np.random.uniform(lower_bound, upper_bound, NumSpikes2Add)
#    return amplitude_array

#@ju.func_cache
#def make_uniform_amplitudes(NumSpikes2Add, lower_bound, upper_bound):
#    ''' returns an array called 
#    amplitude_array = [0.3, 1.2, 0.4,..]
#    whose shape is:
 #   amplitude_array.shape = (NumSpikes2Add,)
    
#    '''
 #   amplitude_array = np.random.uniform(lower_bound, upper_bound, NumSpikes2Add)
#    return amplitude_array

@ju.func_cache
def create_kwd_from_dat(kwdfilenamepath,datfilenamepath, prm=None):#recordings=None,):
    """Add data from a .dat file to a .kwd file
       Minimally we require: 
       prm = {'nchannels':nchannels, 'chunk_size':20000} 
    """
    
    nchannels = prm.get('nchannels')
    #chunk_size = prm.get('chunk_size')
    # Create a DatRawDataReader instance - see SD2/dataio/raw.py
    data_readin = DatRawDataReader([datfilenamepath],dtype = np.int16, dtype_to = np.int16, shape = (-1,nchannels))
    
    #create an empty .kwd file with the specified path
    create_kwd(kwdfilenamepath)
    #convert from .dat to the newly created .kwd file
    convert_dat_to_kwd(data_readin,kwdfilenamepath)


@ju.func_cache
def create_average_hybrid_wave_spike(donor_dict):
    '''Input a dictionary donor_dict or a hybdatadict
      will  create the mean spike file from taking the average of
      spikes in a donor cluster of the donor dataset
      meanspike_file_id = a file called donor_id.msua.1.'''
    clufilename = donor_dict['donor_path'] + donor_dict['donor']+'_'+ donor_dict['donorcluid']+'.clu.1'
    spkfilename = donor_dict['donor_path']+ donor_dict['donor']+'.spk.1'
    uspkfilename = donor_dict['donor_path']+ donor_dict['donor']+'.uspk.1'
    clusters = klustersloader.read_clusters(clufilename)
    #uspk = klustersloader.read_waveforms(uspkfilename,20,32)
    ##print uspkfilename
    uspk = np.array(load_binary(uspkfilename), dtype = np.int16)
   # uspk = np.array(uspk, dtype = np.int16)
   # uspk = normalize(uspk, symmetric = True)
    
    uspk = uspk.reshape((-1, 20, 32))
    
    #uspk = np.transpose(uspk, (0,2,1))
    
    #print uspk[1,:,:]
    #print 'the 8th column, 8th channel recording ', uspk[1,:,7]
    
    #print uspk.shape
 
    #print ' uspk.dtype =  ' , uspk.dtype
    #output [False, True, False, False, ...] where True indicates 
    #the indices corresponding to the chosen donor cluster
    selected_cluster_indices = np.in1d(clusters, donor_dict['donorcluster'])
    num_selected_spikes = np.sum(selected_cluster_indices)
    selected_spikes = uspk[selected_cluster_indices,:,:]
    
    #print 'second selected spike ' , selected_spikes[20,:,:]
    #print 'the 8th channel recording of a selected uspk ', selected_spikes[20,:,7]
    
    #print 'number of selected spikes ', num_selected_spikes
    #print ' sum of spikes ', np.sum(selected_spikes,axis = 0)
    donor_id_msua_data  = np.sum(selected_spikes,axis = 0)/num_selected_spikes
    #print donor_id_msua_data
    #print donor_id_msua_data.dtype
    
    return donor_id_msua_data

@ju.func_cache
def make_average_datamask_from_mean(donor_dict, fmask = True):
    '''Will create the mean mask file from the donor_dict or hybdatadict information
      meanmask_file = a file called donor_id.amsk.1.
       This will be used when computing the spike similarity measure
       to assess the quality of spike detection'''
    clufilename = donor_dict['donor_path'] + donor_dict['donor']+'_'+ donor_dict['donorcluid']+'.clu.1'

    clusters = klustersloader.read_clusters(clufilename)
    if fmask == True: 
        fmaskfilename = donor_dict['donor_path']+ donor_dict['donor']+'.fmask.1'
    else:
        fmaskfilename = donor_dict['donor_path']+ donor_dict['donor']+'.mask.1'
    [fmasks,fmasks_full] = klustersloader.read_masks(fmaskfilename,donor_dict['numchannels'])
    #output [False, True, False, False, ...] where True indicates 
    #the indices corresponding to the chosen donor cluster
    selected_cluster_indices = np.in1d(clusters, donor_dict['donorcluster'])
    num_selected_spikes = np.sum(selected_cluster_indices)
    selected_fmasks = fmasks_full[selected_cluster_indices,:]
    donor_id_amsk_data  = np.sum(selected_fmasks,axis = 0)/num_selected_spikes     
    return donor_id_amsk_data

@ju.func_cache
def precreation_hybridict_ordered(donor_dict_dis,acceptor_dict_dis,time_size_dict_dis):
    ''' Creates one dictionary hybdatadict out of three
    
    The inputs are:
        
        donordict = {'donor': 'n6mab031109', 'donorshanknum': 1, 'donorcluster': 54, 
             'donor_path':'/chandelierhome/skadir/hybrid_analysis/mariano/donors/',
                 'experiment_path': '/chandelierhome/skadir/hybrid_analysis/mariano/', 'donorcluid': 'MKKdistfloat'}
        
        time_size_dict = {'amplitude_generating_function_args':[0.5, 1.5],'amplitude_generating_function':make_uniform_amplitudes,
                  'donorspike_timeseries_generating_function':create_time_series_constant,
                  'sampling_rate':20000, 'firing_rate':3, 'start_time':10,'end_time':None,
                  'donorspike_timeseries_arguments': 'arg'}
                  
        acceptor_dict = {'acceptor_path':'/chandelierhome/skadir/hybrid_analysis/mariano/acceptors/',
                 'acceptor': 'n6mab041109_60sec.dat','numchannels':32,
                 'output_path':'/chandelierhome/skadir/hybrid_analysis/mariano/',
                 }    
    
    '''
    donor_dict = hash_utils.order_dictionary(donor_dict_dis)
    acceptor_dict = hash_utils.order_dictionary(acceptor_dict_dis)
    time_size_dict = hash_utils.order_dictionary(time_size_dict_dis)
    
    hashDlist = hash_utils.get_product_hashlist([donor_dict,acceptor_dict,time_size_dict])
    hashD = hash_utils.make_concatenated_filename(hashDlist)
    hybdatadict_dis = merge_input_dicts(donor_dict,merge_input_dicts(acceptor_dict,time_size_dict))
    hybdatadict_dis['hashD']= hashD
    
    hybdatadict = hash_utils.order_dictionary(hybdatadict_dis)
    
    return hybdatadict

@ju.func_cache
def precreation_hybridict(donor_dict,acceptor_dict,time_size_dict):
    ''' Creates one dictionary hybdatadict out of three
    
    The inputs are:
        
        donordict = {'donor': 'n6mab031109', 'donorshanknum': 1, 'donorcluster': 54, 
             'donor_path':'/chandelierhome/skadir/hybrid_analysis/mariano/donors/',
                 'experiment_path': '/chandelierhome/skadir/hybrid_analysis/mariano/', 'donorcluid': 'MKKdistfloat'}
        
        time_size_dict = {'amplitude_generating_function_args':[0.5, 1.5],'amplitude_generating_function':make_uniform_amplitudes,
                  'donorspike_timeseries_generating_function':create_time_series_constant,
                  'sampling_rate':20000, 'firing_rate':3, 'start_time':10,'end_time':None,
                  'donorspike_timeseries_arguments': 'arg'}
                  
        acceptor_dict = {'acceptor_path':'/chandelierhome/skadir/hybrid_analysis/mariano/acceptors/',
                 'acceptor': 'n6mab041109_60sec.dat','numchannels':32,
                 'output_path':'/chandelierhome/skadir/hybrid_analysis/mariano/',
                 }    
    
    '''
    
    
    hashDlist = hash_utils.get_product_hashlist([donor_dict,acceptor_dict,time_size_dict])
    hashD = hash_utils.make_concatenated_filename(hashDlist)
    hybdatadict = merge_input_dicts(donor_dict,merge_input_dicts(acceptor_dict,time_size_dict))
    hybdatadict['hashD']= hashD
    print 'Printing during execution changed'
    return hybdatadict

@ju.func_cache
def make_hamming_spike(avespike):
    avespikeminusmean = np.zeros_like(avespike)
    SamplesPerSpike = avespike.shape[0]
    avespikechannelmean = np.zeros(avespike.shape[1])
    for chan in np.arange(SamplesPerSpike):
        avespikechannelmean[chan] = np.true_divide(np.sum(avespike[:,chan]),SamplesPerSpike)
        avespikeminusmean[:,chan] = avespike[:,chan] - avespikechannelmean[chan] 
        
    prehamspike = np.zeros_like(avespike)
    hamspike = np.zeros_like(avespike)
    #hamspike.shape = (number of samples per spike,nChannels) Check this! 
    ham = scipy.signal.hamming(SamplesPerSpike-2)
    hammy = np.expand_dims(ham,axis = 1)
    hamnut = np.zeros((SamplesPerSpike,1))
    hamnut[1:SamplesPerSpike-1] = hammy
    for s in np.arange(SamplesPerSpike):
        prehamspike[s,:] = avespikeminusmean[s,:]*hamnut[s]
        
    return prehamspike    



@ju.func_cache
def create_hybrid_kwdfile(hybdatadict):
    ''' This function outputs a file called:
        Hash(hybdatadict = [donor_dict,acceptor_dict,time_size_dict]).kwd,
        
        The input is hybdatadict (an ordered dictionary) which is the output of 
        
        hybdatadict = precreation_hybridict(donor_dict,acceptor_dict,time_size_dict)
                        
        
        it adds a mean waveform to an acceptor .dat file at specified groundtruth times
        It returns the creation groundtruth which is equivalent
        to the old 
        regular3.res.1 file of the times of added spikes
        (i.e. the times and the cluster labels form the added
        hybrid spikes. 
        It also returns hybdatadict = donordict U time_size_dict U acceptor_dict
        '''    
    avespike = create_average_hybrid_wave_spike(hybdatadict)
    SamplesPerSpike = avespike.shape[0]
    
    prehamspike = make_hamming_spike(avespike)
    ##NEW WAY-----------------------------------------
    ##avespikeminusmean = np.zeros_like(avespike)
    ##SamplesPerSpike = avespike.shape[0]
    ##avespikechannelmean = np.zeros(avespike.shape[1])
    ##for chan in np.arange(SamplesPerSpike):
    ##    avespikechannelmean[chan] = np.true_divide(np.sum(avespike[:,chan]),SamplesPerSpike)
    ##    avespikeminusmean[:,chan] = avespike[:,chan] - avespikechannelmean[chan] 
        
    ##prehamspike = np.zeros_like(avespike)
    ##hamspike = np.zeros_like(avespike)
    
    
    ##ham = scipy.signal.hamming(SamplesPerSpike-2)
    ##hammy = np.expand_dims(ham,axis = 1)
    ##hamnut = np.zeros((SamplesPerSpike,1))
    ##hamnut[1:SamplesPerSpike-1] = hammy
    ##for s in np.arange(SamplesPerSpike):
    ##    prehamspike[s,:] = avespikeminusmean[s,:]*hamnut[s]
    ##NEW WAY-----------------------------------------    
        
    #print 'avespike ' , avespike    
    #print 'prehamspike ' , prehamspike    
        
    ##NEW WAY-----------------------------------------
    #avespike.shape[1] = number of samples per spike
    #SamplesPerSpike = avespike.shape[1]
    #ham = scipy.signal.hamming(SamplesPerSpike-2)
    #hammy = np.expand_dims(ham,axis = 1)
    #hamnut = np.zeros((SamplesPerSpike,1))
    #hamnut[1:SamplesPerSpike-1] = hammy
    #for i in np.arange(SamplesPerSpike):
    #    prehamspike[:,i] = avespike[:,i]*hamnut[i]    
        
        
        
        
    # Typical generating function is: 
    #create_time_series_constant(rate, samplerate, num_channels, start = 0, end = None, acceptor = None)
    print 'acceptor path is ', hybdatadict['acceptor_path']
    acceptordat = hybdatadict['acceptor_path']+hybdatadict['acceptor']
    stringtimeforeval =  hybdatadict['donorspike_timeseries_generating_function']+'(hybdatadict[\'firing_rate\'], hybdatadict[\'sampling_rate\'],hybdatadict[\'numchannels\'],hybdatadict[\'start_time\'],hybdatadict[\'end_time\'],acceptor = acceptordat, buffersamples = np.ceil(SamplesPerSpike/2))'
    print stringtimeforeval
#    donorspike_timeseries = hybdatadict['donorspike_timeseries_generating_function'](hybdatadict['firing_rate'], 
#                                                                                        hybdatadict['sampling_rate'],
#                                                        hybdatadict['numchannels'],
#                                                        hybdatadict['start_time'],
#                                                        hybdatadict['end_time'],
#                                                         acceptor = acceptordat, buffersamples = np.ceil     #(SamplesPerSpike/2))                                                
    donorspike_timeseries = eval(stringtimeforeval)                                                  
    # NOTE: The buffer ensures that the spikes can fit inside the acceptor dat file.
    #donorspike_timeseries.shape = (NumSpikes2Add, )
    NumSpikes2Add = donorspike_timeseries.shape[0] #Number of spikes be added
    times_to_start_adding_spikes = donorspike_timeseries - SamplesPerSpike/2 #Subtract half the number of samples
    #per spike
    #print times_to_start_adding_spikes
    fractional_times = np.ceil(times_to_start_adding_spikes) - times_to_start_adding_spikes
    #print fractional_times
    
    
    
    #data_readin = DatRawDataReader([acceptordat],dtype = np.int16, dtype_to = np.int16, shape = (-1,nchannels))
    kwdoutputname = hybdatadict['output_path']+hybdatadict['hashD']+'.kwd'
    print kwdoutputname

    prm = {'nchannels': hybdatadict['numchannels']}
    create_kwd_from_dat(kwdoutputname,acceptordat,prm)
    #tb.close(kwdoutputname)
    with tb.openFile(kwdoutputname, mode = 'a') as kwdfile:
        kwdfile = tb.openFile(kwdoutputname, mode = 'a')
        rawdata = kwdfile.root.recordings._f_getChild('0').data
        #rawdata.shape = (Number of samples, nChannels)
        
    
        
        amp_gen_args = [NumSpikes2Add]
        amp_gen_args.extend(hybdatadict['amplitude_generating_function_args'])
        #amplitude = hybdatadict['amplitude_generating_function'](*amp_gen_args)
        evalampstring =  hybdatadict['amplitude_generating_function']+'(*amp_gen_args)'
        print evalampstring
        amplitude = eval(evalampstring)
        # amplitude.shape = (NumSpikes2Add,)
        
        print 'Adding ', NumSpikes2Add, ' spikes'
        for i in range(NumSpikes2Add): 
            if np.all(fractional_times[i]==0):
                hamspike = prehamspike
            else:
                #OLD WAY---------------------------------------------------------------------------
                knownsample_times = np.arange(SamplesPerSpike+1)
                lastrowprehamspike = prehamspike[SamplesPerSpike-1,:].reshape((1,hybdatadict['numchannels']))
                
                #print prehamspike.shape
                #print lastrowprehamspike.shape
                appendedprehamspike = np.concatenate((prehamspike,lastrowprehamspike), axis = 0)
                #print appendedprehamspike.shape
                #Add one to prevent interpolation error: (ValueError: A value in x_new is above the interpolation range.)
                hamspike = interpolated_waveshift(hybdatadict['numchannels'],appendedprehamspike,fractional_times[i],knownsample_times) 
                
                ##NEW WAY ------------------------------------------------------------------------------
                ##knownsample_times = np.arange(SamplesPerSpike+1)
                ##lastrowprehamspike = prehamspike[:,SamplesPerSpike-1].reshape((1,hybdatadict['numchannels']))
                
                ##print prehamspike.shape
                ##print lastrowprehamspike.shape
                ##appendedprehamspike = np.concatenate((prehamspike,lastrowprehamspike), axis = 1)
                ##print appendedprehamspike.shape
                ##Add one to prevent interpolation error: (ValueError: A value in x_new is above the interpolation range.)
                #hamspike = interpolated_waveshift(hybdatadict['numchannels'],appendedprehamspike,fractional_times[i],knownsample_times) 
                
            for j in range(SamplesPerSpike):        
                #print '(i,j)',i,j
                #rawdata[np.ceil(times_to_start_adding_spikes[i])+j,:] = rawdata[np.ceil(times_to_start_adding_spikes[i])+j,:]+ amplitude[i]*hamspike[j,:]
                #print 'before ', rawdata[np.ceil(times_to_start_adding_spikes[i])+j,:]
                #print 'adding ', amplitude[i]*hamspike[j,:]
                rawdata[np.ceil(times_to_start_adding_spikes[i])+j,:] += amplitude[i]*hamspike[j,:]
                #print rawdata[np.ceil(times_to_start_adding_spikes[i])+j,:]
                
        rawdata.flush()
    #kwdfile.close()
    creation_groundtruth = donorspike_timeseries 
    return hamspike, kwdoutputname, creation_groundtruth, amplitude

def plot_spike(graphpath,avehamspike,ifshow):
    fig1 = plt.figure(1)
    axes1 = fig1.add_axes([0.1,0.1,0.8,0.8])
    axes1.set_xlabel('Samples')
    #axes1.set_ylabel('')

    numchans = avehamspike.shape[1]
    axes1.hold(True)
    const = 500
    for chan in np.arange(numchans):
        #axis = plt.subplot(hybdatadict['numchannels'],1,i+1)
        axes1.plot(avehamspike[:,chan]+const*chan)
    
    if ifshow ==True:
        plt.show()
    fig1.savefig('%s.pdf'%(graphpath))
    return


# Obsolete function below    
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
@ju.func_cache
def create_hybrid_kwdfile_old(donor_dict,acceptor_dict,time_size_dict):
    ''' This function outputs a file called:
        Hash(hybdatadict = [donor_dict,acceptor_dict,time_size_dict]).kwd,
        
        The inputs are:
        
        donordict = {'donor': 'n6mab031109', 'donorshanknum': 1, 'donorcluster': 54, 
             'donor_path':'/chandelierhome/skadir/hybrid_analysis/mariano/donors/',
                 'experiment_path': '/chandelierhome/skadir/hybrid_analysis/mariano/', 'donorcluid': 'MKKdistfloat'}
        
        time_size_dict = {'amplitude_generating_function_args':[0.5, 1.5],'amplitude_generating_function':make_uniform_amplitudes,
                  'donorspike_timeseries_generating_function':create_time_series_constant,
                  'sampling_rate':20000, 'firing_rate':3, 'start_time':10,'end_time':None,
                  'donorspike_timeseries_arguments': 'arg'}
                  
        acceptor_dict = {'acceptor_path':'/chandelierhome/skadir/hybrid_analysis/mariano/acceptors/',
                 'acceptor': 'n6mab041109_60sec.dat','numchannels':32,
                 'output_path':'/chandelierhome/skadir/hybrid_analysis/mariano/',
                 }    
                        
        
        it adds a mean waveform to an acceptor .dat file at specified groundtruth times
        It returns the creation groundtruth which is equivalent
        to the old 
        regular3.res.1 file of the times of added spikes
        (i.e. the times and the cluster labels form the added
        hybrid spikes. 
        It also returns hybdatadict = donordict U time_size_dict U acceptor_dict
        '''
    hashDlist = hash_utils.get_product_hashlist([donor_dict,acceptor_dict,time_size_dict])
    hashD = hash_utils.make_concatenated_filename(hashDlist)
    hybdatadict = merge_input_dicts(donor_dict,merge_input_dicts(acceptor_dict,time_size_dict))
    hybdatadict['hashD']= hashD
    
    avespike = create_average_hybrid_wave_spike(donor_dict)
    prehamspike = np.zeros_like(avespike)
    hamspike = np.zeros_like(avespike)
    #hamspike.shape = (number of samples per spike,nChannels) Check this!
    #avespike.shape[0] = number of samples per spike
    SamplesPerSpike = avespike.shape[0]
    ham = scipy.signal.hamming(SamplesPerSpike-2)
    hammy = np.expand_dims(ham,axis = 1)
    hamnut = np.zeros((SamplesPerSpike,1))
    hamnut[1:SamplesPerSpike-1] = hammy
    for i in np.arange(SamplesPerSpike):
        prehamspike[i,:] = avespike[i,:]*hamnut[i]
        
    
        
    # Typical generating function is: 
    #create_time_series_constant(rate, samplerate, num_channels, start = 0, end = None, acceptor = None)
    print 'acceptor path is ', acceptor_dict['acceptor_path']
    acceptordat = acceptor_dict['acceptor_path']+acceptor_dict['acceptor']
    donorspike_timeseries = time_size_dict['donorspike_timeseries_generating_function'](time_size_dict['firing_rate'], 
                                                                                        time_size_dict['sampling_rate'],
                                                        acceptor_dict['numchannels'],
                                                        time_size_dict['start_time'],
                                                        time_size_dict['end_time'],
                                                         acceptor = acceptordat, buffersamples = np.ceil(SamplesPerSpike/2))
    # NOTE: The buffer ensures that the spikes can fit inside the acceptor dat file.
    #donorspike_timeseries.shape = (NumSpikes2Add, )
    NumSpikes2Add = donorspike_timeseries.shape[0] #Number of spikes be added
    times_to_start_adding_spikes = donorspike_timeseries - SamplesPerSpike/2 #Subtract half the number of samples
    #per spike
    #print times_to_start_adding_spikes
    fractional_times = np.ceil(times_to_start_adding_spikes) - times_to_start_adding_spikes
    #print fractional_times
    
    
    
    #data_readin = DatRawDataReader([acceptordat],dtype = np.int16, dtype_to = np.int16, shape = (-1,nchannels))
    kwdoutputname = acceptor_dict['output_path']+hashD+'.kwd'
    print kwdoutputname

    prm = {'nchannels': acceptor_dict['numchannels']}
    create_kwd_from_dat(kwdoutputname,acceptordat,prm)
    #tb.close(kwdoutputname)
    with tb.openFile(kwdoutputname, mode = 'a') as kwdfile:
        kwdfile = tb.openFile(kwdoutputname, mode = 'a')
        rawdata = kwdfile.root.recordings._f_getChild('0').data
        #rawdata.shape = (Number of samples, nChannels)
        
    
        
        amp_gen_args = [NumSpikes2Add]
        amp_gen_args.extend(time_size_dict['amplitude_generating_function_args'])
        amplitude = time_size_dict['amplitude_generating_function'](*amp_gen_args)
        # amplitude.shape = (NumSpikes2Add,)
        
        print 'Adding ', NumSpikes2Add, ' spikes'
        for i in range(NumSpikes2Add): 
            if np.all(fractional_times[i]==0):
                hamspike = prehamspike
            else:
                knownsample_times = np.arange(SamplesPerSpike+1)
                lastrowprehamspike = prehamspike[SamplesPerSpike-1,:].reshape((1,acceptor_dict['numchannels']))
                
                #print prehamspike.shape
                #print lastrowprehamspike.shape
                appendedprehamspike = np.concatenate((prehamspike,lastrowprehamspike), axis = 0)
                #print appendedprehamspike.shape
                #Add one to prevent interpolation error: (ValueError: A value in x_new is above the interpolation range.)
                hamspike = interpolated_waveshift(acceptor_dict['numchannels'],appendedprehamspike,fractional_times[i],knownsample_times) 
            for j in range(SamplesPerSpike):        
                #print '(i,j)',i,j
                #rawdata[np.ceil(times_to_start_adding_spikes[i])+j,:] = rawdata[np.ceil(times_to_start_adding_spikes[i])+j,:]+ amplitude[i]*hamspike[j,:]
                rawdata[np.ceil(times_to_start_adding_spikes[i])+j,:] += amplitude[i]*hamspike[j,:]
                
        rawdata.flush()
    #kwdfile.close()
    creation_groundtruth = donorspike_timeseries 
    return hybdatadict,kwdoutputname, creation_groundtruth, amplitude




    
# Obsolete function below    
#-------------------------------------------------------------------------------------    
def create_kwd_from_dat_old(kwdfilenamepath,datfilenamepath, prm=None):#recordings=None,):
    """Add data from a .dat file to a .kwd file
       Minimally we require: 
       prm = {'nchannels':nchannels} 
    """
        
    if prm is None:
        print 'Error: Please specify the number of channels as the value of the nchannels key in prm' 
    
    nchannels = prm.get('nchannels')
    datsize = os.path.getsize(datfilenamepath)
    datsamples = datsize/(2.*nchannels)
    #print datsamples
    
    data = np.fromfile(datfilenamepath,dtype = np.int16)
    if data is not None: 
        data.shape = datsamples,nchannels
    
    
    filen = tb.openFile(kwdfilenamepath, mode='a')
    if not filen.__contains__('/'+ 'recordings'):
        filen.createGroup('/', 'recordings')
        
    record = filen.root.recordings
    
    datasetearray = filen.createEArray(record,'data',tb.Int16Atom(),
                                 (0,nchannels),expectedrows = datsamples)
    
    #Add raw data    
    if data is not None:     
        datasetearray.append(data)
    
    filen.close()
    return datsize, datsamples


if __name__== "main":
    donordict = {'donor': 'n6mab031109', 'donorshanknum': 1, 'donorcluster': 54, 
             'donor_path':'/chandelierhome/skadir/hybrid_analysis/mariano/donors/',
                 'experiment_path': '/chandelierhome/skadir/hybrid_analysis/mariano/', 'donorcluid': 'MKKdistfloat'}
        
    time_size_dict = {'amplitude_generating_function_args':[0.5, 1],'amplitude_generating_function':make_uniform_amplitudes,
                  'donorspike_timeseries_generating_function':create_time_series_constant,
                  'sampling_rate':20000, 'firing_rate':3, 'start_time':10,'end_time':None,
                  'donorspike_timeseries_arguments': 'arg'}
                  
    accept_dict = {'acceptor_path':'/chandelierhome/skadir/hybrid_analysis/mariano/acceptors/',
                 'acceptor': 'n6mab041109_60sec.dat','numchannels':32,
                 'output_path':'/chandelierhome/skadir/hybrid_analysis/mariano/',


                 }   
    #Example of hybrid creation 

    #create_hybrid_kwdfile(donordict,accept_dict,time_size_dict)
    hybdatadict = precreation_hybridict(donordict,accept_dict,time_size_dict)
    meanwaveform, kwdoutputname, creation_groundtruth, amplitude = create_hybrid_kwdfile(hybdatadict)
    

