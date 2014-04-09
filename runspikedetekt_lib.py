# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#A collection of functions for running SpikeDetekt
#All decorated with joblib
import hash_utils
import joblib_utils as ju
import hybridata_creation_lib as hcl
import os
#from spikedetekt2.dataio.kwik import (add_recording, create_files, open_files,
#    close_files, add_event_type, add_cluster_group, get_filenames,
#    add_cluster,files_exist)
from spikedetekt2 import *
# functions used are get_params, run
# classes used: Experiment
    
#----------------------------------------------------------------------    

def create_files_Experiment(filename, DIRPATH, prm, prb):
    #files_exist is a function in SD2.dataio.kwik
    if not files_exist(filename, dir=DIRPATH):
        create_files(filename, dir=DIRPATH, prm=prm, prb=prb)
    
        # Open the files.
        files = open_files(filename, dir=DIRPATH, mode='a')
    
        # Add data.
        add_recording(files, 
                  sample_rate=prm['sample_rate'],
                  nchannels=prm['nchannels'])
        add_cluster_group(files, channel_group_id='0', id='0', name='Noise')
        add_cluster(files, channel_group_id='0',)
    
        # Close the files
        close_files(files)

@ju.func_cache
def run_spikedetekt(hybdatadict,sdparams,prb):
    '''This function will call hash_hyb_SD(sdparams,hybdatadict) 
    and will run SpikeDetekt on the hybrid dataset specified by
    hybdatadict with the parameters sd params'''
    filename = hybdatadict['hashD']+'.kwd'
    DIRPATH = hybdatadict['output_path']
    
    
    # Make the product hash output name
    hashSDparams = hash_utils.hash_dictionary_md5(sdparams)
    # chose whether to include the probe, if so uncomment the two lines below
    #hashprobe = hash_utils.hash_dictionary_md5(prb)
    #hashdictlist = [ hybdatadict['hashD'],hashSDparams, hashprobe]
    hashdictlist = [hybdatadict['hashD'],hashSDparams]
    hash_hyb_SD_prb = hash_utils.make_concatenated_filename(hashdictlist)
    outputfilename = hash_hyb_SD_prb +'.kwd'
    
    # Need to create a symlink from hashD.kwd to hash_hyb_SD_prb.kwd 
    datasource = os.path.join(DIRPATH, filename )
    dest = os.path.join(DIRPATH, outputfilename )
    if not os.path.isfile(dest):
        os.symlink(datasource, dest)
    else: 
        print 'Warning: Symbolic link ',dest  ,' already exists'
    
    #Read in the raw data 
    raw_data = read_raw(dest,sdparams['nchannels'])
    
    create_files_Experiment(outputfilename, DIRPATH,  sdparams, prb)
    
    # Run SpikeDetekt2
    with Experiment(hash_hyb_SD_prb, dir= DIRPATH, mode='a') as exp:
        run(raw_data,experiment=exp,prm=sdparams,probe=Probe(prb))
    return hash_hyb_SD_prb

#@ju.func_cache
def run_SD_oneparamfamily(param2vary,paramrange,defaultSDparams, hybdatadict):
    ''' 
     param2vary is a particular user adjustable variable in Spikedetekt
    e.g. threshold_weak_std_factor 
    This function will loop over the values for param2vary over paramrange
    (a one parameter family) and call run_spikedetekt()
    with the defaultSDparams supplemented with the required 
    value for param2vary'''
    pass
    #return (all the hashes for each member of the family)
    


#---------------------------------------------------------------------------------------------
if __name__== "__main__":


    sample_rate = 20000
    duration = 1.
    nchannels = 32
    #chunk_size = 20000 automatically set below
    nsamples = int(sample_rate*duration)
    
    
    #--------------------LIST OF ALL PARAMETERS--------------------------------
    # Filtering
    # ---------
    filter_low = 500. # Low pass frequency (Hz)
    filter_high = 0.95 * .5 * sample_rate
    filter_butter_order = 3  # Order of Butterworth filter.
    
    # Chunks
    # ------
    chunk_size = int(1. * sample_rate)  # 1 second
    chunk_overlap = int(.015 * sample_rate)  # 15 ms
    
    # Spike detection
    # ---------------
    # Uniformly scattered chunks, for computing the threshold from the std of the
    # signal across the whole recording.
    nexcerpts = 50
    excerpt_size = int(1. * sample_rate)
    threshold_strong_std_factor = 4.5
    threshold_weak_std_factor = 2.
    detect_spikes = 'negative'
    #precomputed_threshold = None
    
    # Connected component
    # -------------------
    connected_component_join_size = int(.00005 * sample_rate)
    
    # Spike extraction
    # ----------------
    extract_s_before = 16
    extract_s_after = 16
    waveforms_nsamples = extract_s_before + extract_s_after
    
    # Features
    # --------
    nfeatures_per_channel = 3  # Number of features per channel.
    pca_nwaveforms_max = 10000
    
    #----------------------------------------------------------------------
    
    
    
    
    sdparams = get_params(**{
        'nchannels': nchannels,
        'sample_rate': sample_rate,
        'filter_low': filter_low,
        'filter_high':filter_high,
        'filter_butter_order':filter_butter_order,
        'chunk_size': chunk_size,
        'chunk_overlap':chunk_overlap ,
        'nexcerpts': nexcerpts, 
        'excerpt_size': excerpt_size, 
        'threshold_strong_std_factor': threshold_strong_std_factor,
        'threshold_weak_std_factor' : threshold_weak_std_factor, 
        'detect_spikes': detect_spikes,
        'connected_component_join_size' : connected_component_join_size,
        'extract_s_before' : extract_s_before,
        'extract_s_after': extract_s_after,
        'waveforms_nsamples': waveforms_nsamples,
        'nfeatures_per_channel': nfeatures_per_channel,
        'pca_nwaveforms_max': pca_nwaveforms_max
        
    })
    prb = {'channel_groups': [
        {
            'channels': range(nchannels),
            'graph': [
                        [0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [2, 4],
                        [3, 4], [3, 5], [4, 5], [4, 6], [5, 6], [5, 7],
                        [6, 7], [6, 8], [7, 8], [7, 9], [8, 9], [8, 10],
                        [9, 10], [9, 11], [10, 11], [10, 12], [11, 12], [11, 13],
                        [12, 13], [12, 14], [13, 14], [13, 15], [14, 15], [14, 16],
                        [15, 16], [15, 17], [16, 17], [16, 18], [17, 18], [17, 19],
                        [18, 19], [18, 20], [19, 20], [19, 21], [20, 21], [20, 22],
                        [21, 22], [21, 23], [22, 23], [22, 24], [23, 24], [23, 25],
                        [24, 25], [24, 26], [25, 26], [25, 27], [26, 27], [26, 28],
                        [27, 28], [27, 29], [28, 29], [28, 30], [29, 30], [29, 31],
                        [30, 31]
                    ],
        }
    ]}

    hybdatadict = {'hashD': '233124281b9bb0818d256d7b2ddc07ca_6a6f653c449505ddc30478cd5670f85c_b829bc190ca1a38622e5890c2dce883d', 
                   'donorspike_timeseries_generating_function': hcl.create_time_series_constant,
                   'numchannels': 32, 'donor_path': '/chandelierhome/skadir/hybrid_analysis/mariano/donors/', 'donorcluster': 41,
                   'donorcluid': 'MKKdistfloat', 'amplitude_generating_function': hcl.make_uniform_amplitudes, 
                   'start_time': 10, 'acceptor_path': '/chandelierhome/skadir/hybrid_analysis/mariano/acceptors/', 
                   'amplitude_generating_function_args': [0.5, 1.5], 'acceptor': 'n6mab041109_60sec.dat',
                   'output_path': '/chandelierhome/skadir/hybrid_analysis/mariano/', 'end_time': None, 
                   'experiment_path': '/chandelierhome/skadir/hybrid_analysis/mariano/', 'sampling_rate': 20000, 
                   'donor': 'n6mab031109', 'donorspike_timeseries_arguments': 'arg', 'donorshanknum': 1, 'firing_rate': 3}    
    
    #Run SpikeDetekt
    hash_hyb_SD = run_spikedetekt(hybdatadict,sdparams,prb)
    

