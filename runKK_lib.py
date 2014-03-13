# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os

import hash_utils
import joblib_utils as ju
import numpy as np
import hybridata_creation_lib as hcl
import runspikedetekt_lib as rsd
import detection_statistics as ds
#from spikedetekt2.dataio import Experiment
from spikedetekt2 import *

#write_mask(M, basename+'.fmask.'+str(shank), fmt='%f')


def write_mask(mask, filename, fmt="%f"):
    fd = open(filename, 'w')
    fd.write(str(mask.shape[1])+'\n') # number of features
    np.savetxt(fd, mask, fmt=fmt)
    fd.close()

def write_fet(feats, filepath):
    feat_file = open(filepath, 'w')
    feats = np.array(feats, dtype=np.int32)
    #header line: number of features
    feat_file.write('%i\n' % feats.shape[1])
    #next lines: one feature vector per line
    np.savetxt(feat_file, feats, fmt="%i")
    feat_file.close()    

@ju.func_cache    
def make_KKscript(KKparams, filebase,scriptname):
    
    keylist = KKparams['keylist']
    #keylist = ['MaskStarts','MaxPossibleClusters','FullStepEvery','MaxIter','RandomSeed',
    #           'Debug','SplitFirst','SplitEvery','PenaltyK','PenaltyKLogN','Subset',
    #           'PriorPoint','SaveSorted','SaveCovarianceMeans','UseMaskedInitialConditions',
     #          'AssignToFirstClosestMask','UseDistributional']

    KKlocation = '/martinottihome/skadir/GIT_masters/klustakwik/MaskedKlustaKwik'  
    scriptstring = KKlocation + ' '+ filebase + ' 1 '
    for KKey in keylist: 
        #print '-'+KKey +' '+ str(KKparams[KKey])
        scriptstring = scriptstring + ' -'+ KKey +' '+ str(KKparams[KKey])
    
    print scriptstring
    scriptfile = open('%s.sh' %(scriptname),'w')
    scriptfile.write(scriptstring)
    scriptfile.close
    changeperms='chmod 777 %s.sh' %(scriptname)
    os.system(changeperms)
    
    return scriptstring

@ju.func_cache
def make_KKfiles_Script(hybdatadict, SDparams,prb, detectioncrit, KKparams):
    '''Creates the files required to run KlustaKwik'''
    argSD = [hybdatadict,SDparams,prb]
    if ju.is_cached(rsd.run_spikedetekt,*argSD):
        print 'Yes, SD has been run \n'
        hash_hyb_SD = rsd.run_spikedetekt(hybdatadict,SDparams,prb)
    else:
        print 'You need to run Spikedetekt before attempting to analyse results ' 
    
    
    argTD = [hybdatadict, SDparams,prb, detectioncrit]      
    if ju.is_cached(ds.test_detection_algorithm,*argTD):
        print 'Yes, you have run detection_statistics.test_detection_algorithm() \n'
        detcrit_groundtruth = ds.test_detection_algorithm(hybdatadict, SDparams,prb, detectioncrit)
    else:
        print 'You need to run detection_statistics.test_detection_algorithm() \n in order to obtain a groundtruth' 
    
    KKhash = hash_utils.hash_dictionary_md5(KKparams)
    baselist = [hash_hyb_SD, detcrit_groundtruth['detection_hashname'], KKhash]
    basefilename =  hash_utils.make_concatenated_filename(baselist)
    
    mainbasefilelist = [hash_hyb_SD, detcrit_groundtruth['detection_hashname']]
    mainbasefilename = hash_utils.make_concatenated_filename(mainbasefilelist)
    
    DIRPATH = hybdatadict['output_path']
    os.chdir(DIRPATH)
    with Experiment(hash_hyb_SD, dir= DIRPATH, mode='r') as expt:
        if KKparams['numspikesKK'] is not None: 
            fets = expt.channel_groups[0].spikes.features[0:KKparams['numspikesKK']]
            fmasks = expt.channel_groups[0].spikes.features_masks[0:KKparams['numspikesKK'],:,1]
            
            masks = expt.channel_groups[0].spikes.masks[0:KKparams['numspikesKK']]
        else: 
            fets = expt.channel_groups[0].spikes.features[:]
            fmasks = expt.channel_groups[0].spikes.features_masks[:,:,1]
            #print fmasks[3,:]
            masks = expt.channel_groups[0].spikes.masks[:]
    
    mainfetfile = DIRPATH + mainbasefilename+'.fet.1'
    mainfmaskfile = DIRPATH + mainbasefilename+'.fmask.1'
    mainmaskfile = DIRPATH + mainbasefilename+'.mask.1'
    
    if not os.path.isfile(mainfetfile):
        write_fet(fets,mainfetfile )
    else: 
        print mainfetfile, ' already exists, moving on \n '
        
    if not os.path.isfile(mainfmaskfile):
        write_mask(fmasks,mainfmaskfile,fmt='%f')
    else: 
        print mainfmaskfile, ' already exists, moving on \n '  
    
    if not os.path.isfile(mainmaskfile):
        write_mask(masks,mainmaskfile,fmt='%f')
    else: 
        print mainmaskfile, ' already exists, moving on \n '    
        
    
    
    os.system('ln -s %s %s.fet.1 ' %(mainfetfile,basefilename))
    os.system('ln -s %s %s.fmask.1 ' %(mainfmaskfile,basefilename))
    os.system('ln -s %s %s.mask.1 ' %(mainmaskfile,basefilename))
    
    KKscriptname = basefilename
    make_KKscript(KKparams,basefilename,KKscriptname)
    
    return basefilename

# <codecell>

if __name__== "__main__":    
    
    donordict = {'donor': 'n6mab031109', 'donorshanknum': 1, 'donorcluster': 25, 
             'donor_path':'/chandelierhome/skadir/hybrid_analysis/mariano/donors/',
                 'experiment_path': '/chandelierhome/skadir/hybrid_analysis/mariano/', 'donorcluid': 'MKKdistfloat'}
        
    time_size_dict = {'amplitude_generating_function_args':[1, 2],'amplitude_generating_function':hcl.make_uniform_amplitudes,
                      'donorspike_timeseries_generating_function':hcl.create_time_series_constant,
                      'sampling_rate':20000, 'firing_rate':3, 'start_time':10,'end_time':None,
                      'donorspike_timeseries_arguments': 'arg'}
                      
    accept_dict = {'acceptor_path':'/chandelierhome/skadir/hybrid_analysis/mariano/acceptors/',
                     'acceptor': 'n6mab041109_60sec.dat','numchannels':32,
                     'output_path':'/chandelierhome/skadir/hybrid_analysis/mariano/',
    
                     }     
    
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
    
    detectioncrit = {'allowed_discrepancy':2, 'CSthreshold': 0.8}
    
    hybdatadict = hcl.precreation_hybridict(donordict,accept_dict,time_size_dict)
    
    
    
    numspikesKK = None
    keylist = ['MaskStarts','MaxPossibleClusters','FullStepEvery','MaxIter','RandomSeed',
                   'Debug','SplitFirst','SplitEvery','PenaltyK','PenaltyKLogN','Subset',
                   'PriorPoint','SaveSorted','SaveCovarianceMeans','UseMaskedInitialConditions',
                   'AssignToFirstClosestMask','UseDistributional']
    
    #Default AIC parameters
    MaskStarts = 50
    MaxPossibleClusters =  500
    FullStepEvery =  1
    MaxIter = 10000
    RandomSeed =  654
    Debug = 0
    SplitFirst = 20 
    SplitEvery = 40 
    PenaltyK = 1
    PenaltyKLogN = 0
    Subset = 1
    PriorPoint = 1
    SaveSorted = 0
    SaveCovarianceMeans = 0
    UseMaskedInitialConditions = 1 
    AssignToFirstClosestMask = 1
    UseDistributional = 1


    KKparams = {'keylist': keylist,
                'numspikesKK': numspikesKK,
            'MaskStarts': MaskStarts,
            'MaxPossibleClusters':MaxPossibleClusters,
            'FullStepEvery': FullStepEvery,
             'MaxIter':MaxIter,
             'RandomSeed':RandomSeed,
             'Debug': Debug,
             'SplitFirst':SplitFirst,
             'SplitEvery':SplitEvery,
             'PenaltyK': PenaltyK,
              'PenaltyKLogN': PenaltyKLogN,
             'Subset' : Subset,
             'PriorPoint': PriorPoint,
              'SaveSorted' : SaveSorted, 
              'SaveCovarianceMeans' : SaveCovarianceMeans,
              'UseMaskedInitialConditions' : UseMaskedInitialConditions, 
              'AssignToFirstClosestMask': AssignToFirstClosestMask,
              'UseDistributional':UseDistributional
             
            }
    make_KKfiles_Script(hybdatadict, sdparams,prb, detectioncrit, KKparams)


    
   
    
    
        

