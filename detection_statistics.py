# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os
import hash_utils
import joblib_utils as ju
import numpy as np
import hybridata_creation_lib as hcl
import runspikedetekt_lib as rsd
from spikedetekt2 import *
from kwiklib.dataio import klustersloader
from kwiklib.dataio import selection 
#from IPython import embed

def add_clustering_kwik(fh,clustering_array, clustering_name):
    with tb.openFile(fh, mode = 'a') as kwikfile: 
        clusters = '/channel_groups/0/spikes/clusters'
        #kwikfile.createEArray(clusters, clustering_name, tb.UInt32Atom(), (0,),
                    #      expectedrows=clustering_array.shape[0], obj = clustering_array)
        #embed()
        #print clusters
        pathname = clusters + '/'+ clustering_name
        if kwikfile.__contains__(pathname):
            print pathname, ' exists already, no need to create'
        else: 
            kwikfile.createEArray(clusters, clustering_name, tb.UInt32Atom(), (0,), expectedrows=clustering_array.shape[0])
        #embed()
        clu = kwikfile.root._f_getChild(pathname)
        clu.append(clustering_array)
        
        
        


def SpikeSimilarityMeasure(a,b):
    ''' Computes spike similarity measure between masks a and b  CS(a,b) = a.b/|a||b|'''
    SSmeasure = np.dot(a,b)/(np.sqrt(np.dot(a,a))*np.sqrt(np.dot(b,b)))
    return SSmeasure  

@ju.func_cache
def test_detection_algorithm(hybdatadict, SDparams,prb, detectioncrit):
    '''
     It will query whether the cached function: 
    hybridata_creation_lib.create_hybrid_kwdfile(hybdatadict), 
    has been called already with those arguments using `joblib_utils.is_cached`,
     If it has, it calls it to obtain creation_groundtruth.
     else if the hybrid dataset does not exist, it will raise an Error. 
       creation_groundtruth consists of the equivalent to the old: 
     GroundtruthResfile,GroundtruthClufile,... (i.e. the times and the cluster labels for the
     added hybrid spikes.
     detection criteria include: 
      allowed_discrepancy,CSthreshold
       This function will call SpikeSimilarityMeasure(a,b)
         and output the file: Hash(hybdatadict)_Hash(sdparams)_Hash(detectioncrit).kwik  
       It will return detcrit_groundtruth, the groundtruth relative to the criteria, detectioncrit '''
    
    
    if ju.is_cached(hcl.create_hybrid_kwdfile,hybdatadict):
        print 'Yes, this hybrid dataset exists, I shall now check if you have run SD \n'
    
    meanwaveform,kwdoutputname, creation_groundtruth, amplitude = hcl.create_hybrid_kwdfile(hybdatadict)
    
    #Take the means of the binary donor masks of the donor cluster 
    binmeanmask = hcl.make_average_datamask_from_mean(hybdatadict, fmask= False)
    
    argSD = [hybdatadict,SDparams,prb]
    if ju.is_cached(rsd.run_spikedetekt,*argSD):
        print 'Yes, SD has been run \n'
        hash_hyb_SD = rsd.run_spikedetekt(hybdatadict,SDparams,prb)
    else:
        print 'You need to run Spikedetekt before attempting to analyse results ' 
    
    DIRPATH = hybdatadict['output_path']
    with Experiment(hash_hyb_SD, dir= DIRPATH, mode='a') as expt:
        res_int= expt.channel_groups[0].spikes.time_samples
        res_frac  = expt.channel_groups[0].spikes.time_fractional
        res_int_arr = res_int[:] 
        res_frac_arr = res_frac[:]
        detected_times = res_int_arr + res_frac_arr
        #Masks
        fmask = expt.channel_groups[0].spikes.features_masks
        
        #Spikes within time window
        existencewin = np.zeros_like(creation_groundtruth)
        
        #Mean binary mask for hybrid cluster
        binmeanmask = hcl.make_average_datamask_from_mean(hybdatadict, fmask= False)
        
        indices_in_timewindow = {}
        #indices_in_timewindow = {0 (this is the 1st hybrid spike): (array([0, 1, 3]),),
        #1: (array([89, 90]),),
        #2: (array([154, 156, 157]),),
        #3: (array([191]),),
        #4: (array([259, 260, 261]),),
        
        num_spikes_in_timewindow = {}
        
        CauchySchwarz = {}
        detected = {}
        NumHybSpikes = creation_groundtruth.shape[0]
        detectedgroundtruth = np.zeros_like(detected_times)
        print detectedgroundtruth.shape
        for k in np.arange(NumHybSpikes):
            list_of_differences = np.zeros((detected_times.shape[0]))
            list_of_differences[:]= detected_times[:] - creation_groundtruth[k] 
            
            indices_in_timewindow[k] = np.nonzero(np.absolute(list_of_differences)<=detectioncrit['allowed_discrepancy'])
            num_spikes_in_timewindow[k] = indices_in_timewindow[k][0].shape[0]
            for j in np.arange(num_spikes_in_timewindow[k]):
                CauchySchwarz[k,j] = SpikeSimilarityMeasure(fmask[indices_in_timewindow[k][0][j],:,1],binmeanmask[0:3*hybdatadict['numchannels']])
                if CauchySchwarz[k,j] > detectioncrit['CSthreshold']:
                    detected[k,j] = 1    
                else:
                    detected[k,j] = 0       
                detectedgroundtruth[indices_in_timewindow[k][0][j]]= detected[k,j]
    #Store detectedgroundtruth in a clustering
    kwikfilename = DIRPATH + hash_hyb_SD + '.kwik'
    detectionhashname = hash_utils.hash_dictionary_md5(detectioncrit)
    add_clustering_kwik(kwikfilename, detectedgroundtruth, detectionhashname)
    #add_clustering(
    #clusters = '/channel_groups[0]/spikes/clusters'
    #detectionhash = hash_dictionary_md5(detectioncrit)
    #expt.createEArray(clusters, detectionhash, tb.UInt32Atom(), (0,),
    #                      expectedrows=1000000)
        
        #percentage_detected = float(sum(detected.values()))/NumHybSpikes
        
    
    detcrit_groundtruth ={'detection_hashname':
    detectionhashname,'binmeanmask': binmeanmask, 'numspikes_in_timeswindow': num_spikes_in_timewindow,
    'Cauchy_Schwarz':CauchySchwarz,'detected': detected,'detected_groundtruth': detectedgroundtruth}
    return detcrit_groundtruth    
        
    

