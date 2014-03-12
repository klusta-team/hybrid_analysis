# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import hash_utils
import joblib_utils as ju
import numpy as np
import hybridata_creation_lib as hcl
import runspikedetekt_lib as rsd
import detection_statistics as ds
from spikedetekt2 import *

from sklearn import preprocessing
from sklearn import svm, cross_validation
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import KFold
from sklearn.linear_model import SGDClassifier


def do_cross_validation_shuffle(datasize,k_times):
    #    self.cross_validated = KFold(self.datasize,k_times,indices=True)
    #cross_validated = KFold(datasize,k_times, shuffle=True)
    cross_validated = KFold(datasize,k_times)
        #number_of_parts = k_times
    return cross_validated


def compute_grid_weights(minleftweight, minrightweight, base, maxpowerv, maxpowerw):
    ''' Input:
    gridweightparams = svmparams['gridweightparams'] 
    {'gridweightparams: (minleftweight, minrightweight, base, maxpowerv, maxpowerw)}
    computes grid
    {0:minleftweight*base^w, 1: minrightweight*base^v} and lists them'''
    class_weight = []
    for v in np.arange(maxpowerv):
        for w in np.arange(maxpowerw):
            class_weight.append({0:minleftweight*np.power(base,w), 1: minrightweight*np.power(base,v)}) 
    #print class_weight  
    number_of_weights =  maxpowerv*maxpowerw
    print 'number of weights = ' , number_of_weights
    #return class_weight, number_of_weights
    return class_weight

def scale_data(feature_data):
    ''' Scales feature_data between 0 and 1,
        these features are stored in
        Hash(hybdatadict)_Hash(sdparams)_Hash(detectioncrit).kwx
        Can use sklearn.preprocessing.scale
        ''' 
    print feature_data.shape
    scaled_data = np.zeros(feature_data.shape,dtype=np.float32)
    for i in xrange(feature_data.shape[1]):
        scaled_data[:,i] = preprocessing.scale(feature_data[:,i])
    return scaled_data

@ju.func_cache
def do_supervised_learning(test, train,Cval, supervised_params, scaled_fets, target,classweight):
    '''Do supervised learning'''
    clf = svm.SVC(C= Cval,kernel=supervised_params['kernel'],degree=2,coef0=1,cache_size=1000)
    clf.fit(scaled_fets[train],target[train],class_weight=classweight)
    preds = clf.predict(scaled_fets[test])
    preds_train= clf.predict(scaled_fets[train])
    return preds,preds_train

@ju.func_cache
def pre_learn_data_grid(hybdatadict, SDparams,prb,detectioncrit,supervised_params):
    '''First this function will query whether the cached function: 
       detection_statistics.test_detection_algorithm(hybdatadict, SDparams, detectioncrit):, 
       has been called already with those arguments using `joblib_utils.is_cached`,
       If it has, it calls it to obtain detcrit_groundtruth.
       else if the hybrid dataset does not exist, it will raise an Error
       and tell you to run SpikeDetekt on the dataset.
    
       It scales the data using scale_data() 
    '''
    argTD = [hybdatadict, SDparams,prb, detectioncrit]      
    if ju.is_cached(ds.test_detection_algorithm,*argTD):
        print 'Yes, you have run detection_statistics.test_detection_algorithm() \n'
        detcrit_groundtruth = ds.test_detection_algorithm(hybdatadict, SDparams,prb, detectioncrit)
    else:
        print 'You need to run detection_statistics.test_detection_algorithm() \n in order to obtain a groundtruth'    
    #'detection_hashname'
    
    
    argSD = [hybdatadict,SDparams,prb]
    if ju.is_cached(rsd.run_spikedetekt,*argSD):
        print 'Yes, SD has been run \n'
        hash_hyb_SD = rsd.run_spikedetekt(hybdatadict,SDparams,prb)
    else:
        print 'You need to run Spikedetekt before attempting to analyse results ' 
    
    DIRPATH = hybdatadict['output_path']
    with Experiment(hash_hyb_SD, dir= DIRPATH, mode='r') as expt:
        
        #Load the detcrit groundtruth
        #targetpathname = '/channel_groups/0/spikes/clusters' + '/' + detcrit_groundtruth['detection_hashname']
        targetpathname = detcrit_groundtruth['detection_hashname']
        targetsource = expt.channel_groups[0].spikes.clusters._get_child(targetpathname)
        
        #take the first supervised_params['numfirstspikes'] spikes only
        if supervised_params['numfirstspikes'] is not None: 
            fets = expt.channel_groups[0].spikes.features[0:supervised_params['numfirstspikes']]
            target = targetsource[0:supervised_params['numfirstspikes']]
        else: 
            fets = expt.channel_groups[0].spikes.features[:]
            target = targetsource[:]
        print expt        
        
            
            
    print 'fets.shape = ', fets.shape    
    print 'target.shape = ', target.shape
        
    if supervised_params['subvector'] is not None:
        subsetfets = fets[:,supervised_params['subvector']]
    else:
        subsetfets = fets
        
    scaled_fets = scale_data(subsetfets)
    classweights = compute_grid_weights(*supervised_params['grid_params'])
    #print classweights
    
    
    return classweights,scaled_fets, target
    
    
def learn_data_grid(hybdatadict, SDparams,prb,detectioncrit,supervised_params):  
    '''
     
       calls learn_data() for various values of the
       grids and also the function compute_errors()
    
       Writes output as clusterings labelled by Hash(svmparams) of the grid in 
       Hash(hybdatadict)_Hash(sdparams)_Hash(detectioncrit)_Hash(supervised_params).kwik
       using write_kwik(hybdatadict,sdparams,detectioncrit,svmparams,confusion_test,confusion_train)
       the new .kwik format can store multiple clusterings.
    
       supervised_params consists of the following quantities: 
       supervised_params = {'numfirstspikes': 200000,'kernel': 'rbf','grid_C': [1,100000,0.00001], 'grid_weights': listofweights
       ,gammagrid : [1e-5, 0.001, 0.1, 1, 10, 1000, 100000], cross_param :  2, 
       PCAS : 3, subvector: None}
    '''
    #----------------------------------------------------------
    
    argPLDG = [hybdatadict, SDparams,prb,detectioncrit,supervised_params]
    if ju.is_cached(pre_learn_data_grid,*argPLDG):
        print 'Yes, pre_learn_data_grid has been run \n'    
    else:
        print 'Running pre_learn_data_grid(hybdatadict, SDparams,prb,detectioncrit,supervised_params), \n you have not run it yet' 
        
    classweights,scaled_fets, target = pre_learn_data_grid(hybdatadict, SDparams,prb,detectioncrit,supervised_params)
    
    
    
    number_of_weights = len(classweights)
    
    numspikes = scaled_fets.shape[0]
    cross_valid = do_cross_validation_shuffle(numspikes,supervised_params['cross_param'])
    
    #print cross_valid
    
    #do_supervised(supervised_params,

    #'grid_C': [1,100000,0.00001], number_cvalues = 3
    number_cvalues = len(supervised_params['grid_C'])
    
    #number_support_vectors = {}
    weights_clu_test = np.zeros((number_cvalues,number_of_weights,numspikes,2),dtype=np.int32)
    weights_clu_train = np.zeros((number_cvalues,number_of_weights, numspikes,2),dtype=np.int32)
    cludict= {(0,0):1, (0,1):2, (1,0):3, (1,1):4}
    # (prediction, groundtruth)
    #(0,0) TN, (0,1) FN ,(1,0) FP ,(1,1) TP
    testclu = np.zeros((number_cvalues,number_of_weights,numspikes),dtype=np.int32)
    trainclu = np.zeros((number_cvalues,number_of_weights,numspikes),dtype=np.int32)
    
    
    
    for c, Cval in enumerate(supervised_params['grid_C']):
        preds = {}
        preds_train = {}
                    ##Defined to avoid: TypeError: unhashable type: 'numpy.ndarray', something about dictionaries
                    #testclu_pre = np.zeros((number_of_weights,numspikes),dtype=np.int32)
                    #trainclu_pre = np.zeros((number_of_weights,numspikes),dtype=np.int32)
        for i, (weights) in enumerate(classweights):
            for j, (train, test) in enumerate(cross_valid):
                preds[i,j], preds_train[i,j]= do_supervised_learning(test, train,Cval, supervised_params, scaled_fets, target,classweights[i])
                
                #Used later to make equivalent to 4 seasons clu file
                weights_clu_test[c,i,test,0] = preds[i,j]
                weights_clu_test[c,i,test,1] = target[test]
                
                
                #Used later to make equivalent to 4 seasons clu file but for the training set
                weights_clu_train[c,i,train,0] = preds_train[i,j]
                weights_clu_train[c,i,train,1] = target[train]
                
    
            #Make 4 seasons clu file equivalent
            for k in np.arange(numspikes):
                testclu[c,i,k] = cludict[tuple(weights_clu_test[c,i,k,:])]
                trainclu[c,i,k] = cludict[tuple(weights_clu_train[c,i,k,:])]
                #print 'testclu[',c,',',i,',',k,']=',testclu[c,i,k]
    
   # for c, Cval in enumerate(supervised_params['grid_C']):                    
   #     kwikfilename = DIRPATH + hash_hyb_SD + '.kwik'
   #     supervisedhashname = hash_utils.hash_dictionary_md5(detectioncrit)
   #     add_clustering_kwik(kwikfilename, detectedgroundtruth, detectionhashname)        
   
      
                
   ####Train and test look like this for 2-fold cross validation and 200 spikes
  
   #         j =  0  train =  [100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117
   #      118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135
   #      136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153
   #      154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171
   #      172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189
   #      190 191 192 193 194 195 196 197 198 199]  
   #      test =  [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
   #      25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
   #      50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74
   #      75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99]
   #     j =  1  train =  [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
   #      25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
   #      50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74
   #      75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99]  
   #      test =  [100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117
   #      118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135
   #      136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153
   #      154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171
   #      172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189
   #      190 191 192 193 194 195 196 197 198 199] 
    
        
    return classweights, testclu, trainclu

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
    rootie = np.sqrt(10)
    supervised_params = {'numfirstspikes': 200,'kernel': 'poly','grid_C': [1,100000,0.00001], 
       'gammagrid' : [1e-5, 0.001, 0.1, 1, 10, 1000, 100000], 'cross_param' :  2, 
       'PCAS' : 3, 'subvector': None, 'grid_params' : (1e-3,1e-3, rootie, 14,14)}
    
    classweights3, testclu3, trainclu3 = learn_data_grid(hybdatadict, sdparams,prb,detectioncrit,supervised_params)

