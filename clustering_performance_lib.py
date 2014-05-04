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
import runKK_lib as rkk
#from spikedetekt2.dataio import Experiment
from spikedetekt2 import *
from sklearn.metrics import confusion_matrix

@ju.func_cache
def get_confusion_matrix(a,b):
    conf = confusion_matrix(a,b)
    return conf
    

def EntropyH(Prob):
    ''' Required for computing Meila's VI metric 
        Computes entropy of a discrete random variable taking K values
        entropyH = -dot(prob,logP)'''
    print Prob.shape
    K = Prob.shape[0]
    logP = np.zeros((K,1))
    print logP.shape
    #logP= log(Prob)

    for k in np.arange(K):
        if Prob[k] == 0:
            logP[k] = 0 #avoid the minus infinity when Prob(k) = 0
        else:
            logP[k] = np.log(Prob[k])

    entropyH = -np.dot(Prob.T,logP )
    return entropyH

def MutualInf(Prob, Probprime, ProbJoint):
    '''Mutual information between the associated random variables, prob,
       probprime, and their joint distribution, probjoint'''
    
    K = Prob.shape[0]
    Kprime = Probprime.shape[0]
    Infie = np.zeros((K,Kprime))
    logie = np.zeros((K,Kprime))

    #Inf=0
    for i in np.arange(K):
        for j in np.arange(Kprime):
            if ProbJoint[i,j] == 0:
                Infie[i,j] = 0
            else:
    #logie(i,j)= log(ProbJoint(i,j)/(Prob(i)*Probprime(j)));       
    #Inf = Inf + ProbJoint(i,j).*( log(ProbJoint(i,j)/(Prob(i)*Probprime(j))) );
                Infie[i,j] = ProbJoint[i,j]*( np.log( np.true_divide(ProbJoint[i,j],(Prob[i]*Probprime[j]))) )

    #Infie(i,j) = ProbJoint(i,j).*( logie(i,j) );
    
    #Inf.logie=logie;
    infisum = np.sum(Infie)
    
    mutual_inf = (infisum, Infie)
    return mutual_inf    

def VImetric(ConfusionMatrix):
    '''Computes Meila's VI metric between two
        clusterings of the same data, e.g. KK clustering 
        and the detcrit_groundtruth from the confusion matrix'''
    
 
    #nbklust = ConfusionMatrix.shape[0]
    #nbklustprime = ConfusionMatrix.shape[1]
    
    totalspikes = np.sum(ConfusionMatrix)
    
    #Pk = np.zeros((nbklust,1))
    #Pkprime = np.zeros((nbklustprime, 1))
    
    Pk = np.true_divide(np.sum(ConfusionMatrix, axis = 1),totalspikes)
    #for k in np.arange(nbklust):
    #    Pk[k] = np.sum(ConfusionMatrix[k,:])/totalspikes
    
    Pkprime = np.true_divide(np.sum(ConfusionMatrix, axis = 0),totalspikes)
    #for kk in np.arange(nbklustprime):
    #    Pkprime[kk] = np.sum(ConfusionMatrix[:,kk])/totalspikes
    
    PJoint = np.true_divide(ConfusionMatrix,totalspikes)
    
    HC = EntropyH(Pk)
    HCprime = EntropyH(Pkprime)


    Inff = MutualInf(Pk,Pkprime,PJoint)
    #mutual_inf = (infisum, Infie)
    VI = HC+HCprime - 2*Inff[0];

    VImetrics = {'VI':VI, 'Mutual Inf': Inff, 'PJoint' : PJoint, 'PK': Pk, 'PKprime': Pkprime, 'HC': HC, 'HCprime': HCprime}

    return VImetrics    
    
def create_confusion_matrix_fromclu(hybdatadict, SDparams, prb, detectioncrit, KKparams):
    ''' will create the confusion matrix, using the equivalent to a clu file
      and detcrit groundtruth res and clu files, which is now contained in the kwik file 
       which will either be from KK or SVM and of the form: 
       Hash(hybdatadict)_Hash(sdparams)_Hash(detectioncrit)_KK_Hash(kkparams).kwik
        Hash(hybdatadict)_Hash(sdparams)_Hash(detectioncrit)_SVM_Hash(svmparams).kwik'''
    argSD = [hybdatadict,SDparams,prb]
    if ju.is_cached(rsd.run_spikedetekt,*argSD):
        print 'Yes, SD has been run \n'
        hash_hyb_SD = rsd.run_spikedetekt(hybdatadict,SDparams,prb)
    else:
        print 'You need to run Spikedetekt before attempting to analyse results ' 
    
    
    argTD = [hybdatadict, SDparams,prb, detectioncrit]      
    if ju.is_cached(ds.test_detection_algorithm,*argTD):
        print 'Yes, you have run detection_statistics.test_detection_algorithm() \n'
        detcrit = ds.test_detection_algorithm(hybdatadict, SDparams,prb, detectioncrit)
    else:
        print 'You need to run detection_statistics.test_detection_algorithm() \n in order to obtain a groundtruth' 
    
    
    
    #argKK = [hybdatadict, SDparams, prb, detectioncrit, KKparams]
    #print 'What the bloody hell is going on?'
    #if ju.is_cached(rkk.make_KKfiles_Script,*argKK):
    #    print 'Yes, you have created the scripts for running KK, which you have hopefully run!'
    #    basefilename = rkk.make_KKfiles_Script(hybdatadict, SDparams, prb, detectioncrit, KKparams)
    #else:
    #    print 'You need to run KK to generate a clu file '
        
    #print 'Did you even get here?'    
    
    basefilename = rkk.make_KKfiles_Script_full(hybdatadict, SDparams, prb, detectioncrit, KKparams)
    
    DIRPATH = hybdatadict['output_path']
    KKclufile = DIRPATH+ basefilename + '.clu.1'    
    KKclusters = np.loadtxt(KKclufile,dtype=np.int32,skiprows=1)    
    
    conf = get_confusion_matrix(KKclusters, detcrit['detected_groundtruth'])
    
    return detcrit, KKclusters,conf
    #return conf
    #return confusion_matrix

def create_confusion_matrix_fromclu_ind(hybdatadict, SDparams, prb, detectioncrit, KKparams):
    ''' will create the confusion matrix, using the equivalent to a clu file
      and detcrit groundtruth res and clu files, which is now contained in the kwik file 
       which will either be from KK or SVM and of the form: 
       Hash(hybdatadict)_Hash(sdparams)_Hash(detectioncrit)_KK_Hash(kkparams).kwik
        Hash(hybdatadict)_Hash(sdparams)_Hash(detectioncrit)_SVM_Hash(svmparams).kwik'''
    argSD = [hybdatadict,SDparams,prb]
    if ju.is_cached(rsd.run_spikedetekt,*argSD):
        print 'Yes, SD has been run \n'
        hash_hyb_SD = rsd.run_spikedetekt(hybdatadict,SDparams,prb)
    else:
        print 'You need to run Spikedetekt before attempting to analyse results ' 
    
    
    argTD = [hybdatadict, SDparams,prb, detectioncrit]      
    if ju.is_cached(ds.test_detection_algorithm,*argTD):
        print 'Yes, you have run detection_statistics.test_detection_algorithm() \n'
        detcrit = ds.test_detection_algorithm(hybdatadict, SDparams,prb, detectioncrit)
    else:
        print 'You need to run detection_statistics.test_detection_algorithm() \n in order to obtain a groundtruth' 
    
    
    
    #argKK = [hybdatadict, SDparams, prb, detectioncrit, KKparams]
    #print 'What the bloody hell is going on?'
    #if ju.is_cached(rkk.make_KKfiles_Script,*argKK):
    #    print 'Yes, you have created the scripts for running KK, which you have hopefully run!'
    #    basefilename = rkk.make_KKfiles_Script(hybdatadict, SDparams, prb, detectioncrit, KKparams)
    #else:
    #    print 'You need to run KK to generate a clu file '
        
    #print 'Did you even get here?'    
    
    basefilename = rkk.make_KKfiles_Script_detindep_full(hybdatadict, SDparams, prb, KKparams)
    
    DIRPATH = hybdatadict['output_path']
    KKclufile = DIRPATH+ basefilename + '.clu.1'    
    KKclusters = np.loadtxt(KKclufile,dtype=np.int32,skiprows=1)    
    
    conf = get_confusion_matrix(KKclusters, detcrit['detected_groundtruth'])
    
    return detcrit, KKclusters,conf
    #return conf
    #return confusion_matrix 


    
def create_confusion_matrix_KKhashnameclu(KKhashnameclu):
    '''Get the confusion matrix directly from the .clu file
    by exploiting the fact that the corresponding detcrit.clu.1 
    file has the same name minus one hashname of length 32'''
    KKclusters = np.loadtxt(KKhashnameclu,dtype=np.int32,skiprows=1)
    detcritclufile = KKhashnameclu[:-39]+'.detcrit.clu.1'
    detcrit = np.loadtxt(detcritclufile, dtype = np.int32, skiprows =1)
    conf = confusion_matrix(KKclusters,detcrit)
    return detcrit, KKclusters, conf   

def analysis_confKK(hybdatadict, SDparams,prb, detectioncrit, defaultKKparams, paramtochange, listparamvalues, detcrit = None):
    ''' Analyse results of one parameter family of KK jobs'''
    outlistKK = rkk.one_param_varyKK(hybdatadict, SDparams,prb, detectioncrit, defaultKKparams, paramtochange, listparamvalues) 
    #outlistKK = [listbasefiles, outputdicts]
    
    argTD = [hybdatadict, SDparams,prb, detectioncrit]      
    if ju.is_cached(ds.test_detection_algorithm,*argTD):
        print 'Yes, you have run detection_statistics.test_detection_algorithm() \n'
        detcrit_groundtruth = ds.test_detection_algorithm(hybdatadict, SDparams,prb, detectioncrit)
    else:
        print 'You need to run detection_statistics.test_detection_algorithm() \n in order to obtain a groundtruth' 
        
    detcritclu =  detcrit_groundtruth['detected_groundtruth']
    
    NumSpikes = detcritclu.shape[0]
    
    cluKK = np.zeros((len(outlistKK[0]),NumSpikes))
    confusion = []
    for k, basefilename in enumerate(outlistKK[0]):
        clufile = hybdatadict['output_path'] + basefilename + '.clu.1'
        print os.path.isfile(clufile)
        if os.path.isfile(clufile):
            cluKK[k,:] =   np.loadtxt(clufile, dtype = np.int32, skiprows =1)
        else:
            print '%s does not exist '%(clufile)
        
        conf = get_confusion_matrix(cluKK[k,:],detcritclu)
        print conf
        confusion.append(conf)
        
    return confusion    

def analysis_ind_confKK(hybdatadict, SDparams,prb, detectioncrit, defaultKKparams, paramtochange, listparamvalues, detcrit = None):
    ''' Analyse results of one parameter family of KK jobs
        Not very different to the fucntion above only detcrit independent MKKbasefilenames'''
    outlistKK = rkk.one_param_varyKK_ind(hybdatadict, SDparams,prb, defaultKKparams, paramtochange, listparamvalues) 
    #outlistKK = [listbasefiles, outputdicts]
    
    argTD = [hybdatadict, SDparams,prb, detectioncrit]      
    if ju.is_cached(ds.test_detection_algorithm,*argTD):
        print 'Yes, you have run detection_statistics.test_detection_algorithm() \n'
        detcrit_groundtruth = ds.test_detection_algorithm(hybdatadict, SDparams,prb, detectioncrit)
    else:
        print 'You need to run detection_statistics.test_detection_algorithm() \n in order to obtain a groundtruth' 
        
    detcritclu =  detcrit_groundtruth['detected_groundtruth']
    
    NumSpikes = detcritclu.shape[0]
    
    cluKK = np.zeros((len(outlistKK[0]),NumSpikes))
    confusion = []
    for k, basefilename in enumerate(outlistKK[0]):
        clufile = hybdatadict['output_path'] + basefilename + '.clu.1'
        print os.path.isfile(clufile)
        if os.path.isfile(clufile):
            cluKK[k,:] =   np.loadtxt(clufile, dtype = np.int32, skiprows =1)
        else:
            print '%s does not exist '%(clufile)
        
        conf = get_confusion_matrix(cluKK[k,:],detcritclu)
        print conf
        confusion.append(conf)
        
    return confusion      
    
    
    
     
    
    

