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
import copy 

#from spikedetekt2.dataio import Experiment
from spikedetekt2 import *
from xml.etree.ElementTree import ElementTree,Element,SubElement

from IPython import embed

#write_mask(M, basename+'.fmask.'+str(shank), fmt='%f')

#def write_spk_buffered(table, column, filepath, indices,
#                       channels=slice(None), buffersize=512):
#    with open(filepath, 'wb') as f:
#        numitems = len(indices)
#        for i in xrange(0, numitems, buffersize):
#            waves = table[indices[i:i+buffersize]][column]
#            waves = waves[:, :, channels]
#            waves = np.int16(waves)
#            waves.tofile(f)

def write_trivial_clu(restimes,filepath):
    """writes cluster cluster assignments to text file readable by klusters and neuroscope.
    input: clus is a 1D or 2D numpy array of integers
    output:
        top line: number of clusters (max cluster)
        next lines: one integer per line"""
    clus = np.zeros_like(restimes) 
    clu_file = open( filepath,'w')
    #header line: number of clusters
    if len(clus) == 0:
        n_clu = 1
    else:
        n_clu = clus.max() + 1
    clu_file.write( '%i\n'%n_clu)
    #one cluster per line
    np.savetxt(clu_file,np.int16(clus),fmt="%i")
    clu_file.close() 

def write_clu(clus, filepath):
    """writes cluster cluster assignments to text file readable by klusters and neuroscope.
input: clus is a 1D or 2D numpy array of integers
output:
top line: number of clusters (max cluster)
next lines: one integer per line"""
    clu_file = open( filepath,'w')
    #header line: number of clusters
    n_clu = clus.max()+1
    clu_file.write( '%i\n'%n_clu)
    #one cluster per line
    np.savetxt(clu_file,np.int16(clus),fmt="%i")
    clu_file.close()

def write_spk_buffered(exptable,filepath, indices,
                       buffersize=512): 
    with open(filepath, 'wb') as f:
        numitems = len(indices)
        for i in xrange(0, numitems, buffersize):
            waves = exptable[indices[i:i+buffersize],:,:]  
            #waves = waves[:, :, channels]
            waves = np.int16(waves)
            waves.tofile(f)

def write_xml(probe,n_ch,n_samp,n_feat,sample_rate,filepath):
    """makes an xml parameters file so we can look at the data in klusters"""
    parameters = Element('parameters')
    acquisitionSystem = SubElement(parameters,'acquisitionSystem')
    SubElement(acquisitionSystem,'nBits').text = '16'
    SubElement(acquisitionSystem,'nChannels').text = str(n_ch)
    SubElement(acquisitionSystem,'samplingRate').text = str(int(sample_rate))
    #SubElement(acquisitionSystem,'voltageRange').text = str(Parameters['VOLTAGE_RANGE'])
    #SubElement(acquisitionSystem,'amplification').text = str(Parameters['AMPLIFICATION'])
    #SubElement(acquisitionSystem,'offset').text = str(Parameters['OFFSET'])
    
    anatomicalDescription = SubElement(SubElement(parameters,'anatomicalDescription'),'channelGroups')
    for shank in probe.shanks_set:
        shankgroup = SubElement(anatomicalDescription,'group')
        for i_ch in probe.channel_set[shank]:
            SubElement(shankgroup,'channel').text=str(i_ch)
# channels = SubElement(SubElement(SubElement(parameters,'channelGroups'),'group'),'channels')
# for i_ch in range(n_ch):
# SubElement(channels,'channel').text=str(i_ch)
    
    spikeDetection = SubElement(SubElement(parameters,'spikeDetection'),'channelGroups')
    for shank in probe.shanks_set:
        shankgroup = SubElement(spikeDetection,'group')
        channels = SubElement(shankgroup,'channels')
        for i_ch in probe.channel_set[shank]:
            SubElement(channels,'channel').text=str(i_ch)
# channels = SubElement(group,'channels')
# for i_ch in range(n_ch):
# SubElement(channels,'channel').text=str(i_ch)
        SubElement(shankgroup,'nSamples').text = str(n_samp)
        SubElement(shankgroup,'peakSampleIndex').text = str(n_samp//2)
        SubElement(shankgroup,'nFeatures').text = str(n_feat)
    
    indent_xml(parameters)
    ElementTree(parameters).write(filepath)
    

def indent_xml(elem, level=0):
    """input: elem = root element
changes text of nodes so resulting xml file is nicely formatted.
copied from http://effbot.org/zone/element-lib.htm#prettyprint"""
    i = "\n" + level*" "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + " "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent_xml(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def write_res(samples,filepath):
    """input: 1D vector of times shape = (n_times,) or (n_times, 1)
    output: writes .res file, which has integer sample numbers"""
    np.savetxt(filepath,samples,fmt="%i")
            
            
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
      
#@ju.func_cache - because expt can't be pickled by joblib
#PicklingError("Can't pickle <function remove at 0x4c5e488>: it's not found as weakref.remove",)
def make_spkresdetclu_files(expt,res,mainresfile, mainspkfile, detcritclufilename, trivialclufilename):
    write_res(res,mainresfile)
    write_trivial_clu(res,trivialclufilename)
    write_spk_buffered(expt.channel_groups[0].spikes.waveforms_filtered,
                            mainspkfile,
                           np.arange(len(res)))
    write_clu(detcrit_groundtruth['detected_groundtruth'], detcritclufilename)


def make_KKscript_supercomp(KKparams, filebase,scriptname,supercomparams):
    '''Create bash script on Legion required to run KlustaKwik
    supercomparams = {'time':'36:00:00','mem': '2G', 'tmpfs':'10G'}
    
    '''
    argKKsc = [KKparams, filebase,scriptname]
    if ju.is_cached(make_KKscript,*argKKsc):
        print 'Yes, you have made the scripts for the local machine \n'
        #scriptstring = make_KKscript(KKparams, filebase,scriptname)
    else:
        print 'You need to run make_KKscript  ' 
    
    keylist = KKparams['keylist']
    
    #keylist = ['MaskStarts','MaxPossibleClusters','FullStepEvery','MaxIter','RandomSeed',
    #           'Debug','SplitFirst','SplitEvery','PenaltyK','PenaltyKLogN','Subset',
    #           'PriorPoint','SaveSorted','SaveCovarianceMeans','UseMaskedInitialConditions',
     #          'AssignToFirstClosestMask','UseDistributional']

    #KKlocation = '/martinottihome/skadir/GIT_masters/klustakwik/MaskedKlustaKwik'  
    
    supercompstuff = '''#!/bin/bash -l
#$ -S /bin/bash
#$ -l h_rt=%s
#$ -l mem=%s
#$ -l tmpfs=%s
#$ -N %s_supercomp
#$ -P maskedklustakwik
#$ -wd /home/smgxsk1/Scratch/
cd $TMPDIR
'''%(supercomparams['time'],supercomparams['mem'],supercomparams['tmpfs'],scriptname)
    
    KKsupercomplocation = supercompstuff +  '/home/smgxsk1/MKK_versions/klustakwik/MaskedKlustaKwik'
    scriptstring = KKsupercomplocation + ' /home/smgxsk1/Scratch/'+ filebase + ' 1 '
    for KKey in keylist: 
        #print '-'+KKey +' '+ str(KKparams[KKey])
        scriptstring = scriptstring + ' -'+ KKey +' '+ str(KKparams[KKey])
    
    print scriptstring
    scriptfile = open('%s_supercomp.sh' %(scriptname),'w')
    scriptfile.write(scriptstring)
    scriptfile.close()
    outputdir = ' /chandelierhome/skadir/hybrid_analysis/mariano/'
    #changeperms='chmod 777 %s.sh' %(scriptname)
    sendout = 'scp -r'+ outputdir + scriptname + '_supercomp.sh' + outputdir +scriptname + '.fet.1' + outputdir + scriptname + '.fmask.1 '+ 'smgxsk1@legion.rc.ucl.ac.uk:/home/smgxsk1/Scratch/'
    os.system(sendout)
    
    return scriptstring
    
@ju.func_cache
def make_KKscript(KKparams, filebase,scriptname):
    
    keylist = KKparams['keylist']
    #keylist = ['MaskStarts','MaxPossibleClusters','FullStepEvery','MaxIter','RandomSeed',
    #           'Debug','SplitFirst','SplitEvery','PenaltyK','PenaltyKLogN','Subset',
    #           'PriorPoint','SaveSorted','SaveCovarianceMeans','UseMaskedInitialConditions',
     #          'AssignToFirstClosestMask','UseDistributional']

    #KKlocation = '/martinottihome/skadir/GIT_masters/klustakwik/MaskedKlustaKwik'  
    KKlocation = KKparams['KKlocation']
    scriptstring = KKlocation + ' '+ filebase + ' 1 '
    for KKey in keylist: 
        #print '-'+KKey +' '+ str(KKparams[KKey])
        scriptstring = scriptstring + ' -'+ KKey +' '+ str(KKparams[KKey])
    
    print scriptstring
    scriptfile = open('%s.sh' %(scriptname),'w')
    scriptfile.write(scriptstring)
    scriptfile.close()
    changeperms='chmod 777 %s.sh' %(scriptname)
    os.system(changeperms)
    
    return scriptstring

def make_KKfiles_Script_supercomp(hybdatadict, SDparams,prb, detectioncrit, KKparams,supercomparams):
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
    
    KKscriptname = basefilename
    make_KKscript_supercomp(KKparams,basefilename,KKscriptname,supercomparams)
    
    return basefilename


@ju.func_cache
def make_KKfiles_Script_full(hybdatadict, SDparams,prb, detectioncrit, KKparams):
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
            feats = expt.channel_groups[0].spikes.features[0:KKparams['numspikesKK']]
            prefmasks = expt.channel_groups[0].spikes.features_masks[0:KKparams['numspikesKK'],:,1]
            
            premasks = expt.channel_groups[0].spikes.masks[0:KKparams['numspikesKK']]
            res = expt.channel_groups[0].spikes.time_samples[0:KKparams['numspikesKK']]
        else: 
            feats = expt.channel_groups[0].spikes.features[:]
            prefmasks = expt.channel_groups[0].spikes.features_masks[:,:,1]
            #print fmasks[3,:]
            premasks = expt.channel_groups[0].spikes.masks[:]
            res = expt.channel_groups[0].spikes.time_samples[:]    
            
        mainresfile = DIRPATH + mainbasefilename + '.res.1' 
        mainspkfile = DIRPATH + mainbasefilename + '.spk.1'
        detcritclufilename = DIRPATH + mainbasefilename + '.detcrit.clu.1'
        trivialclufilename = DIRPATH + mainbasefilename + '.clu.1'
        
        #arg_spkresdetclu = [expt,res,mainresfile, mainspkfile, detcritclufilename, trivialclufilename]
        #if ju.is_cached(make_spkresdetclu_files,*arg_spkresdetclu):
        if os.path.isfile(mainspkfile):
            print 'miscellaneous files probably already exist, moving on, saving time'
        else:
            make_spkresdetclu_files(expt,res,mainresfile, mainspkfile, detcritclufilename, trivialclufilename) 
        
        #write_res(res,mainresfile)
        #write_trivial_clu(res,trivialclufilename)
        #write_spk_buffered(expt.channel_groups[0].spikes.waveforms_filtered,
        #                    mainspkfile,
        #                   np.arange(len(res)))
        #write_clu(detcrit_groundtruth['detected_groundtruth'], detcritclufilename)
        
        times = np.expand_dims(res, axis =1)
        masktimezeros = np.zeros_like(times)
        fets = np.concatenate((feats, times),axis = 1)
        fmasks = np.concatenate((prefmasks, masktimezeros),axis = 1)
        masks = np.concatenate((premasks, masktimezeros),axis = 1)
    
    mainfetfile = DIRPATH + mainbasefilename+'.fet.1'
    mainfmaskfile = DIRPATH + mainbasefilename+'.fmask.1'
    mainmaskfile = DIRPATH + mainbasefilename+'.mask.1'
    
    #print fets
    #embed()
    
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
        
    
    mainxmlfile =  hybdatadict['donor_path'] + hybdatadict['donor']+'_afterprocessing.xml'   
    os.system('ln -s %s %s.fet.1 ' %(mainfetfile,basefilename))
    os.system('ln -s %s %s.fmask.1 ' %(mainfmaskfile,basefilename))
    os.system('ln -s %s %s.mask.1 ' %(mainmaskfile,basefilename))
    os.system('ln -s %s %s.trivial.clu.1 ' %(trivialclufilename,basefilename))
    os.system('ln -s %s %s.spk.1 ' %(mainspkfile,basefilename))
    os.system('ln -s %s %s.res.1 ' %(mainresfile,basefilename))
    os.system('cp %s %s.xml ' %(mainxmlfile,mainbasefilename))
    os.system('cp %s %s.xml ' %(mainxmlfile,basefilename))
    
    KKscriptname = basefilename
    make_KKscript(KKparams,basefilename,KKscriptname)
    
    return basefilename

@ju.func_cache
def one_param_varyKK(hybdatadict, SDparams,prb, detectioncrit, defaultKKparams, paramtochange, listparamvalues):
    outputdicts = []
    for paramvalue in listparamvalues:
        newKKparamsdict = copy.deepcopy(defaultKKparams)
        newKKparamsdict[paramtochange] = paramvalue
        make_KKfiles_Script_full(hybdatadict, SDparams,prb, detectioncrit, newKKparamsdict)
        outputdicts.append(newKKparamsdict)
    return outputdicts   
    
def one_param_varyKK_super(hybdatadict, SDparams,prb, detectioncrit, defaultKKparams, paramtochange, listparamvalues,supercomparams):
    outputdicts = []
    for paramvalue in listparamvalues:
        newKKparamsdict = copy.deepcopy(defaultKKparams)
        newKKparamsdict[paramtochange] = paramvalue
        make_KKfiles_Script_supercomp(hybdatadict, SDparams,prb, detectioncrit, newKKparamsdict,supercomparams)
        outputdicts.append(newKKparamsdict)
    return outputdicts  
    


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



def make_KKfiles_viewer(hybdatadict, SDparams,prb, detectioncrit, KKparams):
    
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
        
    argKKfile = [hybdatadict, SDparams,prb, detectioncrit, KKparams]
    if ju.is_cached(make_KKfiles_Script,*argKKfile):
        print 'Yes, make_KKfiles_Script  has been run \n'
        
    else:
        print 'Need to run make_KKfiles_Script first, running now ' 
    basefilename = make_KKfiles_Script(hybdatadict, SDparams,prb, detectioncrit, KKparams)    
        
    mainbasefilelist = [hash_hyb_SD, detcrit_groundtruth['detection_hashname']]
    mainbasefilename = hash_utils.make_concatenated_filename(mainbasefilelist)    
    
    DIRPATH = hybdatadict['output_path']
    os.chdir(DIRPATH)
    with Experiment(hash_hyb_SD, dir= DIRPATH, mode='r') as expt:
        if KKparams['numspikesKK'] is not None: 
            #spk = expt.channel_groups[0].spikes.waveforms_filtered[0:KKparams['numspikesKK'],:,:]
            res = expt.channel_groups[0].spikes.time_samples[0:KKparams['numspikesKK']]
            #fets = expt.channel_groups[0].spikes.features[0:KKparams['numspikesKK']]
            #fmasks = expt.channel_groups[0].spikes.features_masks[0:KKparams['numspikesKK'],:,1]
            
           # masks = expt.channel_groups[0].spikes.masks[0:KKparams['numspikesKK']]

        else: 
            #spk = expt.channel_groups[0].spikes.waveforms_filtered[:,:,:]
            res = expt.channel_groups[0].spikes.time_samples[:]
            #fets = expt.channel_groups[0].spikes.features[:]
            #fmasks = expt.channel_groups[0].spikes.features_masks[:,:,1]
            #print fmasks[3,:]
            #masks = expt.channel_groups[0].spikes.masks[:]
            
        mainresfile = DIRPATH + mainbasefilename + '.res.1' 
        mainspkfile = DIRPATH + mainbasefilename + '.spk.1'
        detcritclufilename = DIRPATH + mainbasefilename + '.detcrit.clu.1'
        trivialclufilename = DIRPATH + mainbasefilename + '.clu.1'
        write_res(res,mainresfile)
        write_trivial_clu(res,trivialclufilename)
        
       # write_spk_buffered(exptable,filepath, indices,
       #                buffersize=512)
        write_spk_buffered(expt.channel_groups[0].spikes.waveforms_filtered,
                            mainspkfile,
                           np.arange(len(res)))
        
        write_clu(detcrit_groundtruth['detected_groundtruth'], detcritclufilename)
            
        #s_total = SDparams['extract_s_before']+SDparams['extract_s_after']
            
        #write_xml(prb,
        #          n_ch = SDparams['nchannels'],
        #          n_samp = SDparams['S_TOTAL'],
        #          n_feat = s_total,
        #          sample_rate = SDparams['sample_rate'],
        #          filepath = basename+'.xml')
    mainxmlfile =  hybdatadict['donor_path'] + hybdatadict['donor']+'_afterprocessing.xml'   
    
    os.system('ln -s %s %s.clu.1 ' %(trivialclufilename,basefilename))
    os.system('ln -s %s %s.spk.1 ' %(mainspkfile,basefilename))
    os.system('ln -s %s %s.res.1 ' %(mainresfile,basefilename))
    os.system('cp %s %s.xml ' %(mainxmlfile,basefilename))
    
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
    basefile = make_KKfiles_Script(hybdatadict, sdparams,prb, detectioncrit, KKparams)


    
   
    
    
        

