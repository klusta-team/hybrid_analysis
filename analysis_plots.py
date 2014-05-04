import subprocess 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyhull.convex_hull import ConvexHull
import itertools
from numpy import linalg as LA
from IPython import embed
import os
import runsupervised_lib as supe
import runKK_lib as rkk

#mpl.rcParams['legend.fontsize'] = 10

def plot_spike(graphpath,avehamspike,const,ifshow):
    fig1 = plt.figure()
    axes1 = fig1.add_axes([0.1,0.1,0.8,0.8])
    axes1.set_xlabel('Samples')
    #axes1.set_ylabel('')

    numchans = avehamspike.shape[1]
    axes1.hold(True)
    for chan in np.arange(numchans):
        #axis = plt.subplot(hybdatadict['numchannels'],1,i+1)
        axes1.plot(avehamspike[:,chan]+const*chan)
    
    if ifshow == True:
        plt.show()
    fig1.savefig('%s.pdf'%(graphpath))
    return

def plot_amp_detcrit(graphpath,detcrit_groundtruth,amplitude, titleplot,ifshow):
    fig1 = plt.figure()
    axes1 = fig1.add_axes([0.1,0.1,0.8,0.8])
    x = np.zeros(len(detcrit_groundtruth['Cauchy_Schwarz']))
    y = np.zeros(len(detcrit_groundtruth['Cauchy_Schwarz']))

    for k, keycs in enumerate(detcrit_groundtruth['Cauchy_Schwarz'].keys()):
       
        x[k] = amplitude[detcrit_groundtruth['Cauchy_Schwarz'].keys()[k][0]]
        y[k] = detcrit_groundtruth['Cauchy_Schwarz'][keycs]
    axes1.scatter(x,y,marker = 'o')
    #plt.title('Cluster '+repr(hybdatadict['donorcluster'])+ ' ' + hybdatadict['acceptor']+' '+scriptname)
    axes1.set_title(titleplot)
    axes1.set_xlabel('amplitude of hybrid spike')
    axes1.set_ylabel('CS measure of spike similarity')
    if ifshow == True:
        plt.show()
    fig1.savefig('%s.pdf'%(graphpath))    

def Error_rates_KK(KKconflist):
    TP_index = np.zeros(len(KKconflist)).astype(np.int32)
    TP = np.zeros(len(KKconflist)).astype(np.int32)
    FP = np.zeros(len(KKconflist)).astype(np.int32)
    FN = np.zeros(len(KKconflist)).astype(np.int32)
    TN = np.zeros(len(KKconflist)).astype(np.int32)
    false_discovery_rate = np.zeros(len(KKconflist))
    true_positive_rate = np.zeros(len(KKconflist))
    
    for k, confusionmat in enumerate(KKconflist):
    #KKconflist is is a list of confusion matrices
         
    
    #e.g. array([[   66,  1495,     0,     0,     0,     0,     0,     0,     0,
    #        0,     0,     0,     0,     0,     0,     0,     0,     0,
    #        0,     0,     0,     0,     0,     0,     0,     0,     0,
     #       0],
    #     [ 5268,    32,     0,     0,     0,     0,     0,     0,     0,
    #        0,     0,     0,     0,     0,     0,     0,     0,     0,
    #        0,     0,     0,     0,     0,     0,     0,     0,     0,
     #       0]])
    #[ 4404,     0,     0,     0,     0,     0,     0,     0,     0,
    #        0,     0,     0,     0,     0,     0,     0,     0,     0,
    #        0,     0,     0,     0,     0,     0,     0,     0,     0,
    #        0],
        #embed()
        #print confusionmat
        #embed()
        TP_index[k] = confusionmat[:,1].argmax() #e.g 0 in above case
        TP[k] = confusionmat[TP_index[k],1] #e.g 1495 in above case
        FP[k] = confusionmat[TP_index[k],0] #e.g. 
        #Numhybridspikes = np.sum(confusionMKKrand[k][:,1])
        FN[k] = np.sum(confusionmat[:,1]) - TP[k]
        TN[k] = np.sum(confusionmat[:,0]) - FP[k]
        false_discovery_rate[k] = np.true_divide(FP[k],(FP[k]+TP[k]))
        true_positive_rate[k] = np.true_divide(TP[k],(TP[k]+FN[k]))
              
    return TP, FP, FN, TN, false_discovery_rate,true_positive_rate 

def plotROC_KK(graphpath,KKconflist):
    #KKconflist is is a list of confusion matrices
         
    
    #e.g. array([[   66,  1495,     0,     0,     0,     0,     0,     0,     0,
    #        0,     0,     0,     0,     0,     0,     0,     0,     0,
    #        0,     0,     0,     0,     0,     0,     0,     0,     0,
     #       0],
    #     [ 5268,    32,     0,     0,     0,     0,     0,     0,     0,
    #        0,     0,     0,     0,     0,     0,     0,     0,     0,
    #        0,     0,     0,     0,     0,     0,     0,     0,     0,
     #       0]])
    #[ 4404,     0,     0,     0,     0,     0,     0,     0,     0,
    #        0,     0,     0,     0,     0,     0,     0,     0,     0,
    #        0,     0,     0,     0,     0,     0,     0,     0,     0,
    #        0],
    
    errors = Error_rates_KK(KKconflist)
    
    fig1 = plt.figure()
    axes1 = fig1.add_axes([0.1,0.1,0.8,0.8]) 
    #Area on page not axes
    #dreadful notation, but we are stuck with it!
    axes1.hold(True)
    #colours = ['r', 'g', 'b']
    axes1.set_xlim([0,1])
    axes1.set_ylim([0,1])

    axes1.set_xlabel('False discovery rate')
    axes1.set_ylabel('True positive rate')
    
    false_discovery_rate = errors[4]
    true_positive_rate = errors[5]
    
    fig1.suptitle(' %s  '%(graphpath), fontsize=14, fontweight='bold')
    #for i in np.arange(fourclu.shape[0]):
    axes1.scatter(false_discovery_rate,true_positive_rate, color = 'b')   
    ratepoints = zip(false_discovery_rate,true_positive_rate)
    hull = ConvexHull(ratepoints)
        
    #for simplex in hull.simplices:
    #    for data in itertools.combinations(simplex.coords,2):
    #        data = np.array(data)
    #        print data
    #        print 'next!'
    #        axes1.plot(data[:,0],data[:,1],color = 'b')
            
    plt.show()
    fig1.savefig('%s.pdf'%(graphpath))      
    #plt.figure(2)
    
    #fig2 = plt.figure(2)
    #axes2 = fig2.add_axes([0.1,0.1,0.8,0.8]) 
    
    #return ratepoints
    #return xes, yes, 
    return hull, ratepoints, errors   
    
def plotROC_KK_several(graphpath,KKconflistsquared):
    fig1 = plt.figure()
    axes1 = fig1.add_axes([0.1,0.1,0.8,0.8]) 
    #Area on page not axes
    #dreadful notation, but we are stuck with it!
    axes1.hold(True)
    colours = ['r', 'g', 'b']
    axes1.set_xlim([0,1])
    axes1.set_ylim([0,1])
    axes1.set_xlabel('False discovery rate')
    axes1.set_ylabel('True positive rate')
    
    for k in np.arange(len(KKconflistsquared)):
        errors = Error_rates_KK(KKconflistsquared[k])  
        false_discovery_rate = errors[4]
        true_positive_rate = errors[5]
        axes1.scatter(false_discovery_rate,true_positive_rate, color = colours[np.mod(k,len(colours))])   
        
        
    plt.show()
    fig1.savefig('%s.pdf'%(graphpath))   
    
    
def plotROCcomplete(graphpath,fourclu,KKconflistsquared):
    false_discovery_rate, true_positive_rate = supe.pre_plotROC(fourclu)          
    
    fig1 = plt.figure(1)
    axes1 = fig1.add_axes([0.1,0.1,0.8,0.8]) 
    #Area on page not axes
    #dreadful notation, but we are stuck with it!
    axes1.hold(True)
    colours = ['r', 'g', 'b']
    axes1.set_xlim([0,1])
    axes1.set_ylim([0,1])
    #plt.xlim([0,1]) 
    #plt.ylim([0,1])
    axes1.set_xlabel('False discovery rate')
    axes1.set_ylabel('True positive rate')
    fig1.suptitle(' %s  '%(graphpath), fontsize=14, fontweight='bold')
    #embed()
    for i in np.arange(fourclu.shape[0]):
        axes1.scatter(false_discovery_rate[i,:],true_positive_rate[i,:],marker = 'x', color = 'c')   
        ratepoints = zip(false_discovery_rate[i,:],true_positive_rate[i,:])
        hull = ConvexHull(ratepoints)
        #simplexlist = [simplex for simplex in hull.simplices]
        #print simplexlist
        for simplex in hull.simplices:
            for data in itertools.combinations(simplex.coords,2):
                data = np.array(data)
                print data
                print 'next!'
                axes1.plot(data[:,0],data[:,1],color = 'c')
            #print vertex_index
            #xes = ratepoints[vertex_index,0]
            #print xes
            #yes = ratepoints[vertex_index,1]
            #axes1.plot(ratepoints[vertex_index,0],ratepoints[vertex_index,1],'k-')
    #plt.show()
    #plt.savefig('%s_%g.pdf'%(graphpath,i))    
    errorlist = []
    for k in np.arange(len(KKconflistsquared)):
        errors = Error_rates_KK(KKconflistsquared[k])  
        false_discovery_rateKK = errors[4]
        true_positive_rateKK = errors[5]
        axes1.scatter(false_discovery_rateKK,true_positive_rateKK, color = colours[np.mod(k,len(colours))])
        #embed()
        errorlist.append(errors)
    
    fig1.savefig('%s'%(graphpath))      
    #plt.figure(2)
    plt.show()
    #fig2 = plt.figure(2)
    #axes2 = fig2.add_axes([0.1,0.1,0.8,0.8]) 
    
    #return ratepoints
    #return xes, yes, 
    return errorlist, hull, ratepoints, false_discovery_rate, true_positive_rate   

def plotROCcomplete_noshow(graphpath,fourclu,KKconflistsquared):
    false_discovery_rate, true_positive_rate = supe.pre_plotROC(fourclu)          
    
    fig1 = plt.figure(1)
    axes1 = fig1.add_axes([0.1,0.1,0.8,0.8]) 
    #Area on page not axes
    #dreadful notation, but we are stuck with it!
    axes1.hold(True)
    colours = ['r', 'g', 'b']
    axes1.set_xlim([0,1])
    axes1.set_ylim([0,1])
    #plt.xlim([0,1]) 
    #plt.ylim([0,1])
    axes1.set_xlabel('False discovery rate')
    axes1.set_ylabel('True positive rate')
    fig1.suptitle(' %s  '%(graphpath), fontsize=14, fontweight='bold')
    for i in np.arange(fourclu.shape[0]):
        axes1.scatter(false_discovery_rate[i,:],true_positive_rate[i,:],marker = 'x', color = 'c')   
        ratepoints = zip(false_discovery_rate[i,:],true_positive_rate[i,:])
        hull = ConvexHull(ratepoints)
        #simplexlist = [simplex for simplex in hull.simplices]
        #print simplexlist
        for simplex in hull.simplices:
            for data in itertools.combinations(simplex.coords,2):
                data = np.array(data)
                print data
                print 'next!'
                axes1.plot(data[:,0],data[:,1],color = 'c')
            #print vertex_index
            #xes = ratepoints[vertex_index,0]
            #print xes
            #yes = ratepoints[vertex_index,1]
            #axes1.plot(ratepoints[vertex_index,0],ratepoints[vertex_index,1],'k-')
    #plt.show()
    #plt.savefig('%s_%g.pdf'%(graphpath,i))    
    
    for k in np.arange(len(KKconflistsquared)):
        errors = Error_rates_KK(KKconflistsquared[k])  
        false_discovery_rateKK = errors[4]
        true_positive_rateKK = errors[5]
        axes1.scatter(false_discovery_rateKK,true_positive_rateKK, color = colours[np.mod(k,len(colours))])
    
    
    fig1.savefig('%s'%(graphpath))      
    #plt.figure(2)
    #plt.show()
    #fig2 = plt.figure(2)
    #axes2 = fig2.add_axes([0.1,0.1,0.8,0.8]) 
    
    #return ratepoints
    #return xes, yes, 
    return hull, ratepoints, false_discovery_rate, true_positive_rate   

def plotROCcomplete_simple(graphpath,fourclu,KKconflistsquared):
    false_discovery_rate, true_positive_rate = supe.pre_plotROC(fourclu)          
    
    fig1 = plt.figure(1)
    axes1 = fig1.add_axes([0.1,0.1,0.8,0.8]) 
    #Area on page not axes
    #dreadful notation, but we are stuck with it!
    axes1.hold(True)
    colours = ['r', 'g', 'b']
    axes1.set_xlim([0,1])
    axes1.set_ylim([0,1])
    #plt.xlim([0,1]) 
    #plt.ylim([0,1])
    axes1.set_xlabel('False discovery rate')
    axes1.set_ylabel('True positive rate')
    fig1.suptitle(' %s  '%(graphpath), fontsize=14, fontweight='bold')
    for i in np.arange(fourclu.shape[0]):
        axes1.scatter(false_discovery_rate[i,:],true_positive_rate[i,:],marker = 'x', color = 'c')   
        
        #Simple plot - No convex hull 
    
    for k in np.arange(len(KKconflistsquared)):
        errors = Error_rates_KK(KKconflistsquared[k])  
        false_discovery_rateKK = errors[4]
        true_positive_rateKK = errors[5]
        axes1.scatter(false_discovery_rateKK,true_positive_rateKK, color = colours[np.mod(k,len(colours))])
    
    plt.show()
    fig1.savefig('%s'%(graphpath))      
    #plt.figure(2)
    
    #fig2 = plt.figure(2)
    #axes2 = fig2.add_axes([0.1,0.1,0.8,0.8]) 
    
    #return ratepoints
    #return xes, yes, 
    return  false_discovery_rate, true_positive_rate   

def make_one_list(fourclulist):
    false_discovery_rate = []
    true_positive_rate = []
    for s, tclu in enumerate(fourclulist):
        fdr, tpr = supe.pre_plotROC(tclu)  
        for c in np.arange(fdr.shape[0]):
            false_discovery_rate.append(fdr)
            true_positive_rate.append(tpr)        
    return false_discovery_rate, true_positive_rate        

def hstack_fourclulist(fourclulist):
    false_discovery_rate, true_positive_rate = make_one_list(fourclulist)
    fdr = false_discovery_rate[0]
    tpr = true_positive_rate[0]
    for l in np.arange(len(false_discovery_rate)-1):
        fdr = np.hstack((fdr, false_discovery_rate[l+1]))
        tpr = np.hstack((tpr, true_positive_rate[l+1]))
    return fdr, tpr
    
def plotROCcomplete_multiple(graphpath,fourclulist,KKconflistsquared):

    false_discovery_rate,true_positive_rate= hstack_fourclulist(fourclulist)          
    Numsuppoints = false_discovery_rate.shape[1]
    print 'Number of supervised learning points = ', Numsuppoints 
    
    fig1 = plt.figure()
    axes1 = fig1.add_axes([0.1,0.1,0.8,0.8]) 
    #Area on page not axes
    #dreadful notation, but we are stuck with it!
    axes1.hold(True)
    colours = ['r', 'g', 'b']
    axes1.set_xlim([0,1])
    axes1.set_ylim([0,1])
    #plt.xlim([0,1]) 
    #plt.ylim([0,1])
    axes1.set_xlabel('False discovery rate')
    axes1.set_ylabel('True positive rate')
    fig1.suptitle(' %s  '%(graphpath), fontsize=14, fontweight='bold')
   
    axes1.scatter(false_discovery_rate,true_positive_rate,marker = 'x', color = 'c')   
    ratepoints = zip(false_discovery_rate.reshape(Numsuppoints),true_positive_rate.reshape(Numsuppoints))
    print ratepoints
    hull = ConvexHull(ratepoints)
        #simplexlist = [simplex for simplex in hull.simplices]
        #print simplexlist
        
        
    for simplex in hull.simplices:
            for data in itertools.combinations(simplex.coords,2):
                data = np.array(data)
                print data
                print 'next!'
                axes1.plot(data[:,0],data[:,1],color = 'c')    
    
    #plt.show()
    #plt.savefig('%s_%g.pdf'%(graphpath,i))    
    
    for k in np.arange(len(KKconflistsquared)):
        errors = Error_rates_KK(KKconflistsquared[k])  
        false_discovery_rateKK = errors[4]
        true_positive_rateKK = errors[5]
        axes1.scatter(false_discovery_rateKK,true_positive_rateKK, color = colours[np.mod(k,len(colours))])
    
    
    fig1.savefig('%s'%(graphpath))      
    #plt.figure(2)
    plt.show()
    #fig2 = plt.figure(2)
    #axes2 = fig2.add_axes([0.1,0.1,0.8,0.8]) 
    
    #return ratepoints
    #return xes, yes, 
    return hull, ratepoints, false_discovery_rate, true_positive_rate   

def plotROCcomplete_multiple_convex(graphpath,title,fourclulist,KKconflistsquared, labellist):
    #labellist = ['Classical EM', 'Masked EM', 'Masked EM with maskstarts']
    false_discovery_rate,true_positive_rate= hstack_fourclulist(fourclulist)          
    Numsuppoints = false_discovery_rate.shape[1]
    print 'Number of supervised learning points = ', Numsuppoints 
    
    fig1 = plt.figure(frameon = False)
    axes1 = fig1.add_axes([0.1,0.1,0.8,0.8]) 
    #Area on page not axes
    #dreadful notation, but we are stuck with it!
    axes1.hold(True)
    colours = ['r', 'b', 'g']
    axes1.set_xlim([0,1])
    axes1.set_ylim([0,1])
    #plt.xlim([0,1]) 
    #plt.ylim([0,1])
    axes1.set_xlabel('False discovery rate',fontsize=12, fontweight='bold')
    axes1.set_ylabel('True positive rate',fontsize=12, fontweight='bold')
    #fig1.suptitle(' %s  '%(title), fontsize=14, fontweight='bold')
   
    #axes1.scatter(false_discovery_rate,true_positive_rate,marker = 'x', color = 'c')   
    #axes1.legend(loc = 4, label = 'SVM') 
    ratepoints = zip(false_discovery_rate.reshape(Numsuppoints),true_positive_rate.reshape(Numsuppoints))
    print ratepoints
    ratepoints.append((0,0))
    ratepoints.append((1,0))
    ratepoints.append((1,1))
    
    #embed()
    hull = ConvexHull(ratepoints)
        #simplexlist = [simplex for simplex in hull.simplices]
        #print simplexlist
        
        
    for simplex in hull.simplices:
            for data in itertools.combinations(simplex.coords,2):
                data = np.array(data)
                #print 'data.shape = ', data.shape
                #print 'next!'
                klast = axes1.plot(data[:,0],data[:,1],color = 'c', linewidth=2)  
    axes1.plot([1,1],[0,1],color = 'c',linewidth=2,label = 'Theoretical upper bound')            
    #axes1.legend(loc = 4, label = klast) 
    
    #plt.show()
    #plt.savefig('%s_%g.pdf'%(graphpath,i))    
    
    for k in np.arange(len(KKconflistsquared)):
        errors = Error_rates_KK(KKconflistsquared[k])  
        false_discovery_rateKK = errors[4]
        true_positive_rateKK = errors[5]
        axes1.scatter(false_discovery_rateKK,true_positive_rateKK, s = 50,color = colours[np.mod(k,len(colours))],label = labellist[k])
    axes1.legend(loc = 3, numpoints = 1,prop = {'size':12}, scatterpoints = 1)  
    
    axes1.plot([1,1],[0,1],color = 'k',linewidth=2)
    axes1.plot([0,1],[0,0],color = 'k',linewidth=2)
    
    fig1.savefig('%s'%(graphpath))      
    #plt.figure(2)
    plt.show()
    #fig2 = plt.figure(2)
    #axes2 = fig2.add_axes([0.1,0.1,0.8,0.8]) 
    
    #return ratepoints
    #return xes, yes, 
    return hull, ratepoints, false_discovery_rate, true_positive_rate  
    
def fourbyfourconfusion_sup(fourclu):
    errorquad = supe.compute_errors(fourclu)     
    simplified = np.zeros((errorquad[0].shape[1],2,2))
    simplified[:,0,0] = errorquad[0][0,:] #TN
    simplified[:,0,1] = errorquad[1][0,:] #FN
    simplified[:,1,0] = errorquad[2][0,:] #FP
    simplified[:,1,1] = errorquad[3][0,:] #FP   
    simplified = simplified.astype(np.int32)
    
    return simplified      
    
def distance_from_perfection(false_discovery_rate, true_positive_rate):
    numplottedpoints = len(false_discovery_rate)
    tuply = zip(false_discovery_rate, true_positive_rate)
    tuply = np.array(tuply).reshape((numplottedpoints,2))
    perfdist = LA.norm(tuply[:]-[0,1],axis = 1) #distance from point (0,1)
    return perfdist

def plot_errors_classweights_C(classweights, fourclu, Clist):
    
    false_discovery_rate, true_positive_rate = supe.pre_plotROC(fourclu)  
    #false_discovery_train_rate, true_positive_train_rate = supe.pre_plotROC(fourclutest)  
    
    weightzero = np.zeros(len(classweights))
    weightone = np.zeros(len(classweights))
    #Extract the weights
    for i in np.arange(len(classweights)):
        weightzero[i] = classweights[i][0]
        weightone[i] = classweights[i][1]
        
    
    fig1 = plt.figure()
    axes1 = fig1.add_axes([0.1,0.1,0.8,0.8]) 
    
    axes1.set_xlabel('log(C*weight 0)')
    axes1.set_ylabel('log(C*weight 1)')
    axes1.hold(True)
    
    numCvalues = len(Clist)
    #Number of values of C, 
    # fourclu should have shape (numCvalues, len(classweights), numspikes)
    cm = plt.cm.get_cmap('RdYlBu')
    for c, cval in enumerate(Clist):
        sc = axes1.scatter(np.log(cval*weightzero[:]),np.log(cval*weightone[:]), c=false_discovery_rate[c,:] , cmap=cm)
        plt.colorbar(sc)
    plt.show()
    return axes1
    
def plot_errors_classweights_C(classweights, fourclutest, fourclutrain, Clist):
    
    false_discovery_rate, true_positive_rate = supe.pre_plotROC(fourclutest)  
    false_discovery_train_rate, true_positive_train_rate = supe.pre_plotROC(fourclutrain)  
    
    weightzero = np.zeros(len(classweights))
    weightone = np.zeros(len(classweights))
    #Extract the weights
    for i in np.arange(len(classweights)):
        weightzero[i] = classweights[i][0]
        weightone[i] = classweights[i][1]
        
    
    fig = plt.figure()
    #axes1 = fig1.add_axes([0.1,0.1,0.8,0.8]) 
    axes1 = fig.add_subplot(221)
    axes1.set_xlabel('log(C*weight 0)')
    axes1.set_ylabel('log(C*weight 1)')
    axes1.set_title('Test set FDR')
    axes1.hold(True)
    axes2 = fig.add_subplot(222)
    axes2.set_xlabel('log(C*weight 0)')
    axes2.set_ylabel('log(C*weight 1)')
    axes2.set_title('Test set 1-TPR')
    axes2.hold(True)
    axes3 = fig.add_subplot(223)
    axes3.set_xlabel('log(C*weight 0)')
    axes3.set_ylabel('log(C*weight 1)')
    axes3.set_title('Training set FDR')
    axes3.hold(True)
    axes4 = fig.add_subplot(224)
    axes4.set_xlabel('log(C*weight 0)')
    axes4.set_ylabel('log(C*weight 1)')
    axes4.set_title('Training set 1-TPR')
    axes4.hold(True)
    
    numCvalues = len(Clist)
    #Number of values of C, 
    # fourclu should have shape (numCvalues, len(classweights), numspikes)
    cm = plt.cm.get_cmap('RdYlBu')
    sizeadjust = 100
    for c, cval in enumerate(Clist):
        sc1 = axes1.scatter(np.log(cval*weightzero[:]),np.log(cval*weightone[:]),marker = 's',s=sizeadjust, c=false_discovery_rate[c,:] , cmap=cm)
        sc1.set_clim(vmin=0,vmax=1)
        fig.colorbar(sc1, ax=axes1, shrink=0.9)
        sc2 = axes2.scatter(np.log(cval*weightzero[:]),np.log(cval*weightone[:]),marker = 's',s=sizeadjust, c=(1-true_positive_rate[c,:]) , cmap=cm)
        #plt.colorbar(sc2)
        sc2.set_clim(vmin=0,vmax=1)
        fig.colorbar(sc2, ax=axes2, shrink=0.9)
        sc3 = axes3.scatter(np.log(cval*weightzero[:]),np.log(cval*weightone[:]),marker = 's',s=sizeadjust, c=false_discovery_train_rate[c,:] , cmap=cm)
        #plt.colorbar(sc3)
        sc3.set_clim(vmin=0,vmax=1)
        fig.colorbar(sc3, ax=axes3, shrink=0.9)
        sc4 = axes4.scatter(np.log(cval*weightzero[:]),np.log(cval*weightone[:]), marker = 's',s=sizeadjust,c=(1-true_positive_train_rate[c,:]) , cmap=cm)
        sc4.set_clim(vmin=0,vmax=1)
        fig.colorbar(sc4, ax=axes4, shrink=0.9)
        #plt.colorbar(sc4)
    plt.show()
    #return axes1    
#def plot_errors_classweights_C_resultslist(resultslist):
    
#    for result in enumerate(resultslist):



def get_execution_times(hybdatadict, SDparams,prb,detectioncrit, defaultKKparams, paramtochange, listparamvalues, extralabel = None):
    outlistKK = rkk.one_param_varyKK(hybdatadict, SDparams,prb,detectioncrit, defaultKKparams, paramtochange, listparamvalues)
    outputdir = hybdatadict['output_path']
    #embed()
    
    
    for k, basefilename in enumerate(outlistKK[0]):
        p= subprocess.Popen(['grep "That took" %s/%s.klg.1'%(outputdir,basefilename)], stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        out, err = p.communicate()
        #p= subprocess.Popen(['grep','"That took"', '%s.clu.1'%(basefilename)], stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        #p= subprocess.Popen(['more', '%s.clu.1'%(basefilename)], stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        print out
    return out    
        
        
        
    
def get_execution_times_ind(hybdatadict, SDparams,prb, defaultKKparams, paramtochange, listparamvalues, extralabel = None):
    outlistKK = rkk.one_param_varyKK_ind(hybdatadict, SDparams,prb, defaultKKparams, paramtochange, listparamvalues)
    outputdir = hybdatadict['output_path']
    
    
    for k, basefilename in enumerate(outlistKK[0]):
        p= subprocess.Popen(['/bin/sh','-c','grep "That took" %s.klg.1'%(basefilename)], stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        out, err = p.communicate()
        #p= subprocess.Popen(['grep','"That took"', '%s.clu.1'%(basefilename)], stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        #p= subprocess.Popen(['more', '%s.clu.1'%(basefilename)], stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        print out  
    return out              
        
def get_execution_times_ind_simple(hybdatadict, SDparams,prb, defaultKKparams, paramtochange, listparamvalues,outtimesfile, extralabel = None):
    outlistKK = rkk.one_param_varyKK_ind(hybdatadict, SDparams,prb, defaultKKparams, paramtochange, listparamvalues)
    outputdir = hybdatadict['output_path']
    
    
    for k, basefilename in enumerate(outlistKK[0]):
        os.system('echo "%s" >> %s/%s '%(basefilename,outputdir,outtimesfile))
        os.system('more %s/%s.klg.1 | grep "MinClusters">> %s/%s'%(outputdir,basefilename,outputdir,outtimesfile))
        os.system('more %s/%s.klg.1 | grep "MaxClusters">> %s/%s'%(outputdir,basefilename,outputdir,outtimesfile))
        os.system('more %s/%s.klg.1 | grep "Maskstarts">> %s/%s'%(outputdir,basefilename,outputdir,outtimesfile))
        os.system('more %s/%s.klg.1 | grep "UseDistributional">> %s/%s'%(outputdir,basefilename,outputdir,outtimesfile))
        os.system('more %s/%s.klg.1 | grep "PenaltyK">> %s/%s'%(outputdir,basefilename,outputdir,outtimesfile))
        os.system('more %s/%s.klg.1 | grep "That took">> %s/%s'%(outputdir,basefilename,outputdir,outtimesfile))
        os.system('more %s/%s.klg.1 | grep "iterations">> %s/%s'%(outputdir,basefilename,outputdir,outtimesfile))
        os.system('more %s/%s.klg.1 | grep "Iterations">> %s/%s'%(outputdir,basefilename,outputdir,outtimesfile))
        
def get_execution_times_simple(hybdatadict, SDparams,prb,detectioncrit,defaultKKparams, paramtochange, listparamvalues,outtimesfile, extralabel = None):
    outlistKK = rkk.one_param_varyKK(hybdatadict, SDparams,prb,detectioncrit,  defaultKKparams, paramtochange, listparamvalues)
    outputdir = hybdatadict['output_path']
    
    
    for k, basefilename in enumerate(outlistKK[0]):
        os.system('echo "%s" >> %s/%s '%(basefilename,outputdir,outtimesfile))
        os.system('more %s/%s.klg.1 | grep "MinClusters">> %s/%s'%(outputdir,basefilename,outputdir,outtimesfile))
        os.system('more %s/%s.klg.1 | grep "MaxClusters">> %s/%s'%(outputdir,basefilename,outputdir,outtimesfile))
        os.system('more %s/%s.klg.1 | grep "UseDistributional">> %s/%s'%(outputdir,basefilename,outputdir,outtimesfile))
        os.system('more %s/%s.klg.1 | grep "PenaltyK">> %s/%s'%(outputdir,basefilename,outputdir,outtimesfile))
        os.system('more %s/%s.klg.1 | grep "That took">> %s/%s'%(outputdir,basefilename,outputdir,outtimesfile))
        os.system('more %s/%s.klg.1 | grep "iterations">> %s/%s'%(outputdir,basefilename,outputdir,outtimesfile))
        os.system('more %s/%s.klg.1 | grep "Iterations">> %s/%s'%(outputdir,basefilename,outputdir,outtimesfile))         
    
