import numpy as np
import matplotlib.pyplot as plt


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
        

