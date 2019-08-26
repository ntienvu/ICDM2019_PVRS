# -*- coding: utf-8 -*-
"""

"""
from __future__ import division

import sys
sys.path.insert(0,'../../')
sys.path.insert(0,'..')
import numpy as np
#import mayavi.mlab as mlab
#from scipy.stats import norm
#import matplotlib as plt
from mpl_toolkits.mplot3d import Axes3D
from bayes_opt import BayesOpt
from bayes_opt.batchBO.batch_pvrs import BatchPVRS

#from bayes_opt import PradaBayOptBatch
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.metrics.pairwise import euclidean_distances
from bayes_opt.acquisition_maximization import acq_max
from scipy.stats import norm as norm_dist

import random
from bayes_opt.acquisition_functions import AcquisitionFunction, unique_rows
import os
from pylab import *

cdict = {'red': ((0.0, 0.0, 0.0),
                  (0.5, 1.0, 0.7),
                  (1.0, 1.0, 1.0)),
          'green': ((0.0, 0.0, 0.0),
                    (0.5, 1.0, 0.0),
                    (1.0, 1.0, 1.0)),
          'blue': ((0.0, 0.0, 0.0),
                   (0.5, 1.0, 0.0),
                   (1.0, 0.5, 1.0))}

#my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)
#my_cmap = plt.get_cmap('cubehelix')
my_cmap = plt.get_cmap('Blues')

        
counter = 0

#class Visualization(object):
    
    #def __init__(self,bo):
       #self.plot_gp=0     
       #self.posterior=0
       #self.myBo=bo
       
def plot_bo(bo):
    if bo.dim==1:
        plot_bo_1d(bo)
    if bo.dim==2:
        plot_bo_2d(bo)

        
def plot_acq_bo_1d(bo):
    
    global counter
    counter=counter+1
    
    func=bo.f
    #x_original = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 1000)
    x_original=x*bo.max_min_gap+bo.bounds[:,0]

    y_original = func(x_original)
    #y = func(x)
    #y_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)

    fig=plt.figure(figsize=(10, 10))
    fig.suptitle('Gaussian Process and Utility Function After {} Points'.format(len(bo.X)), fontdict={'size':18})
    
    gs = gridspec.GridSpec(8, 1, height_ratios=[3, 1,1,1,1,1,1,1]) 
    axis = plt.subplot(gs[0])
    acq_UCB = plt.subplot(gs[1])
    acq_EI = plt.subplot(gs[2])
    acq_POI = plt.subplot(gs[3])
    
    #acq_TS2 = plt.subplot(gs[5])
    acq_ES = plt.subplot(gs[4])
    acq_PES = plt.subplot(gs[5])
    acq_MRS = plt.subplot(gs[6])
    
    acq_Consensus = plt.subplot(gs[7])

    mu, sigma = bo.posterior(x)
    #mu_original=mu*(np.max(y_original)-np.min(y_original))+np.mean(y_original)
    mu_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)
    sigma_original=sigma*np.std(bo.Y_original)+np.mean(bo.Y_original)**2
    
    axis.plot(x_original, y_original, linewidth=3, label='Real Function')
    axis.plot(bo.X_original.flatten(), bo.Y_original, 'D', markersize=8, label=u'Observations', color='r')
    axis.plot(x_original, mu_original, '--', color='k', label='GP mean')
    
    #samples*bo.max_min_gap+bo.bounds[:,0]
    
    temp_xaxis=np.concatenate([x_original, x_original[::-1]])
    #temp_xaxis=temp*bo.max_min_gap+bo.bounds[:,0]
    
    temp_yaxis_original=np.concatenate([mu_original - 1.9600 * sigma_original, (mu_original + 1.9600 * sigma_original)[::-1]])
    temp_yaxis=np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]])
    temp_yaxis_original2=temp_yaxis*np.std(bo.Y_original)+np.mean(bo.Y_original)
    axis.fill(temp_xaxis, temp_yaxis_original2,alpha=.6, fc='c', ec='None', label='95% CI')
    
    axis.set_xlim((np.min(x_original), np.max(x_original)))
    #axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size':16})
    axis.set_xlabel('x', fontdict={'size':16})
    
    # UCB 
    acq_func={}
    acq_func['name']='ucb'
    acq_func['kappa']=2
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp, np.max(bo.Y))
    acq_UCB.plot(x_original, utility, label='Utility Function', color='purple')
    acq_UCB.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)

    # check batch BO     
    try:
        nSelectedPoints=np.int(bo.NumPoints[-1])
    except:
        nSelectedPoints=1
    max_point=np.max(utility)
    
    #acq_UCB.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         #label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
             
    acq_UCB.set_xlim((np.min(x_original), np.max(x_original)))
    acq_UCB.set_ylabel('UCB', fontdict={'size':16})
    acq_UCB.set_xlabel('x', fontdict={'size':16})
    
    
    # EI 
    acq_func={}
    acq_func['name']='ei'
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds
    myacq=AcquisitionFunction(acq_func)
	
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp, np.max(bo.Y))
    acq_EI.plot(x_original, utility, label='Utility Function', color='purple')
    acq_EI.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)

    max_point=np.max(utility)
    
    #acq_EI.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         #label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
             
    acq_EI.set_xlim((np.min(x_original), np.max(x_original)))
    acq_EI.set_ylabel('EI', fontdict={'size':16})
    acq_EI.set_xlabel('x', fontdict={'size':16})
    
    
    # POI 
    acq_func={}
    acq_func['name']='poi'
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds
    myacq=AcquisitionFunction(acq_func)
	
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp, np.max(bo.Y))
    acq_POI.plot(x_original, utility, label='Utility Function', color='purple')
    acq_POI.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)

    max_point=np.max(utility)
    
    #acq_POI.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         #label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
             
    acq_POI.set_xlim((np.min(x_original), np.max(x_original)))
    acq_POI.set_ylabel('POI', fontdict={'size':16})
    acq_POI.set_xlabel('x', fontdict={'size':16})
    	
    #axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #acq_EI.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    
    # MRS     
    acq_func={}
    acq_func['name']='mrs'
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
	
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp, np.max(bo.Y))
    acq_MRS.plot(x_original, utility, label='Utility Function', color='purple')
    acq_MRS.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)

    max_point=np.max(utility)
    
    #acq_MRS.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         #label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
             
    acq_MRS.set_xlim((np.min(x_original), np.max(x_original)))
    acq_MRS.set_ylabel('MRS', fontdict={'size':16})
    acq_MRS.set_xlabel('x', fontdict={'size':16})
	

    # PES
    acq_func={}
    acq_func['name']='pes'
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
	
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp, np.max(bo.Y))
    acq_PES.plot(x_original, utility, label='Utility Function', color='purple')
    acq_PES.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)

    max_point=np.max(utility)
    
    #acq_PES.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         #label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
             
    acq_PES.set_xlim((np.min(x_original), np.max(x_original)))
    acq_PES.set_ylabel('PES', fontdict={'size':16})
    acq_PES.set_xlabel('x', fontdict={'size':16})
     
    # TS1   
    
    acq_func={}
    acq_func['name']='consensus'
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
	
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp, np.max(bo.Y))
    acq_Consensus.plot(x_original, utility, label='Utility Function', color='purple')


    temp=np.asarray(myacq.object.xstars)
    xt_suggestion_original=temp*bo.max_min_gap+bo.bounds[:,0]
    
    acq_Consensus.plot(xt_suggestion_original, [np.max(utility)]*xt_suggestion_original.shape[0], 's', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='red', markeredgecolor='k', markeredgewidth=1)
   
    max_point=np.max(utility)
    
    acq_Consensus.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
        
    #acq_TS.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         #label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
             
    acq_Consensus.set_xlim((np.min(x_original), np.max(x_original)))
    #acq_TS.set_ylim((np.min(utility)*0.9, np.max(utility)*1.1))
    acq_Consensus.set_ylabel('Consensus', fontdict={'size':16})
    acq_Consensus.set_xlabel('x', fontdict={'size':16})


    # ES     
    acq_func={}
    acq_func['name']='es'
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
	
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp, np.max(bo.Y))
    acq_ES.plot(x_original, utility, label='Utility Function', color='purple')
    acq_ES.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)

    max_point=np.max(utility)
    
    #acq_ES.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         #label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
             
    acq_ES.set_xlim((np.min(x_original), np.max(x_original)))
    acq_ES.set_ylabel('ES', fontdict={'size':16})
    acq_ES.set_xlabel('x', fontdict={'size':16})
    
    strFileName="{:d}_GP_acquisition_functions.eps".format(counter)
    fig.savefig(strFileName, bbox_inches='tight')
    #axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #acq_TS.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)

def plot_acq_bo_1d_vrs(bo):
    
    global counter
    counter=counter+1
    
    func=bo.f
    #x_original = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 1000)
    x_original=x*bo.max_min_gap+bo.bounds[:,0]

    y_original = func(x_original)
    #y = func(x)
    #y_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)

    fig=plt.figure(figsize=(10, 11))
    #fig.suptitle('Gaussian Process and Utility Function After {} Points'.format(len(bo.X)), fontdict={'size':18})
    
    gs = gridspec.GridSpec(8, 1, height_ratios=[2, 1,1,1,1,1,1,1]) 
    axis = plt.subplot(gs[0])
    acq_UCB = plt.subplot(gs[1])
    acq_EI = plt.subplot(gs[2])
    #acq_POI = plt.subplot(gs[3])
    
    #acq_TS2 = plt.subplot(gs[5])
    acq_MES = plt.subplot(gs[3])    

    acq_ES = plt.subplot(gs[4])
    acq_MRS = plt.subplot(gs[5])
    acq_PES = plt.subplot(gs[6])
    
    acq_Consensus = plt.subplot(gs[7])

    mu, sigma = bo.posterior(x)
    # get maximum of mu function
    mu_max=mu.max()
    #mu_original=mu*(np.max(y_original)-np.min(y_original))+np.mean(y_original)
    mu_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)
    sigma_original=sigma*np.std(bo.Y_original)+np.mean(bo.Y_original)**2
    
    axis.plot(x_original, y_original, linewidth=3, label='f(x)')
    axis.plot(bo.X_original.flatten(), bo.Y_original, 'o', markersize=8, label=u'Data X', color='g')
    axis.plot(x_original, mu_original, '--', color='k', label='$\mu(x)$')
    #samples*bo.max_min_gap+bo.bounds[:,0]
    
    temp_xaxis=np.concatenate([x_original, x_original[::-1]])
    #temp_xaxis=temp*bo.max_min_gap+bo.bounds[:,0]
    
    temp_yaxis_original=np.concatenate([mu_original - 1.9600 * sigma_original, (mu_original + 1.9600 * sigma_original)[::-1]])
    temp_yaxis=np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]])
    temp_yaxis_original2=temp_yaxis*np.std(bo.Y_original)+np.mean(bo.Y_original)
    axis.fill(temp_xaxis, temp_yaxis_original2,alpha=.3, fc='c', ec='None', label='$\sigma(x)$')
    axis.get_xaxis().set_visible(False)
    
    axis.set_yticklabels([])
    axis.set_xticklabels([])
    

    axis.set_xlim((np.min(x_original)-0.05, np.max(x_original)+0.05))
    #axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size':16})
    axis.set_xlabel('x', fontdict={'size':16})

    axis.legend(loc='center left', bbox_to_anchor=(0.01, 1.15),prop={'size':16},ncol=6)


    # UCB 
    acq_func={}
    acq_func['name']='ucb'
    acq_func['kappa']=2
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp, np.max(bo.Y))
    acq_UCB.plot(x_original, utility, label='Utility Function', color='purple')
    acq_UCB.plot(x_original[np.argmax(utility)], np.max(utility), 's', markersize=10, 
             label=u'Next Best Guess', markerfacecolor='red', markeredgecolor='k', markeredgewidth=1)

       
    # check batch BO     
    try:
        nSelectedPoints=np.int(bo.NumPoints[-1])
    except:
        nSelectedPoints=1
    max_point=np.max(utility)
    
    acq_UCB.get_xaxis().set_visible(False)
    
    acq_UCB.set_yticklabels([])
    acq_UCB.set_xticklabels([])
    
    #acq_UCB.get_yaxis().set_visible(False) 
    
    #acq_UCB.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         #label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
       
        
    acq_UCB.set_xlim((np.min(x_original)-0.05, np.max(x_original)+0.05))
    acq_UCB.set_ylim((np.min(utility), 1.2*np.max(utility)))
    acq_UCB.set_ylabel('UCB', fontdict={'size':16})
    acq_UCB.set_xlabel('x', fontdict={'size':16})

    
    # EI 
    acq_func={}
    acq_func['name']='ei'
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds
    myacq=AcquisitionFunction(acq_func)
	
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp, np.max(bo.Y))
    acq_EI.plot(x_original, utility, label='Utility Function', color='purple')
    acq_EI.plot(x_original[np.argmax(utility)], np.max(utility), 's', markersize=10, 
             label=u'Next Best Guess', markerfacecolor='red', markeredgecolor='k', markeredgewidth=1)

    max_point=np.max(utility)
    
    acq_EI.get_xaxis().set_visible(False)
    #acq_EI.get_yaxis().set_visible(False)
    
    #acq_EI.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         #label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
            
    acq_EI.set_yticklabels([])
    acq_EI.set_xticklabels([])

    acq_EI.set_xlim((np.min(x_original)-0.05, np.max(x_original)+0.05))
    acq_EI.set_ylim((np.min(utility), 1.2*np.max(utility)))

    acq_EI.set_ylabel('EI', fontdict={'size':16})
    #acq_EI.set_xlabel('x', fontdict={'size':16})

    
 
	
    #axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #acq_EI.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    
    xstars=[]
    ystars=[]

    # TS1   
    # finding the xt of Thompson Sampling
    numXtar=100
    for ii in range(numXtar):
        mu_acq={}
        mu_acq['name']='thompson'
        mu_acq['dim']=bo.dim
        mu_acq['scalebounds']=bo.scalebounds    
        acq_mu=AcquisitionFunction(mu_acq)
        xt_TS = acq_max(ac=acq_mu.acq_kind,gp=bo.gp,bounds=bo.scalebounds ,opt_toolbox='scipy')
        xstars.append(xt_TS)
        
        yt_TS=acq_mu.acq_kind(xt_TS,bo.gp,y_max=np.max(bo.Y))
        if yt_TS>mu_max:
            ystars.append(yt_TS)

    if not ystars:
        ystars.append([mu_max])
        
    temp=np.asarray(xstars)
    xt_suggestion_original=temp*bo.max_min_gap+bo.bounds[:,0]
        
    # MRS     
    acq_func={}
    acq_func['name']='mrs'
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds
    acq_func['xstars']=xstars

    myacq=AcquisitionFunction(acq_func)
	
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp, np.max(bo.Y))
    
    #temp=np.asarray(myacq.object.xstars)
    #xt_suggestion_original=temp*bo.max_min_gap+bo.bounds[:,0]
    mymean=np.percentile([np.min(utility),np.max(utility)],20)

    acq_MRS.plot(x_original, utility, label='Utility Function', color='purple')
    acq_MRS.plot(xt_suggestion_original, [mymean]*xt_suggestion_original.shape[0], '*', markersize=12, 
         label=u'Next Best Guess', markerfacecolor='yellow', markeredgecolor='k', markeredgewidth=1)
        
    acq_MRS.plot(x_original[np.argmax(utility)], np.max(utility), 's', markersize=10, 
             label=u'Next Best Guess', markerfacecolor='red', markeredgecolor='k', markeredgewidth=1)

    #acq_MRS.plot(xt_suggestion_original, [np.max(utility)]*xt_suggestion_original.shape[0], 's', markersize=15, 
             #label=u'Next Best Guess', markerfacecolor='yellow', markeredgecolor='k', markeredgewidth=1)
    
    max_point=np.max(utility)
    acq_MRS.get_xaxis().set_visible(False)
    
    acq_MRS.set_yticklabels([])
    acq_MRS.set_xticklabels([])
    
    #acq_MRS.get_yaxis().set_visible(False)
    
    #acq_MRS.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         #label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
             
    acq_MRS.set_xlim((np.min(x_original)-0.05, np.max(x_original)+0.05))
    acq_MRS.set_ylim((np.min(utility), 1.2*np.max(utility)))
    acq_MRS.set_ylabel('MRS', fontdict={'size':16})
    #acq_MRS.set_xlabel('x', fontdict={'size':16})

    
	
    # MES     
    acq_func={}
    acq_func['name']='mes'
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds
    acq_func['xstars']=xstars
    acq_func['ystars']=ystars

    myacq=AcquisitionFunction(acq_func)
	
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp, np.max(bo.Y))
    
    #temp=np.asarray(myacq.object.xstars)
    #xt_suggestion_original=temp*bo.max_min_gap+bo.bounds[:,0]

    acq_MES.plot(x_original, utility, label='Utility Function', color='purple')
    acq_MES.plot(x_original[np.argmax(utility)], np.max(utility), 's', markersize=10, 
             label=u'Next Best Guess', markerfacecolor='red', markeredgecolor='k', markeredgewidth=1)

    #acq_MRS.plot(xt_suggestion_original, [np.max(utility)]*xt_suggestion_original.shape[0], 's', markersize=15, 
             #label=u'Next Best Guess', markerfacecolor='yellow', markeredgecolor='k', markeredgewidth=1)
    
    max_point=np.max(utility)
    acq_MES.get_xaxis().set_visible(False)
    
    acq_MES.set_yticklabels([])
    acq_MES.set_xticklabels([])
    
    #acq_MES.get_yaxis().set_visible(False)
    
    #acq_MRS.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         #label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
             
    acq_MES.set_xlim((np.min(x_original)-0.05, np.max(x_original)+0.05))
    acq_MES.set_ylim((np.min(utility), 1.2*np.max(utility)))

    acq_MES.set_ylabel('MES', fontdict={'size':16})
    
    #acq_MES.set_xlabel('x', fontdict={'size':16})

    
    # PES
    acq_func={}
    acq_func['name']='pes'
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds
    acq_func['xstars']=xstars

    myacq=AcquisitionFunction(acq_func)
	
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp, np.max(bo.Y))
    
    #temp=np.asarray(myacq.object.xstars)
    #xt_suggestion_original=temp*bo.max_min_gap+bo.bounds[:,0]
    mymean=np.percentile([np.min(utility),np.max(utility)],20)

    acq_PES.plot(x_original, utility, label='Utility Function', color='purple')
    
    acq_PES.plot(xt_suggestion_original, [mymean]*xt_suggestion_original.shape[0], '*', markersize=12, 
         label=u'Next Best Guess', markerfacecolor='yellow', markeredgecolor='k', markeredgewidth=1)
        
    acq_PES.plot(x_original[np.argmax(utility)], np.max(utility), 's', markersize=10, 
             label=u'Selected point $x_t$', markerfacecolor='red', markeredgecolor='k', markeredgewidth=1)

    max_point=np.max(utility)
    acq_PES.get_xaxis().set_visible(False)
    
    acq_PES.set_yticklabels([])
    acq_PES.set_xticklabels([])
    
    #acq_PES.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         #label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
             
    acq_PES.set_xlim((np.min(x_original)-0.05, np.max(x_original)+0.05))
    acq_PES.set_ylim((np.min(utility), 1.2*np.max(utility)))

    acq_PES.set_ylabel('PES', fontdict={'size':16})
    acq_PES.set_xlabel('x', fontdict={'size':16})
    
    #acq_PES.get_yaxis().set_visible(False)

    ### VRS
    acq_func={}
    acq_func['name']='vrs_of_ts'
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds
    acq_func['xstars']=xstars

    myacq=AcquisitionFunction(acq_func)
	
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp, np.max(bo.Y))
    
    #mytest=np.vstack((x.reshape(-1,1),bo.gp.X))
    #utility_existing_X = myacq.acq_kind(mytest, bo.gp, np.max(bo.Y))


    #utility=0-utility
    temp=np.asarray(myacq.object.xstars)
    xt_suggestion_original=temp*bo.max_min_gap+bo.bounds[:,0]
    
    acq_Consensus.plot(x_original, utility, label=r'$\alpha(x)$', color='purple')
    
    #acq_Consensus.plot(x_original, [np.asscalar(myacq.object.average_predvar)]*len(x_original), label=r'threshold', color='black')
    #print np.asscalar(myacq.object.average_predvar)
    #print np.min(utility)

    mymean=np.percentile([np.min(utility),np.max(utility)],20)
    acq_Consensus.plot(xt_suggestion_original, [mymean]*xt_suggestion_original.shape[0], '*', markersize=12, 
             label='$x^*$ samples', markerfacecolor='yellow', markeredgecolor='k', markeredgewidth=1)
   
    max_point=np.max(utility)
    
    acq_Consensus.plot(x_original[np.argmax(utility)], np.max(utility), 's', markersize=10, 
             label=u'Selected point $x_t$', markerfacecolor='red', markeredgecolor='k', markeredgewidth=1)
    
    #acq_Consensus.get_yaxis().set_visible(False)
        
    #acq_TS.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         #label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
     


    acq_Consensus.set_xlim((np.min(x_original)-0.05, np.max(x_original)+0.05))
    acq_Consensus.set_ylim((np.min(utility), 1.2*np.max(utility)))
    
    acq_Consensus.set_yticklabels([])
    

    #acq_TS.set_ylim((np.min(utility)*0.9, np.max(utility)*1.1))
    acq_Consensus.set_ylabel('PVRS', fontdict={'size':16})
    acq_Consensus.set_xlabel('x', fontdict={'size':16})

    acq_Consensus.legend(loc='center left', bbox_to_anchor=(0.01, -1.1),prop={'size':16},ncol=3)
    
    #acq_ES.get_xaxis().set_visible(False)


    # ES     
    acq_func={}
    acq_func['name']='es'
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds
    acq_func['xstars']=xstars


    myacq=AcquisitionFunction(acq_func)
	
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp, np.max(bo.Y))
    
    #temp=np.asarray(myacq.object.xstars)
    #xt_suggestion_original=temp*bo.max_min_gap+bo.bounds[:,0]
    mymean=np.percentile([np.min(utility),np.max(utility)],20)

    acq_ES.plot(x_original, utility, label='Utility Function', color='purple')
    acq_ES.plot(xt_suggestion_original, [mymean]*xt_suggestion_original.shape[0], '*', markersize=12, 
             label=u'Next Best Guess', markerfacecolor='yellow', markeredgecolor='k', markeredgewidth=1)
        
    acq_ES.plot(x_original[np.argmax(utility)], np.max(utility), 's', markersize=10, 
             label=u'Next Best Guess', markerfacecolor='red', markeredgecolor='k', markeredgewidth=1)


    
    #max_point=np.max(utility)
    
    #acq_ES.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         #label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
    #acq_ES.get_yaxis().set_visible(False)
    acq_ES.get_xaxis().set_visible(False)
    
    acq_ES.set_yticklabels([])
    acq_ES.set_xticklabels([])
    

    acq_ES.set_xlim((np.min(x_original)-0.05, np.max(x_original)+0.05))
    acq_ES.set_ylim((np.min(utility), 1.2*np.max(utility)))

    acq_ES.set_ylabel('ES', fontdict={'size':16})
    #acq_ES.set_xlabel('x', fontdict={'size':16})
    
    
    strFileName="{:d}_GP_acquisition_functions_vrs.eps".format(counter)
    fig.savefig(strFileName, bbox_inches='tight')
    #axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #acq_TS.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)

	
def plot_bo_1d(bo):
    func=bo.f
    #x_original = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 1000)
    x_original=x*bo.max_min_gap+bo.bounds[:,0]

    y_original = func(x_original)
    #y = func(x)
    #y_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)

    fig=plt.figure(figsize=(8, 5))
    fig.suptitle('Gaussian Process and Utility Function After {} Points'.format(len(bo.X)), fontdict={'size':18})
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])
    
    mu, sigma = bo.posterior(x)
    #mu_original=mu*(np.max(y_original)-np.min(y_original))+np.mean(y_original)
    mu_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)
    sigma_original=sigma*np.std(bo.Y_original)+np.mean(bo.Y_original)**2
    
    axis.plot(x_original, y_original, linewidth=3, label='Real Function')
    axis.plot(bo.X_original.flatten(), bo.Y_original, 'D', markersize=8, label=u'Observations', color='r')
    axis.plot(x_original, mu_original, '--', color='k', label='GP mean')
    
    #samples*bo.max_min_gap+bo.bounds[:,0]
    
    temp_xaxis=np.concatenate([x_original, x_original[::-1]])
    #temp_xaxis=temp*bo.max_min_gap+bo.bounds[:,0]
    
    temp_yaxis_original=np.concatenate([mu_original - 1.9600 * sigma_original, (mu_original + 1.9600 * sigma_original)[::-1]])
    temp_yaxis=np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]])
    temp_yaxis_original2=temp_yaxis*np.std(bo.Y_original)+np.mean(bo.Y_original)
    axis.fill(temp_xaxis, temp_yaxis_original2,alpha=.6, fc='c', ec='None', label='95% CI')
    
    
    axis.set_xlim((np.min(x_original), np.max(x_original)))
    #axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size':16})
    axis.set_xlabel('x', fontdict={'size':16})

    utility = bo.acq_func.acq_kind(x.reshape((-1, 1)), bo.gp, np.max(bo.Y))
    acq.plot(x_original, utility, label='Utility Function', color='purple')
    acq.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
             
    # check batch BO     
    try:
        nSelectedPoints=np.int(bo.NumPoints[-1])
    except:
        nSelectedPoints=1
    max_point=np.max(utility)
    
    acq.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
             
    acq.set_xlim((np.min(x_original), np.max(x_original)))
    #acq.set_ylim((0, np.max(utility) + 0.5))
    #acq.set_ylim((np.min(utility), 1.1*np.max(utility)))
    acq.set_ylabel('Acq', fontdict={'size':16})
    acq.set_xlabel('x', fontdict={'size':16})
    
    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)


def plot_bo_1d_variance(bo):
    
    global counter
    counter=counter+1
    
    func=bo.f
    #x_original = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 1000)
    x_original=x*bo.max_min_gap+bo.bounds[:,0]

    y_original = func(x_original)
    #y = func(x)
    #y_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)
    
    #fig=plt.figure(figsize=(8, 5))
    fig, ax1 = plt.subplots(figsize=(8.5, 4))

    mu, sigma = bo.posterior(x)
    mu_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)
    sigma_original=sigma*(np.max(bo.Y_original)-np.min(bo.Y_original))

    utility = bo.acq_func.acq_kind(x.reshape((-1, 1)), bo.gp, np.max(bo.Y))


    def distance_function(x,X):            
        Euc_dist=euclidean_distances(x,X)
          
        dist=Euc_dist.min(axis=1)
        return dist
        
    utility_distance=distance_function(x.reshape((-1, 1)),bo.X)
    idxMaxVar=np.argmax(utility)
    #idxMaxVar=[idx for idx,val in enumerate(utility) if val>=0.995]
    ax1.plot(x_original, utility, label='GP $\sigma(x)$', color='purple')  

    
    ax1.scatter(x_original[idxMaxVar], utility[idxMaxVar], marker='s',label='x=argmax $\sigma(x)$', color='blue',linewidth=2)            
          
    #ax1.scatter(x_original[idxMaxVar], utility[idxMaxVar], label='$||x-[x]||$', color='blue',linewidth=2)            

    ax1.plot(bo.X_original.flatten(), [0]*len(bo.X_original.flatten()), 'D', markersize=10, label=u'Observations', color='r')


    idxMaxDE=np.argmax(utility_distance)
    ax2 = ax1.twinx()
    ax2.plot(x_original, utility_distance, label='$d(x)=||x-[x]||^2$', color='black') 
    ax2.plot(x_original[idxMaxDE], utility_distance[idxMaxDE], 'o',label='x=argmax d(x)', color='black',markersize=10)            
           
    ax2.set_ylim((0, 0.45))


         
    ax1.set_xlim((np.min(x_original)-0.01, 0.01+np.max(x_original)))
    ax1.set_ylim((-0.02, np.max(utility) + 0.05))
    #acq.set_ylim((np.min(utility), 1.1*np.max(utility)))
    ax1.set_ylabel(r'$\sigma(x)$', fontdict={'size':18})
    ax2.set_ylabel('d(x)', fontdict={'size':18})

    ax1.set_xlabel('x', fontdict={'size':18})
    
    #axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #ax1.legend(loc=2, bbox_to_anchor=(1.1, 1), borderaxespad=0.,fontsize=14)
    #ax2.legend(loc=2, bbox_to_anchor=(1.1, 0.3), borderaxespad=0.,fontsize=14)

    plt.title('Exploration by GP variance vs distance',fontsize=22)
    ax1.legend(loc=3, bbox_to_anchor=(0.05,-0.32,1, -0.32), borderaxespad=0.,fontsize=14,ncol=4)
    ax2.legend(loc=3, bbox_to_anchor=(0.05,-0.46,1, -0.46), borderaxespad=0.,fontsize=14,ncol=2)

    #plt.legend(fontsize=14)
    strFolder="P:\\03.Research\\05.BayesianOptimization\\PradaBayesianOptimization\\demo_geometric"

    strFileName="{:d}_var_DE.eps".format(counter)
    strPath=os.path.join(strFolder,strFileName)
    fig.savefig(strPath, bbox_inches='tight')

def plot_acq_bo_2d_vrs(bo):
    
    global counter
    counter=counter+1
    
    func=bo.f
    #x_original = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x1 = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 50)
    x2 = np.linspace(bo.scalebounds[1,0], bo.scalebounds[1,1], 50)
    x1g,x2g=np.meshgrid(x1,x2)
    
    X=np.c_[x1g.flatten(), x2g.flatten()]
    
    x1_ori = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 50)
    x2_ori = np.linspace(bo.bounds[1,0], bo.bounds[1,1], 50)    
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]

    fig=plt.figure(figsize=(14, 20))
    #fig.suptitle('Gaussian Process and Utility Function After {} Points'.format(len(bo.X)), fontdict={'size':18})
    
    #gs = gridspec.GridSpec(7, 1, height_ratios=[1,1,1,1,1,1,1]) 
    
    nRows=6
    axis_mean2d = fig.add_subplot(nRows, 2, 1)
    axis_variance2d = fig.add_subplot(nRows, 2, 2)
    acq_UCB = fig.add_subplot(nRows, 2, 3)
    #acq_EI =fig.add_subplot(nRows, 2,4)
    #acq_POI = plt.subplot(gs[3])
    

    acq_ES = fig.add_subplot(nRows, 2, 4)
    acq_PES = fig.add_subplot(nRows, 2, 5)
    acq_MRS = fig.add_subplot(nRows, 2, 6)
    #acq_ydist = fig.add_subplot(nRows, 2, 8)

    acq_VRS = fig.add_subplot(nRows, 2, 7)
    acq_Batch_VRS_B_2 = fig.add_subplot(nRows, 2, 8)
    acq_Batch_VRS_B_3 = fig.add_subplot(nRows, 2, 9)
    acq_Batch_VRS_B_4 = fig.add_subplot(nRows, 2, 10)
        
    mu, sigma = bo.posterior(X)
    #mu_original=mu*(np.max(y_original)-np.min(y_original))+np.mean(y_original)
    #mu_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)
    #sigma_original=sigma*np.std(bo.Y_original)+np.mean(bo.Y_original)**2
    
    # get maximum of mu function
    mu_max=mu.max()
    
    # mean
    CS=axis_mean2d.contourf(x1g_ori,x2g_ori,mu.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2 = plt.contour(CS, levels=CS.levels[::2],colors='r',origin='lower',hold='on')
    
    axis_mean2d.scatter(bo.X_original[:,0],bo.X_original[:,1], label=u'Observations', color='g')    
    axis_mean2d.set_title('Gaussian Process Mean $\mu(x)$',fontsize=16)
    axis_mean2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    axis_mean2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS, ax=axis_mean2d, shrink=0.9)
    axis_mean2d.get_xaxis().set_visible(False)
    axis_mean2d.get_yaxis().set_visible(False)


    # variance
    CS=axis_variance2d.contourf(x1g_ori,x2g_ori,sigma.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2 = plt.contour(CS, levels=CS.levels[::2],colors='r',origin='lower',hold='on')
    
    axis_variance2d.scatter(bo.X_original[:,0],bo.X_original[:,1], label=u'Observations', color='g')    
    axis_variance2d.set_title('Gaussian Process Variance $\sigma(x)$',fontsize=16)
    axis_variance2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    axis_variance2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS, ax=axis_variance2d, shrink=0.9)
    axis_variance2d.get_xaxis().set_visible(False)
    axis_variance2d.get_yaxis().set_visible(False)
    
    # UCB 
    acq_func={}
    acq_func['name']='ucb'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
    utility = myacq.acq_kind(X, bo.gp)
    CS_acq=acq_UCB.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    acq_UCB.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data') 
    acq_UCB.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='s',color='r',s=100,label='Peak')

    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    #acq_UCB.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')
    
    xt_UCB=X[idxBest,:]
    
    acq_UCB.set_title('UCB',fontsize=16)
    acq_UCB.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq_UCB.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS_acq, ax=acq_UCB, shrink=0.9)
    acq_UCB.get_xaxis().set_visible(False)
    acq_UCB.get_yaxis().set_visible(False)

    """
    # EI 
    acq_func={}
    acq_func['name']='ei'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
    utility = myacq.acq_kind(X, bo.gp)
    CS_acq=acq_EI.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    acq_EI.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  

    acq_EI.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='*',color='r',s=300,label='Peak')
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    #acq_EI.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')

    xt_EI=X[idxBest,:]

    acq_EI.set_title('EI',fontsize=16)
    acq_EI.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq_EI.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS_acq, ax=acq_EI, shrink=0.9)
"""

    # ==================================================================================	
    # finding the xt of Thompson Sampling then use for PES, ES and VRS
    y_max=np.max(bo.Y)
    xstars=[]
    y_stars=[]
    xstars_VRS=[]
    numXtar=25*bo.dim
    for ii in range(numXtar):
        mu_acq={}
        mu_acq['name']='thompson'
        mu_acq['dim']=bo.dim
        mu_acq['scalebounds']=bo.scalebounds    
        acq_mu=AcquisitionFunction(mu_acq)
        xt_TS = acq_max(ac=acq_mu.acq_kind,gp=bo.gp,bounds=bo.scalebounds ,opt_toolbox='scipy')
        
        y_xt_TS=acq_mu.acq_kind(xt_TS,bo.gp)
        #if y_xt_TS>mu_max:
        y_stars.append(y_xt_TS)

        xstars.append(xt_TS)
        #if y_xt_TS>=y_max:
        xstars_VRS.append(xt_TS)
            
            
    # MRS         
    acq_func={}
    acq_func['name']='mes'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['scalebounds']=bo.scalebounds
    acq_func['ystars']=y_stars

    myacq=AcquisitionFunction(acq_func)
    utility = myacq.acq_kind(X, bo.gp)
    CS_acq=acq_MRS.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    acq_MRS.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    #acq_MRS.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')

    xt_suggestion_original=xstars*bo.max_min_gap+bo.bounds[:,0]
    acq_MRS.scatter(xt_suggestion_original[:,0],xt_suggestion_original[:,1],marker='*',color='y',s=150,label='xstars')

    acq_MRS.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='s',color='r',s=100,label='Peak')

    acq_MRS.set_title('MES',fontsize=16)
    acq_MRS.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq_MRS.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS_acq, ax=acq_MRS, shrink=0.9)
    acq_MRS.get_xaxis().set_visible(False)
    acq_MRS.get_yaxis().set_visible(False)
    """
    # plot distribution of y_star
    mu_ydist, std_ydist = norm_dist.fit(y_stars)
    # Plot the histogram.
    acq_ydist.hist(y_stars,bins=20,normed=True,alpha=.6,color='g',label=ur'Histogram of $y^*$')
    # Plot the PDF.
    x = np.linspace(np.min(y_stars), np.max(y_stars), 100)
    p = norm_dist.pdf(x, mu_ydist, std_ydist)
    acq_ydist.plot(x,p,'k', linewidth=2,label='Gaussian curve')
    acq_ydist.legend()
    acq_ydist.set_title(ur'Distribution of $y^*$',fontsize=16)
    """
    
    # PES
    acq_func={}
    acq_func['name']='pes'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['scalebounds']=bo.scalebounds
    #acq_func['xstars']=xstars
    acq_func['xstars']=xstars_VRS
    myacq=AcquisitionFunction(acq_func)
    utility = myacq.acq_kind(X, bo.gp)
    CS_acq=acq_PES.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    
    acq_PES.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    #acq_PES.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')

    temp=np.asarray(myacq.object.x_stars) 
    temp=temp.reshape(-1,2)
    # convert from scale data points to original data points
    xt_suggestion_original=temp*bo.max_min_gap+bo.bounds[:,0]
    
    acq_PES.scatter(xt_suggestion_original[:,0],xt_suggestion_original[:,1],marker='*',color='y',s=150,label='xstars')
    acq_PES.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='s',color='r',s=100,label='Peak')
    xt_PES=X[idxBest,:]

    acq_PES.set_title('PES',fontsize=16)
    acq_PES.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq_PES.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS_acq, ax=acq_PES, shrink=0.9)
    acq_PES.get_xaxis().set_visible(False)
    acq_PES.get_yaxis().set_visible(False)
    
    # ES     
    acq_func={}
    acq_func['name']='es'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['scalebounds']=bo.scalebounds
    acq_func['xstars']=xstars

    myacq=AcquisitionFunction(acq_func)
    utility = myacq.acq_kind(X, bo.gp)
    CS_acq=acq_ES.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    acq_ES.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    #acq_ES.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')
    temp=np.asarray(myacq.object.x_stars) 
    #temp=temp.reshape(-1,2)
    # convert from scale data points to original data points
    xt_suggestion_original=temp*bo.max_min_gap+bo.bounds[:,0]
    acq_ES.scatter(xt_suggestion_original[:,0],xt_suggestion_original[:,1],marker='*',color='y',s=150,label='xstars')

    acq_ES.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='s',color='r',s=100,label='Peak')

    #xt_ES=X[idxBest,:]


    acq_ES.set_title('ES',fontsize=16)
    acq_ES.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq_ES.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS_acq, ax=acq_ES, shrink=0.9)
    acq_ES.get_xaxis().set_visible(False)
    acq_ES.get_yaxis().set_visible(False)
    #xstars.append(xt_UCB)
    #xstars.append(xt_EI)
    #xstars.append(xt_ES)
    #xstars.append(xt_PES)
    
    # Variance Reduction Search
    acq_func={}
    acq_func['name']='pvrs'
    acq_func['kappa']=2
    acq_func['n_xstars_x_dim']=50
    acq_func['dim']=2
    acq_func['scalebounds']=bo.scalebounds
    acq_func['xstars']=xstars_VRS

    myacq=AcquisitionFunction(acq_func)

    utility = myacq.acq_kind(X, bo.gp)
    CS_acq=acq_VRS.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    acq_VRS.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  
    temp=np.asarray(myacq.object.xstars) 
    # convert from scale data points to original data points
    xt_suggestion_original=temp*bo.max_min_gap+bo.bounds[:,0]
    
    acq_VRS.scatter(xt_suggestion_original[:,0],xt_suggestion_original[:,1],marker='*',color='y',s=150,label='xstars')
    acq_VRS.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='s',color='r',s=100,label='Peak')

    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    #acq_ES.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')

    acq_VRS.set_title('PVRS',fontsize=16)
    acq_VRS.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq_VRS.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS_acq, ax=acq_VRS, shrink=0.9)
    acq_VRS.get_xaxis().set_visible(False)
    acq_VRS.get_yaxis().set_visible(False)
    
    
    # Batch Variance Reduction Search B=2
    
    acq_Batch_VRS_B_2.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  
    temp=np.asarray(myacq.object.xstars) 
    # convert from scale data points to original data points
    xt_suggestion_original=temp*bo.max_min_gap+bo.bounds[:,0]
    
    acq_Batch_VRS_B_2.scatter(xt_suggestion_original[:,0],xt_suggestion_original[:,1],marker='*',color='y',s=150,label='xstars')
    #acq_VRS.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='*',color='r',s=300,label='Peak')

    acq_func={}
    acq_func['name']='ucb'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['xstars']=xstars_VRS


    acq_params={}
    acq_params['acq_func']=acq_func
    acq_params['optimize_gp']=1
    acq_params['n_xstars']=100

    
    func_params={}
    func_params['bounds']=bo.bounds
    func_params['f']=func

    gp_params = {'lengthscale':0.1*2,'noise_delta':0.00000001}
    bo2=BatchPVRS(gp_params,func_params, acq_params)
    bo2.init_with_data(bo.X_original,bo.Y_original)
       
    #new_X=bo2.maximize_batch_sequential_greedy_PVRS(gp_params,B=3)
      
    new_X,temp=bo2.maximize_batch_PVRS_iterative_greedy(gp_params,B=2)
    new_X_original=new_X*bo.max_min_gap+bo.bounds[:,0]


    
    acq_Batch_VRS_B_2.scatter(new_X_original[:,0],new_X_original[:,1],marker='s',color='r',s=100,label='Selected')
    #acq_ES.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')

    acq_Batch_VRS_B_2.set_title('Batch PVRS B=2',fontsize=16)
    acq_Batch_VRS_B_2.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq_Batch_VRS_B_2.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS_acq, ax=acq_Batch_VRS_B_2, shrink=0.9)
    acq_Batch_VRS_B_2.get_xaxis().set_visible(False)
    acq_Batch_VRS_B_2.get_yaxis().set_visible(False)
    
    
    # Batch Variance Reduction Search B=3
    
    acq_Batch_VRS_B_3.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Existing data X')  
    temp=np.asarray(myacq.object.xstars) 
    # convert from scale data points to original data points
    xt_suggestion_original=temp*bo.max_min_gap+bo.bounds[:,0]
    
    acq_Batch_VRS_B_3.scatter(xt_suggestion_original[:,0],xt_suggestion_original[:,1],marker='*',color='y',s=150,label=r'$x^*$ samples')
    #acq_VRS.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='*',color='r',s=300,label='Peak')

    acq_func={}
    acq_func['name']='ucb'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['xstars']=xstars_VRS


    acq_params={}
    acq_params['acq_func']=acq_func
    acq_params['optimize_gp']=1
    acq_params['n_xstars']=100

    
    func_params={}
    func_params['bounds']=bo.bounds
    func_params['f']=func

    gp_params = {'lengthscale':0.1*2,'noise_delta':0.00000001}
    bo2=BatchPVRS(gp_params,func_params, acq_params)
    bo2.init_with_data(bo.X_original,bo.Y_original)
       
    #new_X=bo2.maximize_batch_sequential_greedy_PVRS(gp_params,B=3)
      
    new_X,temp=bo2.maximize_batch_PVRS_iterative_greedy(gp_params,B=3)
    new_X_original=new_X*bo.max_min_gap+bo.bounds[:,0]


    
    acq_Batch_VRS_B_3.scatter(new_X_original[:,0],new_X_original[:,1],marker='s',color='r',s=100,label='Selected point $x_t$')
    #acq_ES.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')

    acq_Batch_VRS_B_3.set_title('Batch PVRS B=3',fontsize=16)
    acq_Batch_VRS_B_3.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq_Batch_VRS_B_3.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS_acq, ax=acq_Batch_VRS_B_3, shrink=0.9)
    acq_Batch_VRS_B_3.get_xaxis().set_visible(False)
    acq_Batch_VRS_B_3.get_yaxis().set_visible(False)
    
    
    acq_Batch_VRS_B_3.legend(loc='center left', bbox_to_anchor=(0.01, -0.2),prop={'size':20},ncol=3)




    
    # Batch Variance Reduction Search B=4
    
    acq_Batch_VRS_B_4.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  
    temp=np.asarray(myacq.object.xstars) 
    # convert from scale data points to original data points
    xt_suggestion_original=temp*bo.max_min_gap+bo.bounds[:,0]
    
    acq_Batch_VRS_B_4.scatter(xt_suggestion_original[:,0],xt_suggestion_original[:,1],marker='*',color='y',s=150,label='xstars')
    #acq_VRS.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='*',color='r',s=300,label='Peak')

    acq_func={}
    acq_func['name']='ucb'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['xstars']=xstars_VRS


    acq_params={}
    acq_params['acq_func']=acq_func
    acq_params['optimize_gp']=1
    acq_params['n_xstars']=100

    
    func_params={}
    func_params['bounds']=bo.bounds
    func_params['f']=func

    gp_params = {'lengthscale':0.1*2,'noise_delta':0.00000001}
    bo2=BatchPVRS(gp_params,func_params, acq_params)
    bo2.init_with_data(bo.X_original,bo.Y_original)
       
    #new_X=bo2.maximize_batch_sequential_greedy_PVRS(gp_params,B=3)
      
    new_X,temp=bo2.maximize_batch_PVRS_iterative_greedy(gp_params,B=4)
    new_X_original=new_X*bo.max_min_gap+bo.bounds[:,0]


    
    acq_Batch_VRS_B_4.scatter(new_X_original[:,0],new_X_original[:,1],marker='s',color='r',s=100,label='Selected')
    #acq_ES.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')

    acq_Batch_VRS_B_4.set_title('Batch PVRS B=4',fontsize=16)
    acq_Batch_VRS_B_4.set_xlim(bo.bounds[0,0]-0.1, bo.bounds[0,1]+0.1)
    acq_Batch_VRS_B_4.set_ylim(bo.bounds[1,0]-0.1, bo.bounds[1,1]+0.1)
    fig.colorbar(CS_acq, ax=acq_Batch_VRS_B_4, shrink=0.9)
    acq_Batch_VRS_B_4.get_xaxis().set_visible(False)
    acq_Batch_VRS_B_4.get_yaxis().set_visible(False)
    
    
    
    
    strFileName="{:d}_GP2d_acquisition_functions_vrs.eps".format(counter)
    fig.savefig(strFileName, bbox_inches='tight')
    #axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #acq_TS.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    

def plot_acq_bo_2d_vrs_3x2(bo):
    
    global counter
    counter=counter+1
    
    func=bo.f
    #x_original = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x1 = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 50)
    x2 = np.linspace(bo.scalebounds[1,0], bo.scalebounds[1,1], 50)
    x1g,x2g=np.meshgrid(x1,x2)
    
    X=np.c_[x1g.flatten(), x2g.flatten()]
    
    x1_ori = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 50)
    x2_ori = np.linspace(bo.bounds[1,0], bo.bounds[1,1], 50)    
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]

    fig=plt.figure(figsize=(14, 16))
    #fig.suptitle('Gaussian Process and Utility Function After {} Points'.format(len(bo.X)), fontdict={'size':18})
    
    #gs = gridspec.GridSpec(7, 1, height_ratios=[1,1,1,1,1,1,1]) 
    
    nRows=4
    axis_mean2d = fig.add_subplot(nRows, 2, 1)
    axis_variance2d = fig.add_subplot(nRows, 2, 2)
    acq_UCB = fig.add_subplot(nRows, 2, 3)
    acq_ES = fig.add_subplot(nRows, 2, 4)
    acq_PES = fig.add_subplot(nRows, 2, 5)


    acq_VRS = fig.add_subplot(nRows, 2, 6)
    acq_Batch_VRS_B_2 = fig.add_subplot(nRows, 2, 7)
    acq_Batch_VRS_B_3 = fig.add_subplot(nRows, 2, 8)
    #acq_Batch_VRS_B_4 = fig.add_subplot(nRows, 2, 10)
        
    mu, sigma = bo.posterior(X)
    #mu_original=mu*(np.max(y_original)-np.min(y_original))+np.mean(y_original)
    #mu_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)
    #sigma_original=sigma*np.std(bo.Y_original)+np.mean(bo.Y_original)**2
    
    # get maximum of mu function
    mu_max=mu.max()
    
    # mean
    CS=axis_mean2d.contourf(x1g_ori,x2g_ori,mu.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2 = plt.contour(CS, levels=CS.levels[::2],colors='r',origin='lower',hold='on')
    
    axis_mean2d.scatter(bo.X_original[:,0],bo.X_original[:,1], label=u'Observations', color='g')    
    axis_mean2d.set_title('Gaussian Process Mean $\mu(x)$',fontsize=16)
    axis_mean2d.set_xlim(bo.bounds[0,0]-0.1, bo.bounds[0,1]+0.1)
    axis_mean2d.set_ylim(bo.bounds[1,0]-0.1, bo.bounds[1,1]+0.1)
    fig.colorbar(CS, ax=axis_mean2d, shrink=0.9)
    axis_mean2d.get_xaxis().set_visible(False)
    axis_mean2d.get_yaxis().set_visible(False)


    # variance
    CS=axis_variance2d.contourf(x1g_ori,x2g_ori,sigma.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2 = plt.contour(CS, levels=CS.levels[::2],colors='r',origin='lower',hold='on')
    
    axis_variance2d.scatter(bo.X_original[:,0],bo.X_original[:,1], label=u'Observations', color='g')    
    axis_variance2d.set_title('Gaussian Process Variance $\sigma(x)$',fontsize=16)
    axis_variance2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    axis_variance2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS, ax=axis_variance2d, shrink=0.9)
    axis_variance2d.get_xaxis().set_visible(False)
    axis_variance2d.get_yaxis().set_visible(False)
    
    # UCB 
    acq_func={}
    acq_func['name']='ucb'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
    utility = myacq.acq_kind(X, bo.gp)
    CS_acq=acq_UCB.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    acq_UCB.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data') 
    acq_UCB.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='s',color='r',s=100,label='Peak')

    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    #acq_UCB.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')
    
    xt_UCB=X[idxBest,:]
    
    acq_UCB.set_title('UCB',fontsize=16)
    acq_UCB.set_xlim(bo.bounds[0,0]-0.1, bo.bounds[0,1]+0.1)
    acq_UCB.set_ylim(bo.bounds[1,0]-0.1, bo.bounds[1,1]+0.1)
    fig.colorbar(CS_acq, ax=acq_UCB, shrink=0.9)
    acq_UCB.get_xaxis().set_visible(False)
    acq_UCB.get_yaxis().set_visible(False)


    # ==================================================================================	
    # finding the xt of Thompson Sampling then use for PES, ES and VRS
    y_max=np.max(bo.Y)
    xstars=[]
    y_stars=[]
    xstars_VRS=[]
    numXtar=25*bo.dim
    for ii in range(numXtar):
        mu_acq={}
        mu_acq['name']='thompson'
        mu_acq['dim']=bo.dim
        mu_acq['scalebounds']=bo.scalebounds    
        acq_mu=AcquisitionFunction(mu_acq)
        xt_TS = acq_max(ac=acq_mu.acq_kind,gp=bo.gp,bounds=bo.scalebounds ,opt_toolbox='scipy')
        
        y_xt_TS=acq_mu.acq_kind(xt_TS,bo.gp)
        #if y_xt_TS>mu_max:
        y_stars.append(y_xt_TS)

        xstars.append(xt_TS)
        #if y_xt_TS>=y_max:
        xstars_VRS.append(xt_TS)
            
            

     # ES     
    acq_func={}
    acq_func['name']='es'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['scalebounds']=bo.scalebounds
    acq_func['xstars']=xstars

    myacq=AcquisitionFunction(acq_func)
    utility = myacq.acq_kind(X, bo.gp)
    CS_acq=acq_ES.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    acq_ES.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    #acq_ES.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')
    temp=np.asarray(myacq.object.x_stars) 
    #temp=temp.reshape(-1,2)
    # convert from scale data points to original data points
    xt_suggestion_original=temp*bo.max_min_gap+bo.bounds[:,0]
    acq_ES.scatter(xt_suggestion_original[:,0],xt_suggestion_original[:,1],marker='*',color='y',s=150,label='xstars')

    acq_ES.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='s',color='r',s=100,label='Peak')

    #xt_ES=X[idxBest,:]


    acq_ES.set_title('ES',fontsize=16)
    acq_ES.set_xlim(bo.bounds[0,0]-0.1, bo.bounds[0,1]+0.1)
    acq_ES.set_ylim(bo.bounds[1,0]-0.1, bo.bounds[1,1]+0.1)
    fig.colorbar(CS_acq, ax=acq_ES, shrink=0.9)
    acq_ES.get_xaxis().set_visible(False)
    acq_ES.get_yaxis().set_visible(False)
    #xstars.append(xt_UCB)
    #xstars.append(xt_EI)
    #xstars.append(xt_ES)
    #xstars.append(xt_PES)
    
    
    # PES
    acq_func={}
    acq_func['name']='pes'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['scalebounds']=bo.scalebounds
    #acq_func['xstars']=xstars
    acq_func['xstars']=xstars_VRS
    myacq=AcquisitionFunction(acq_func)
    utility = myacq.acq_kind(X, bo.gp)
    CS_acq=acq_PES.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    
    acq_PES.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    #acq_PES.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')

    temp=np.asarray(myacq.object.x_stars) 
    temp=temp.reshape(-1,2)
    # convert from scale data points to original data points
    xt_suggestion_original=temp*bo.max_min_gap+bo.bounds[:,0]
    
    acq_PES.scatter(xt_suggestion_original[:,0],xt_suggestion_original[:,1],marker='*',color='y',s=150,label='xstars')
    acq_PES.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='s',color='r',s=100,label='Peak')
    xt_PES=X[idxBest,:]

    acq_PES.set_title('PES',fontsize=16)
    acq_PES.set_xlim(bo.bounds[0,0]-0.1, bo.bounds[0,1]+0.1)
    acq_PES.set_ylim(bo.bounds[1,0]-0.1, bo.bounds[1,1]+0.1)
    fig.colorbar(CS_acq, ax=acq_PES, shrink=0.9)
    acq_PES.get_xaxis().set_visible(False)
    acq_PES.get_yaxis().set_visible(False)
    
    """
    # plot distribution of y_star
    mu_ydist, std_ydist = norm_dist.fit(y_stars)
    # Plot the histogram.
    acq_ydist.hist(y_stars,bins=20,normed=True,alpha=.6,color='g',label=ur'Histogram of $y^*$')
    # Plot the PDF.
    x = np.linspace(np.min(y_stars), np.max(y_stars), 100)
    p = norm_dist.pdf(x, mu_ydist, std_ydist)
    acq_ydist.plot(x,p,'k', linewidth=2,label='Gaussian curve')
    acq_ydist.legend()
    acq_ydist.set_title(ur'Distribution of $y^*$',fontsize=16)
    """
    
 
    # Variance Reduction Search
    acq_func={}
    acq_func['name']='vrs'
    acq_func['kappa']=2
    acq_func['n_xstars_x_dim']=50
    acq_func['dim']=2
    acq_func['scalebounds']=bo.scalebounds
    acq_func['xstars']=xstars_VRS

    myacq=AcquisitionFunction(acq_func)

    utility = myacq.acq_kind(X, bo.gp)
    CS_acq=acq_VRS.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    temp=np.asarray(myacq.object.xstars) 
    # convert from scale data points to original data points
    xt_suggestion_original=temp*bo.max_min_gap+bo.bounds[:,0]
    
    acq_VRS.scatter(xt_suggestion_original[:,0],xt_suggestion_original[:,1],marker='*',color='y',s=150,label='xstars')
    acq_VRS.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  

    acq_VRS.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='s',color='r',s=100,label='Peak')

    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    #acq_ES.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')

    acq_VRS.set_title('PVRS',fontsize=16)
    acq_VRS.set_xlim(bo.bounds[0,0]-0.1, bo.bounds[0,1]+0.1)
    acq_VRS.set_ylim(bo.bounds[1,0]-0.1, bo.bounds[1,1]+0.1)
    fig.colorbar(CS_acq, ax=acq_VRS, shrink=0.9)
    acq_VRS.get_xaxis().set_visible(False)
    acq_VRS.get_yaxis().set_visible(False)
    
    
    # Batch Variance Reduction Search B=2
    
    acq_Batch_VRS_B_2.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Existing data X')  
    temp=np.asarray(myacq.object.xstars) 
    # convert from scale data points to original data points
    xt_suggestion_original=temp*bo.max_min_gap+bo.bounds[:,0]
    
    acq_Batch_VRS_B_2.scatter(xt_suggestion_original[:,0],xt_suggestion_original[:,1],marker='*',color='y',s=150,label=r'$x^*$ samples')
    #acq_VRS.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='*',color='r',s=300,label='Peak')

    acq_func={}
    acq_func['name']='ucb'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['xstars']=xstars_VRS


    acq_params={}
    acq_params['acq_func']=acq_func
    acq_params['optimize_gp']=1
    acq_params['n_xstars']=100

    
    func_params={}
    func_params['bounds']=bo.bounds
    func_params['f']=func

    gp_params = {'lengthscale':0.1*2,'noise_delta':0.00000001}
    bo2=BatchPVRS(gp_params,func_params, acq_params)
    bo2.init_with_data(bo.X_original,bo.Y_original)
       
    #new_X=bo2.maximize_batch_sequential_greedy_PVRS(gp_params,B=3)
      
    new_X,temp=bo2.maximize_batch_PVRS_iterative_greedy(gp_params,B=2)
    new_X_original=new_X*bo.max_min_gap+bo.bounds[:,0]


    
    acq_Batch_VRS_B_2.scatter(new_X_original[:,0],new_X_original[:,1],marker='s',color='r',s=100,label='Selected point $x_t$')
    #acq_ES.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')

    acq_Batch_VRS_B_2.set_title('B-PVRS B=2',fontsize=16)
    acq_Batch_VRS_B_2.set_xlim(bo.bounds[0,0]-0.1, bo.bounds[0,1]+0.1)
    acq_Batch_VRS_B_2.set_ylim(bo.bounds[1,0]-0.1, bo.bounds[1,1]+0.1)
    fig.colorbar(CS_acq, ax=acq_Batch_VRS_B_2, shrink=0.9)
    acq_Batch_VRS_B_2.get_xaxis().set_visible(False)
    acq_Batch_VRS_B_2.get_yaxis().set_visible(False)
    
    
    # Batch Variance Reduction Search B=3
    
    acq_Batch_VRS_B_3.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Existing data X')  
    temp=np.asarray(myacq.object.xstars) 
    # convert from scale data points to original data points
    xt_suggestion_original=temp*bo.max_min_gap+bo.bounds[:,0]
    
    acq_Batch_VRS_B_3.scatter(xt_suggestion_original[:,0],xt_suggestion_original[:,1],marker='*',color='y',s=150,label=r'$x^*$ samples')
    #acq_VRS.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='*',color='r',s=300,label='Peak')

    acq_func={}
    acq_func['name']='ucb'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['xstars']=xstars_VRS


    acq_params={}
    acq_params['acq_func']=acq_func
    acq_params['optimize_gp']=1
    acq_params['n_xstars']=100

    
    func_params={}
    func_params['bounds']=bo.bounds
    func_params['f']=func

    gp_params = {'lengthscale':0.1*2,'noise_delta':0.00000001}
    bo2=BatchPVRS(gp_params,func_params, acq_params)
    bo2.init_with_data(bo.X_original,bo.Y_original)
       
    #new_X=bo2.maximize_batch_sequential_greedy_PVRS(gp_params,B=3)
      
    new_X,temp=bo2.maximize_batch_PVRS_iterative_greedy(gp_params,B=3)
    new_X_original=new_X*bo.max_min_gap+bo.bounds[:,0]


    
    acq_Batch_VRS_B_3.scatter(new_X_original[:,0],new_X_original[:,1],marker='s',color='r',s=100,label='Selected point $x_t$')
    #acq_ES.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')

    acq_Batch_VRS_B_3.set_title('B-PVRS B=3',fontsize=16)
    acq_Batch_VRS_B_3.set_xlim(bo.bounds[0,0]-0.1, bo.bounds[0,1]+0.1)
    acq_Batch_VRS_B_3.set_ylim(bo.bounds[1,0]-0.1, bo.bounds[1,1]+0.1)
    fig.colorbar(CS_acq, ax=acq_Batch_VRS_B_3, shrink=0.9)
    acq_Batch_VRS_B_3.get_xaxis().set_visible(False)
    acq_Batch_VRS_B_3.get_yaxis().set_visible(False)
    
    
    acq_Batch_VRS_B_2.legend(loc='center left', bbox_to_anchor=(0.1, -0.2),prop={'size':20},ncol=3)
    
    # Batch Variance Reduction Search B=4
    """
    acq_Batch_VRS_B_4.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  
    temp=np.asarray(myacq.object.xstars) 
    # convert from scale data points to original data points
    xt_suggestion_original=temp*bo.max_min_gap+bo.bounds[:,0]
    
    acq_Batch_VRS_B_4.scatter(xt_suggestion_original[:,0],xt_suggestion_original[:,1],marker='*',color='y',s=150,label='xstars')
    #acq_VRS.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='*',color='r',s=300,label='Peak')

    acq_func={}
    acq_func['name']='ucb'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['xstars']=xstars_VRS


    acq_params={}
    acq_params['acq_func']=acq_func
    acq_params['optimize_gp']=1
    acq_params['n_xstars']=100

    
    func_params={}
    func_params['bounds']=bo.bounds
    func_params['f']=func

    gp_params = {'lengthscale':0.1*2,'noise_delta':0.00000001}
    bo2=BatchPVRS(gp_params,func_params, acq_params)
    bo2.init_with_data(bo.X_original,bo.Y_original)
       
    #new_X=bo2.maximize_batch_sequential_greedy_PVRS(gp_params,B=3)
      
    new_X,temp=bo2.maximize_batch_PVRS_iterative_greedy(gp_params,B=4)
    new_X_original=new_X*bo.max_min_gap+bo.bounds[:,0]


    
    acq_Batch_VRS_B_4.scatter(new_X_original[:,0],new_X_original[:,1],marker='s',color='r',s=100,label='Selected')
    #acq_ES.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')

    acq_Batch_VRS_B_4.set_title('Batch PVRS B=4',fontsize=16)
    acq_Batch_VRS_B_4.set_xlim(bo.bounds[0,0]-0.1, bo.bounds[0,1]+0.1)
    acq_Batch_VRS_B_4.set_ylim(bo.bounds[1,0]-0.1, bo.bounds[1,1]+0.1)
    fig.colorbar(CS_acq, ax=acq_Batch_VRS_B_4, shrink=0.9)
    acq_Batch_VRS_B_4.get_xaxis().set_visible(False)
    acq_Batch_VRS_B_4.get_yaxis().set_visible(False)
    """
  
    strFileName="{:d}_GP2d_acquisition_functions_vrs.eps".format(counter)
    fig.savefig(strFileName, bbox_inches='tight')

    
def plot_acq_bo_2d_vrs_backup(bo):
    
    global counter
    counter=counter+1
    
    func=bo.f
    #x_original = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x1 = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 80)
    x2 = np.linspace(bo.scalebounds[1,0], bo.scalebounds[1,1], 80)
    x1g,x2g=np.meshgrid(x1,x2)
    
    X=np.c_[x1g.flatten(), x2g.flatten()]
    
    x1_ori = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 80)
    x2_ori = np.linspace(bo.bounds[1,0], bo.bounds[1,1], 80)    
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]

    fig=plt.figure(figsize=(14, 20))
    #fig.suptitle('Gaussian Process and Utility Function After {} Points'.format(len(bo.X)), fontdict={'size':18})
    
    #gs = gridspec.GridSpec(7, 1, height_ratios=[1,1,1,1,1,1,1]) 
    
    nRows=5
    axis_mean2d = fig.add_subplot(nRows, 2, 1)
    axis_variance2d = fig.add_subplot(nRows, 2, 2)
    acq_UCB = fig.add_subplot(nRows, 2, 3)
    acq_EI =fig.add_subplot(nRows, 2,4)
    #acq_POI = plt.subplot(gs[3])
    

    acq_ES = fig.add_subplot(nRows, 2, 5)
    acq_PES = fig.add_subplot(nRows, 2, 6)
    acq_MRS = fig.add_subplot(nRows, 2, 7)
    acq_ydist = fig.add_subplot(nRows, 2, 8)

    acq_VRS = fig.add_subplot(nRows, 2, 9)

    
    mu, sigma = bo.posterior(X)
    #mu_original=mu*(np.max(y_original)-np.min(y_original))+np.mean(y_original)
    #mu_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)
    #sigma_original=sigma*np.std(bo.Y_original)+np.mean(bo.Y_original)**2
    
    # get maximum of mu function
    mu_max=mu.max()
    
    # mean
    CS=axis_mean2d.contourf(x1g_ori,x2g_ori,mu.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2 = plt.contour(CS, levels=CS.levels[::2],colors='r',origin='lower',hold='on')
    
    axis_mean2d.scatter(bo.X_original[:,0],bo.X_original[:,1], label=u'Observations', color='g')    
    axis_mean2d.set_title('Gaussian Process Mean',fontsize=16)
    axis_mean2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    axis_mean2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS, ax=axis_mean2d, shrink=0.9)
    


    # variance
    CS=axis_variance2d.contourf(x1g_ori,x2g_ori,sigma.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2 = plt.contour(CS, levels=CS.levels[::2],colors='r',origin='lower',hold='on')
    
    axis_variance2d.scatter(bo.X_original[:,0],bo.X_original[:,1], label=u'Observations', color='g')    
    axis_variance2d.set_title('Gaussian Process Variance',fontsize=16)
    axis_variance2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    axis_variance2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS, ax=axis_variance2d, shrink=0.9)

    
    # UCB 
    acq_func={}
    acq_func['name']='ucb'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
    utility = myacq.acq_kind(X, bo.gp)
    CS_acq=acq_UCB.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    acq_UCB.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data') 
    acq_UCB.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='*',color='r',s=300,label='Peak')

    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    #acq_UCB.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')
    
    xt_UCB=X[idxBest,:]
    
    acq_UCB.set_title('UCB',fontsize=16)
    acq_UCB.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq_UCB.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS_acq, ax=acq_UCB, shrink=0.9)
    
    # EI 
    acq_func={}
    acq_func['name']='ei'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
    utility = myacq.acq_kind(X, bo.gp)
    CS_acq=acq_EI.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    acq_EI.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  

    acq_EI.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='*',color='r',s=300,label='Peak')
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    #acq_EI.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')

    xt_EI=X[idxBest,:]

    acq_EI.set_title('EI',fontsize=16)
    acq_EI.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq_EI.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS_acq, ax=acq_EI, shrink=0.9)


    # ==================================================================================	
    # finding the xt of Thompson Sampling then use for PES, ES and VRS
    y_max=np.max(bo.Y)
    xstars=[]
    y_stars=[]
    xstars_VRS=[]
    numXtar=50*bo.dim
    for ii in range(numXtar):
        mu_acq={}
        mu_acq['name']='thompson'
        mu_acq['dim']=bo.dim
        mu_acq['scalebounds']=bo.scalebounds    
        acq_mu=AcquisitionFunction(mu_acq)
        xt_TS = acq_max(ac=acq_mu.acq_kind,gp=bo.gp,bounds=bo.scalebounds ,opt_toolbox='scipy')
        
        y_xt_TS=acq_mu.acq_kind(xt_TS,bo.gpmyfunction)
        if y_xt_TS>mu_max:
            y_stars.append(y_xt_TS)

        xstars.append(xt_TS)
        if y_xt_TS>=y_max:
            xstars_VRS.append(xt_TS)
            
            
    # MRS         
    acq_func={}
    acq_func['name']='mes'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['scalebounds']=bo.scalebounds
    acq_func['ystar_suggestions']=y_stars

    myacq=AcquisitionFunction(acq_func)
    utility = myacq.acq_kind(X, bo.gp)
    CS_acq=acq_MRS.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    acq_MRS.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    #acq_MRS.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')

    xt_suggestion_original=xstars*bo.max_min_gap+bo.bounds[:,0]
    acq_MRS.scatter(xt_suggestion_original[:,0],xt_suggestion_original[:,1],marker='s',color='y',s=40,label='xstars')

    acq_MRS.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='*',color='r',s=300,label='Peak')

    acq_MRS.set_title('MES',fontsize=16)
    acq_MRS.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq_MRS.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS_acq, ax=acq_MRS, shrink=0.9)
  
    
    # plot distribution of y_star
    mu_ydist, std_ydist = norm_dist.fit(y_stars)
    # Plot the histogram.
    acq_ydist.hist(y_stars,bins=20,normed=True,alpha=.6,color='g',label=r'Histogram of $y^*$')
    # Plot the PDF.
    x = np.linspace(np.min(y_stars), np.max(y_stars), 100)
    p = norm_dist.pdf(x, mu_ydist, std_ydist)
    acq_ydist.plot(x,p,'k', linewidth=2,label='Gaussian curve')
    acq_ydist.legend()
    acq_ydist.set_title(r'Distribution of $y^*$',fontsize=16)
    
    # PES
    acq_func={}
    acq_func['name']='pes'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['scalebounds']=bo.scalebounds
    #acq_func['xstars']=xstars
    acq_func['xstars']=xstars_VRS
    myacq=AcquisitionFunction(acq_func)
    utility = myacq.acq_kind(X, bo.gp)
    CS_acq=acq_PES.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    
    acq_PES.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    #acq_PES.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')

    temp=np.asarray(myacq.object.x_stars) 
    temp=temp.reshape(-1,2)
    # convert from scale data points to original data points
    xt_suggestion_original=temp*bo.max_min_gap+bo.bounds[:,0]
    
    acq_PES.scatter(xt_suggestion_original[:,0],xt_suggestion_original[:,1],marker='s',color='y',s=40,label='xstars')
    acq_PES.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='*',color='r',s=300,label='Peak')
    xt_PES=X[idxBest,:]

    acq_PES.set_title('PES',fontsize=16)
    acq_PES.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq_PES.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS_acq, ax=acq_PES, shrink=0.9)

    
    # ES     
    acq_func={}
    acq_func['name']='es'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['scalebounds']=bo.scalebounds
    acq_func['xstars']=xstars

    myacq=AcquisitionFunction(acq_func)
    utility = myacq.acq_kind(X, bo.gp)
    CS_acq=acq_ES.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    acq_ES.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    #acq_ES.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')
    temp=np.asarray(myacq.object.x_stars) 
    #temp=temp.reshape(-1,2)
    # convert from scale data points to original data points
    xt_suggestion_original=temp*bo.max_min_gap+bo.bounds[:,0]
    acq_ES.scatter(xt_suggestion_original[:,0],xt_suggestion_original[:,1],marker='s',color='y',s=40,label='xstars')

    acq_ES.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='*',color='r',s=300,label='Peak')

    xt_ES=X[idxBest,:]


    acq_ES.set_title('ES',fontsize=16)
    acq_ES.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq_ES.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS_acq, ax=acq_ES, shrink=0.9)
    
    xstars.append(xt_UCB)
    xstars.append(xt_EI)
    xstars.append(xt_ES)
    #xstars.append(xt_PES)
    
    # Variance Reduction Search
    acq_func={}
    acq_func['name']='vrs'
    acq_func['kappa']=2
    acq_func['n_xstars_x_dim']=50
    acq_func['dim']=2
    acq_func['scalebounds']=bo.scalebounds
    acq_func['xstars']=xstars_VRS

    myacq=AcquisitionFunction(acq_func)

    utility = myacq.acq_kind(X, bo.gp)
    CS_acq=acq_VRS.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    acq_VRS.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  
    temp=np.asarray(myacq.object.xstars) 
    # convert from scale data points to original data points
    xt_suggestion_original=temp*bo.max_min_gap+bo.bounds[:,0]
    
    acq_VRS.scatter(xt_suggestion_original[:,0],xt_suggestion_original[:,1],marker='s',color='y',s=40,label='xstars')
    acq_VRS.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='*',color='r',s=300,label='Peak')

    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    #acq_ES.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')

    acq_VRS.set_title('VRS',fontsize=16)
    acq_VRS.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq_VRS.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS_acq, ax=acq_VRS, shrink=0.9)
    
    strFileName="{:d}_GP2d_acquisition_functions_vrs.eps".format(counter)
    fig.savefig(strFileName, bbox_inches='tight')
    #axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #acq_TS.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    
def plot_acq_bo_2d(bo):
    
    global counter
    counter=counter+1
    
    func=bo.f
    #x_original = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x1 = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 80)
    x2 = np.linspace(bo.scalebounds[1,0], bo.scalebounds[1,1], 80)
    x1g,x2g=np.meshgrid(x1,x2)
    
    X=np.c_[x1g.flatten(), x2g.flatten()]
    
    x1_ori = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 80)
    x2_ori = np.linspace(bo.bounds[1,0], bo.bounds[1,1], 80)    
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]

    #y_original = func(x_original)
    #y = func(x)
    #y_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)

    fig=plt.figure(figsize=(14, 20))
    #fig.suptitle('Gaussian Process and Utility Function After {} Points'.format(len(bo.X)), fontdict={'size':18})
    
    #gs = gridspec.GridSpec(7, 1, height_ratios=[1,1,1,1,1,1,1]) 
    axis_mean2d = fig.add_subplot(4, 2, 1)
    axis_variance2d = fig.add_subplot(4, 2, 2)
    acq_UCB = fig.add_subplot(4, 2, 3)
    acq_EI =fig.add_subplot(4, 2,4)
    #acq_POI = plt.subplot(gs[3])
    

    acq_ES = fig.add_subplot(4, 2, 5)
    acq_PES = fig.add_subplot(4, 2, 6)
    acq_MRS = fig.add_subplot(4, 2, 7)
    acq_Consensus = fig.add_subplot(4, 2, 8)

    
    mu, sigma = bo.posterior(X)
    #mu_original=mu*(np.max(y_original)-np.min(y_original))+np.mean(y_original)
    #mu_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)
    #sigma_original=sigma*np.std(bo.Y_original)+np.mean(bo.Y_original)**2
    
    
    # mean
    CS=axis_mean2d.contourf(x1g_ori,x2g_ori,mu.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2 = plt.contour(CS, levels=CS.levels[::2],colors='r',origin='lower',hold='on')
    
    axis_mean2d.scatter(bo.X_original[:,0],bo.X_original[:,1], label=u'Observations', color='g')    
    axis_mean2d.set_title('Gaussian Process Mean',fontsize=16)
    axis_mean2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    axis_mean2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS, ax=axis_mean2d, shrink=0.9)
    


    # variance
    CS=axis_variance2d.contourf(x1g_ori,x2g_ori,sigma.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2 = plt.contour(CS, levels=CS.levels[::2],colors='r',origin='lower',hold='on')
    
    axis_variance2d.scatter(bo.X_original[:,0],bo.X_original[:,1], label=u'Observations', color='g')    
    axis_variance2d.set_title('Gaussian Process Variance',fontsize=16)
    axis_variance2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    axis_variance2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS, ax=axis_variance2d, shrink=0.9)

    
    # UCB 
    acq_func={}
    acq_func['name']='ucb'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
    utility = myacq.acq_kind(X, bo.gp)
    CS_acq=acq_UCB.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    acq_UCB.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='*',color='r',s=300,label='Peak')
    acq_UCB.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    #acq_UCB.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')
    
    xt_UCB=X[idxBest,:]
    
    acq_UCB.set_title('UCB',fontsize=16)
    acq_UCB.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq_UCB.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS_acq, ax=acq_UCB, shrink=0.9)
    
    # EI 
    acq_func={}
    acq_func['name']='ei'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
    utility = myacq.acq_kind(X, bo.gp)
    CS_acq=acq_EI.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    acq_EI.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='*',color='r',s=300,label='Peak')
    acq_EI.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    #acq_EI.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')

    xt_EI=X[idxBest,:]


    acq_EI.set_title('EI',fontsize=16)
    acq_EI.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq_EI.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS_acq, ax=acq_EI, shrink=0.9)
    
    # MRS         
    acq_func={}
    acq_func['name']='mrs'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
    utility = myacq.acq_kind(X, bo.gp)
    CS_acq=acq_MRS.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    acq_MRS.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='*',color='r',s=300,label='Peak')
    acq_MRS.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    #acq_MRS.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')

    acq_MRS.set_title('MRS',fontsize=16)
    acq_MRS.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq_MRS.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS_acq, ax=acq_MRS, shrink=0.9)	

    # PES
    acq_func={}
    acq_func['name']='pes'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
    utility = myacq.acq_kind(X, bo.gp)
    CS_acq=acq_PES.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    acq_PES.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='*',color='r',s=300,label='Peak')
    acq_PES.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    #acq_PES.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')

    xt_PES=X[idxBest,:]


    acq_PES.set_title('PES',fontsize=16)
    acq_PES.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq_PES.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS_acq, ax=acq_PES, shrink=0.9)
    
    # ES     
    acq_func={}
    acq_func['name']='es'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
    utility = myacq.acq_kind(X, bo.gp)
    CS_acq=acq_ES.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    acq_ES.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='*',color='r',s=300,label='Peak')
    acq_ES.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    #acq_ES.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')

    xt_ES=X[idxBest,:]

    acq_ES.set_title('ES',fontsize=16)
    acq_ES.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq_ES.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS_acq, ax=acq_ES, shrink=0.9)
    
    xstars=[]
    xstars.append(xt_UCB)
    xstars.append(xt_EI)
    xstars.append(xt_ES)
    xstars.append(xt_PES)
    
    # Consensus     
    acq_func={}
    acq_func['name']='consensus'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['scalebounds']=bo.scalebounds
    acq_func['xstars']=xstars

    myacq=AcquisitionFunction(acq_func)

    utility = myacq.acq_kind(X, bo.gp)
    CS_acq=acq_Consensus.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    acq_Consensus.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  
    temp=np.asarray(myacq.object.xstars) 
    # convert from scale data points to original data points
    xt_suggestion_original=temp*bo.max_min_gap+bo.bounds[:,0]
    
    acq_Consensus.scatter(xt_suggestion_original[:,0],xt_suggestion_original[:,1],marker='s',color='y',s=100,label='xstars')
    acq_Consensus.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='*',color='r',s=300,label='Peak')

    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    #acq_ES.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')

    acq_Consensus.set_title('Consensus',fontsize=16)
    acq_Consensus.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq_Consensus.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS_acq, ax=acq_Consensus, shrink=0.9)
    
    strFileName="{:d}_GP2d_acquisition_functions.eps".format(counter)
    fig.savefig(strFileName, bbox_inches='tight')
    #axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #acq_TS.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)


def plot_bo_2d_pvrs(bo):
    
    global counter
    counter=counter+1
    
    func=bo.f
    #x_original = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x1 = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 80)
    x2 = np.linspace(bo.scalebounds[1,0], bo.scalebounds[1,1], 80)
    x1g,x2g=np.meshgrid(x1,x2)
    
    X=np.c_[x1g.flatten(), x2g.flatten()]
    
    x1_ori = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 80)
    x2_ori = np.linspace(bo.bounds[1,0], bo.bounds[1,1], 80)    
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]

    #y_original = func(x_original)
    #y = func(x)
    #y_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)

    fig=plt.figure(figsize=(12, 13))
    #fig.suptitle('Gaussian Process and Utility Function After {} Points'.format(len(bo.X)), fontdict={'size':18})
    
    #gs = gridspec.GridSpec(7, 1, height_ratios=[1,1,1,1,1,1,1]) 
    axis_mean2d = fig.add_subplot(3, 2, 1)
    axis_variance2d = fig.add_subplot(3, 2, 2)
    acq_UCB = fig.add_subplot(3, 2, 3)
    acq_EI =fig.add_subplot(3, 2,4)
    #acq_POI = plt.subplot(gs[3])
    
    acq_VRS = fig.add_subplot(3, 2, 5)
    acq_Batch_VRS_B_2 = fig.add_subplot(3, 2, 6)
    
    mu, sigma = bo.posterior(X)
    #mu_original=mu*(np.max(y_original)-np.min(y_original))+np.mean(y_original)
    #mu_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)
    #sigma_original=sigma*np.std(bo.Y_original)+np.mean(bo.Y_original)**2
    
    
    # mean
    CS=axis_mean2d.contourf(x1g_ori,x2g_ori,mu.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2 = plt.contour(CS, levels=CS.levels[::2],colors='r',origin='lower',hold='on')
    
    axis_mean2d.scatter(bo.X_original[:,0],bo.X_original[:,1], label=u'Observations', color='g')    
    axis_mean2d.set_title('Gaussian Process Mean',fontsize=16)
    axis_mean2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    axis_mean2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    axis_mean2d.get_xaxis().set_visible(False)
    axis_mean2d.get_yaxis().set_visible(False)
    fig.colorbar(CS, ax=axis_mean2d, shrink=0.9)
    

    # variance
    CS=axis_variance2d.contourf(x1g_ori,x2g_ori,sigma.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2 = plt.contour(CS, levels=CS.levels[::2],colors='r',origin='lower',hold='on')
    
    axis_variance2d.scatter(bo.X_original[:,0],bo.X_original[:,1], label=u'Observations', color='g')    
    axis_variance2d.set_title('Gaussian Process Variance',fontsize=16)
    axis_variance2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    axis_variance2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    
    axis_variance2d.get_xaxis().set_visible(False)
    axis_variance2d.get_yaxis().set_visible(False)
    fig.colorbar(CS, ax=axis_variance2d, shrink=0.9)

    # ==================================================================================	
    # finding the xt of Thompson Sampling then use for PES, ES and VRS
    y_max=np.max(bo.Y)
    xstars=[]
    y_stars=[]
    xstars_VRS=[]
    numXtar=25*bo.dim
    for ii in range(numXtar):
        mu_acq={}
        mu_acq['name']='thompson'
        mu_acq['dim']=bo.dim
        mu_acq['scalebounds']=bo.scalebounds    
        acq_mu=AcquisitionFunction(mu_acq)
        xt_TS = acq_max(ac=acq_mu.acq_kind,gp=bo.gp,bounds=bo.scalebounds ,opt_toolbox='scipy')
        
        y_xt_TS=acq_mu.acq_kind(xt_TS,bo.gp)
        #if y_xt_TS>mu_max:
        y_stars.append(y_xt_TS)

        xstars.append(xt_TS)
        #if y_xt_TS>=y_max:
        xstars_VRS.append(xt_TS)
            
        
        
    # UCB 
    acq_func={}
    acq_func['name']='ucb'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
    utility = myacq.acq_kind(X, bo.gp)
    CS_acq=acq_UCB.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    acq_UCB.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='s',color='r',s=100,label='Peak')
    acq_UCB.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    #acq_UCB.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')
    
    xt_UCB=X[idxBest,:]
    
    acq_UCB.set_title('UCB',fontsize=16)
    acq_UCB.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq_UCB.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    acq_UCB.get_xaxis().set_visible(False)
    acq_UCB.get_yaxis().set_visible(False)
    
    
    fig.colorbar(CS_acq, ax=acq_UCB, shrink=0.9)
    
    # EI 
    acq_func={}
    acq_func['name']='ei'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
    utility = myacq.acq_kind(X, bo.gp)
    CS_acq=acq_EI.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    acq_EI.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='s',color='r',s=100,label='Peak')
    acq_EI.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    #acq_EI.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')

    xt_EI=X[idxBest,:]


    acq_EI.set_title('EI',fontsize=16)
    acq_EI.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq_EI.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS_acq, ax=acq_EI, shrink=0.9)
    acq_EI.get_xaxis().set_visible(False)
    acq_EI.get_yaxis().set_visible(False)
    
    
    
    # Predictive Variance Reduction Search
    acq_func={}
    acq_func['name']='pvrs'
    acq_func['kappa']=2
    acq_func['n_xstars_x_dim']=50
    acq_func['dim']=2
    acq_func['scalebounds']=bo.scalebounds
    acq_func['xstars']=xstars_VRS

    myacq=AcquisitionFunction(acq_func)

    utility = myacq.acq_kind(X, bo.gp)
    CS_acq=acq_VRS.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    acq_VRS.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Existing data X')  
    temp=np.asarray(myacq.object.xstars) 
    # convert from scale data points to original data points
    xt_suggestion_original=temp*bo.max_min_gap+bo.bounds[:,0]
    
    acq_VRS.scatter(xt_suggestion_original[:,0],xt_suggestion_original[:,1],marker='*',color='y',s=150,label=r'$x^*$')
    #acq_VRS.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='s',color='r',s=100,label='Peak')

    acq_VRS.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='s',color='r',s=100,label='Selected point $x_t$')

    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    #acq_ES.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')

    acq_VRS.set_title('PVRS',fontsize=16)
    acq_VRS.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq_VRS.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS_acq, ax=acq_VRS, shrink=0.9)
    acq_VRS.get_xaxis().set_visible(False)
    acq_VRS.get_yaxis().set_visible(False)
    
    
    
    # Batch Variance Reduction Search B=2
    
    acq_Batch_VRS_B_2.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Existing data X')  
    temp=np.asarray(myacq.object.xstars) 
    # convert from scale data points to original data points
    xt_suggestion_original=temp*bo.max_min_gap+bo.bounds[:,0]
    
    acq_Batch_VRS_B_2.scatter(xt_suggestion_original[:,0],xt_suggestion_original[:,1],marker='*',color='y',s=150,label=r'$x^*$ samples')
    #acq_VRS.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='*',color='r',s=300,label='Peak')

    acq_func={}
    acq_func['name']='ucb'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['xstars']=xstars_VRS


    acq_params={}
    acq_params['acq_func']=acq_func
    acq_params['optimize_gp']=1
    acq_params['n_xstars']=100

    
    func_params={}
    func_params['bounds']=bo.bounds
    func_params['f']=func
    func_params['function']=bo.function


    gp_params = {'lengthscale':0.2*2,'noise_delta':1e-8}
    bo2=BatchPVRS(gp_params,func_params, acq_params)
    bo2.init_with_data(bo.X_original,bo.Y_original)
       
    #new_X=bo2.maximize_batch_sequential_greedy_PVRS(gp_params,B=3)
      
    new_X=bo2.maximize_batch_greedy_PVRS(B=2)
    new_X_original=new_X*bo.max_min_gap+bo.bounds[:,0]


    
    acq_Batch_VRS_B_2.scatter(new_X_original[:,0],new_X_original[:,1],marker='s',color='r',s=100,label='Selected point $x_t$')
    #acq_ES.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')

    acq_Batch_VRS_B_2.set_title('B-PVRS B=2',fontsize=16)
    acq_Batch_VRS_B_2.set_xlim(bo.bounds[0,0]-0.1, bo.bounds[0,1]+0.1)
    acq_Batch_VRS_B_2.set_ylim(bo.bounds[1,0]-0.1, bo.bounds[1,1]+0.1)
    fig.colorbar(CS_acq, ax=acq_Batch_VRS_B_2, shrink=0.9)
    acq_Batch_VRS_B_2.get_xaxis().set_visible(False)
    acq_Batch_VRS_B_2.get_yaxis().set_visible(False)
    
    
    acq_VRS.legend(loc='center left', bbox_to_anchor=(0.1, -0.2),prop={'size':20},ncol=3)

    
    strFileName="{:d}_GP2d_acquisition_functions.eps".format(counter)
    fig.savefig(strFileName, bbox_inches='tight')
    #axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #acq_TS.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)



def plot_bo_2d_pvrs_short(bo):
    
    global counter
    counter=counter+1
    
    func=bo.f
    #x_original = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x1 = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 80)
    x2 = np.linspace(bo.scalebounds[1,0], bo.scalebounds[1,1], 80)
    x1g,x2g=np.meshgrid(x1,x2)
    
    X=np.c_[x1g.flatten(), x2g.flatten()]
    
    x1_ori = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 80)
    x2_ori = np.linspace(bo.bounds[1,0], bo.bounds[1,1], 80)    
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]

    #y_original = func(x_original)
    #y = func(x)
    #y_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)

    fig=plt.figure(figsize=(13, 7))
    #fig.suptitle('Gaussian Process and Utility Function After {} Points'.format(len(bo.X)), fontdict={'size':18})
    
    #gs = gridspec.GridSpec(7, 1, height_ratios=[1,1,1,1,1,1,1]) 
    axis_mean2d = fig.add_subplot(2, 2, 1)
    axis_variance2d = fig.add_subplot(2, 2, 2)
    #acq_UCB = fig.add_subplot(2, 2, 3)
    #acq_EI =fig.add_subplot(3, 2,4)
    #acq_POI = plt.subplot(gs[3])
    
    acq_VRS = fig.add_subplot(2, 2, 3)
    acq_Batch_VRS_B_2 = fig.add_subplot(2, 2, 4)
    
    mu, sigma = bo.posterior(X)
    #mu_original=mu*(np.max(y_original)-np.min(y_original))+np.mean(y_original)
    #mu_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)
    #sigma_original=sigma*np.std(bo.Y_original)+np.mean(bo.Y_original)**2
    
    
    # mean
    CS=axis_mean2d.contourf(x1g_ori,x2g_ori,mu.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2 = plt.contour(CS, levels=CS.levels[::2],colors='r',origin='lower',hold='on')
    
    axis_mean2d.scatter(bo.X_original[:,0],bo.X_original[:,1], label=u'Observations', color='g')    
    axis_mean2d.set_title('Gaussian Process Mean',fontsize=16)
    axis_mean2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    axis_mean2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    axis_mean2d.get_xaxis().set_visible(False)
    axis_mean2d.get_yaxis().set_visible(False)
    fig.colorbar(CS, ax=axis_mean2d, shrink=0.9)
    

    # variance
    CS=axis_variance2d.contourf(x1g_ori,x2g_ori,sigma.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2 = plt.contour(CS, levels=CS.levels[::2],colors='r',origin='lower',hold='on')
    
    axis_variance2d.scatter(bo.X_original[:,0],bo.X_original[:,1], label=u'Observations', color='g')    
    axis_variance2d.set_title('Gaussian Process Variance',fontsize=16)
    axis_variance2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    axis_variance2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    
    axis_variance2d.get_xaxis().set_visible(False)
    axis_variance2d.get_yaxis().set_visible(False)
    fig.colorbar(CS, ax=axis_variance2d, shrink=0.9)

    # ==================================================================================	
    # finding the xt of Thompson Sampling then use for PES, ES and VRS
    y_max=np.max(bo.Y)
    xstars=[]
    y_stars=[]
    xstars_VRS=[]
    numXtar=25*bo.dim
    for ii in range(numXtar):
        mu_acq={}
        mu_acq['name']='thompson'
        mu_acq['dim']=bo.dim
        mu_acq['scalebounds']=bo.scalebounds    
        acq_mu=AcquisitionFunction(mu_acq)
        xt_TS = acq_max(ac=acq_mu.acq_kind,gp=bo.gp,bounds=bo.scalebounds ,opt_toolbox='scipy')
        
        y_xt_TS=acq_mu.acq_kind(xt_TS,bo.gp)
        #if y_xt_TS>mu_max:
        y_stars.append(y_xt_TS)

        xstars.append(xt_TS)
        #if y_xt_TS>=y_max:
        xstars_VRS.append(xt_TS)
            
        
        
    # UCB 
    """
    acq_func={}
    acq_func['name']='ucb'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
    utility = myacq.acq_kind(X, bo.gp)
    CS_acq=acq_UCB.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    acq_UCB.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='s',color='r',s=100,label='Peak')
    acq_UCB.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    #acq_UCB.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')
    
    xt_UCB=X[idxBest,:]
    
    acq_UCB.set_title('UCB',fontsize=16)
    acq_UCB.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq_UCB.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    acq_UCB.get_xaxis().set_visible(False)
    acq_UCB.get_yaxis().set_visible(False)
    
    
    fig.colorbar(CS_acq, ax=acq_UCB, shrink=0.9)
    
    # EI 
    acq_func={}
    acq_func['name']='ei'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
    utility = myacq.acq_kind(X, bo.gp)
    CS_acq=acq_EI.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    acq_EI.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='s',color='r',s=100,label='Peak')
    acq_EI.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    #acq_EI.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')

    xt_EI=X[idxBest,:]


    acq_EI.set_title('EI',fontsize=16)
    acq_EI.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq_EI.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS_acq, ax=acq_EI, shrink=0.9)
    acq_EI.get_xaxis().set_visible(False)
    acq_EI.get_yaxis().set_visible(False)
    """
    
    
    # Predictive Variance Reduction Search
    acq_func={}
    acq_func['name']='pvrs'
    acq_func['kappa']=2
    acq_func['n_xstars_x_dim']=50
    acq_func['dim']=2
    acq_func['scalebounds']=bo.scalebounds
    acq_func['xstars']=xstars_VRS

    myacq=AcquisitionFunction(acq_func)

    utility = myacq.acq_kind(X, bo.gp)
    CS_acq=acq_VRS.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    acq_VRS.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data X')  
    temp=np.asarray(myacq.object.xstars) 
    # convert from scale data points to original data points
    xt_suggestion_original=temp*bo.max_min_gap+bo.bounds[:,0]
    
    acq_VRS.scatter(xt_suggestion_original[:,0],xt_suggestion_original[:,1],marker='*',color='y',s=150,label=r'Sampled $x^*$')
    #acq_VRS.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='s',color='r',s=100,label='Peak')

    acq_VRS.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='s',color='r',s=100,label='Selected $x_t$')

    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    #acq_ES.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')

    acq_VRS.set_title('PVRS',fontsize=16)
    acq_VRS.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq_VRS.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS_acq, ax=acq_VRS, shrink=0.9)
    acq_VRS.get_xaxis().set_visible(False)
    acq_VRS.get_yaxis().set_visible(False)
    
    
    
    # Batch Variance Reduction Search B=2
    
    acq_Batch_VRS_B_2.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data X')  
    temp=np.asarray(myacq.object.xstars) 
    # convert from scale data points to original data points
    xt_suggestion_original=temp*bo.max_min_gap+bo.bounds[:,0]
    
    acq_Batch_VRS_B_2.scatter(xt_suggestion_original[:,0],xt_suggestion_original[:,1],marker='*',color='y',s=150,label=r'$x^*$ samples')
    #acq_VRS.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='*',color='r',s=300,label='Peak')

    acq_func={}
    acq_func['name']='ucb'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['xstars']=xstars_VRS


    acq_params={}
    acq_params['acq_func']=acq_func
    acq_params['optimize_gp']=1
    acq_params['n_xstars']=100

    
    func_params={}
    func_params['bounds']=bo.bounds
    func_params['f']=func
    func_params['function']=bo.function


    gp_params = {'lengthscale':0.2*2,'noise_delta':1e-8}
    bo2=BatchPVRS(gp_params,func_params, acq_params)
    bo2.init_with_data(bo.X_original,bo.Y_original)
       
    #new_X=bo2.maximize_batch_sequential_greedy_PVRS(gp_params,B=3)
      
    new_X=bo2.maximize_batch_greedy_PVRS(B=2)
    new_X_original=new_X*bo.max_min_gap+bo.bounds[:,0]


    
    acq_Batch_VRS_B_2.scatter(new_X_original[:,0],new_X_original[:,1],marker='s',color='r',s=100,label='Selected point $x_t$')
    #acq_ES.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')

    acq_Batch_VRS_B_2.set_title('B-PVRS B=2',fontsize=16)
    acq_Batch_VRS_B_2.set_xlim(bo.bounds[0,0]-0.1, bo.bounds[0,1]+0.1)
    acq_Batch_VRS_B_2.set_ylim(bo.bounds[1,0]-0.1, bo.bounds[1,1]+0.1)
    fig.colorbar(CS_acq, ax=acq_Batch_VRS_B_2, shrink=0.9)
    acq_Batch_VRS_B_2.get_xaxis().set_visible(False)
    acq_Batch_VRS_B_2.get_yaxis().set_visible(False)
    
    
    #acq_VRS.legend(loc='center left', bbox_to_anchor=(0.1, -0.2),prop={'size':20},ncol=1)
    acq_VRS.legend( bbox_to_anchor=(3.45, 1.4),prop={'size':17},ncol=1)

    
    strFileName="{:d}_GP2d_acquisition_functions.eps".format(counter)
    fig.savefig(strFileName, bbox_inches='tight')
    #axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #acq_TS.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)

    
def plot_bo_2d(bo):
    
    x1 = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 100)
    x2 = np.linspace(bo.scalebounds[1,0], bo.scalebounds[1,1], 100)
    x1g,x2g=np.meshgrid(x1,x2)
    
    X=np.c_[x1g.flatten(), x2g.flatten()]
    
    x1_ori = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x2_ori = np.linspace(bo.bounds[1,0], bo.bounds[1,1], 100)    
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]
  
    fig = plt.figure()
    
    #axis2d = fig.add_subplot(1, 2, 1)
    acq2d = fig.add_subplot(1, 1, 1)
    
    #mu, sigma = bo.posterior(X)
    # plot the acquisition function

    utility = bo.acq_func.acq_kind(X, bo.gp)
    #acq3d.plot_surface(x1g,x1g,utility.reshape(x1g.shape))
    
    CS_acq=acq2d.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    acq2d.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='s',color='r',s=30,label='Peak')
    acq2d.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')

    acq2d.set_title('Acquisition Function',fontsize=16)
    acq2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    
    #acq2d.legend(loc=1, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq2d.legend(loc='center left',ncol=3,bbox_to_anchor=(0, -0.2))
      
    fig.colorbar(CS_acq, ax=acq2d, shrink=0.9)

    #acq.set_xlim((np.min(x), np.max(x)))
    #acq.set_ylim((np.min(utility), 1.1*np.max(utility)))
    #acq.set_ylabel('Acq', fontdict={'size':16})
    #acq.set_xlabel('x', fontdict={'size':16})
    
    #axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)


def plot_bo_2d_withGPmeans(bo):
    
    x1 = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 100)
    x2 = np.linspace(bo.scalebounds[1,0], bo.scalebounds[1,1], 100)
    
    x1g,x2g=np.meshgrid(x1,x2)
    
    X=np.c_[x1g.flatten(), x2g.flatten()]
    
    x1_ori = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x2_ori = np.linspace(bo.bounds[1,0], bo.bounds[1,1], 100)
    
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]
    
    #fig.suptitle('Gaussian Process and Utility Function After {} Points'.format(len(bo.X)), fontdict={'size':18})
    
    fig = plt.figure(figsize=(12, 5))
    
    #axis3d = fig.add_subplot(1, 2, 1, projection='3d')
    axis2d = fig.add_subplot(1, 2, 1)
    #acq3d = fig.add_subplot(2, 2, 3, projection='3d')
    acq2d = fig.add_subplot(1, 2, 2)
    
    mu, sigma = bo.posterior(X)
    #axis.plot(x, y, linewidth=3, label='Target')
    #axis3d.plot_surface(x1g,x1g,mu.reshape(x1g.shape))
    #axis3d.scatter(bo.X[:,0],bo.X[:,1], bo.Y,zdir='z',  label=u'Observations', color='r')    

    
    CS=axis2d.contourf(x1g_ori,x2g_ori,mu.reshape(x1g.shape),cmap=plt.cm.bone,origin='lower')
    CS2 = plt.contour(CS, levels=CS.levels[::2],colors='r',origin='lower',hold='on')
    
    axis2d.scatter(bo.X_original[:,0],bo.X_original[:,1], label=u'Observations', color='g')    
    axis2d.set_title('Gaussian Process Mean',fontsize=16)
    axis2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    axis2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS, ax=axis2d, shrink=0.9)

    #plt.colorbar(ax=axis2d)

    #axis.plot(x, mu, '--', color='k', label='Prediction')
    
    
    #axis.set_xlim((np.min(x), np.max(x)))
    #axis.set_ylim((None, None))
    #axis.set_ylabel('f(x)', fontdict={'size':16})
    #axis.set_xlabel('x', fontdict={'size':16})
    
    # plot the acquisition function

    utility = bo.acq_func.acq_kind(X, bo.gp)
    #acq3d.plot_surface(x1g,x1g,utility.reshape(x1g.shape))
    
    #CS_acq=acq2d.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=plt.cm.bone,origin='lower')
    CS_acq=acq2d.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)

    
    acq2d.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g')  
    
        
    acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=60)
    acq2d.scatter(X_ori[idxBest,0],X_ori[idxBest,1],color='b',s=60)
    
    
    acq2d.set_title('Acquisition Function',fontsize=16)
    acq2d.set_xlim(bo.bounds[0,0]-0.2, bo.bounds[0,1]+0.2)
    acq2d.set_ylim(bo.bounds[1,0]-0.2, bo.bounds[1,1]+0.2)
             
    #acq.set_xlim((np.min(x), np.max(x)))
    #acq.set_ylim((np.min(utility), 1.1*np.max(utility)))
    #acq.set_ylabel('Acq', fontdict={'size':16})
    #acq.set_xlabel('x', fontdict={'size':16})
    
    #axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    
    fig.colorbar(CS_acq, ax=acq2d, shrink=0.9)

def plot_bo_2d_withGPmeans_Sigma(bo):
    
    x1 = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 100)
    x2 = np.linspace(bo.scalebounds[1,0], bo.scalebounds[1,1], 100)
    
    x1g,x2g=np.meshgrid(x1,x2)
    
    X=np.c_[x1g.flatten(), x2g.flatten()]
    
    x1_ori = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x2_ori = np.linspace(bo.bounds[1,0], bo.bounds[1,1], 100)
    
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]
    
    #fig.suptitle('Gaussian Process and Utility Function After {} Points'.format(len(bo.X)), fontdict={'size':18})
    
    fig = plt.figure(figsize=(12, 3))
    
    #axis3d = fig.add_subplot(1, 2, 1, projection='3d')
    axis2d = fig.add_subplot(1, 2, 1)
    #acq3d = fig.add_subplot(2, 2, 3, projection='3d')
    acq2d = fig.add_subplot(1, 2, 2)
    
    mu, sigma = bo.posterior(X)
    #axis.plot(x, y, linewidth=3, label='Target')
    #axis3d.plot_surface(x1g,x1g,mu.reshape(x1g.shape))
    #axis3d.scatter(bo.X[:,0],bo.X[:,1], bo.Y,zdir='z',  label=u'Observations', color='r')    

    utility = bo.acq_func.acq_kind(X, bo.gp)

    CS=axis2d.contourf(x1g_ori,x2g_ori,mu.reshape(x1g.shape),cmap=plt.cm.bone,origin='lower')
    CS2 = plt.contour(CS, levels=CS.levels[::2],colors='r',origin='lower',hold='on')
    
    axis2d.scatter(bo.X_original[:,0],bo.X_original[:,1], label=u'Observations', color='g')    
    axis2d.set_title('Gaussian Process Mean',fontsize=16)
    axis2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    axis2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS, ax=axis2d, shrink=0.9)

    
    #CS_acq=acq2d.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=plt.cm.bone,origin='lower')
    CS_acq=acq2d.contourf(x1g_ori,x2g_ori,sigma.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)

    
    acq2d.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g')  
    
        
    acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=60)
    acq2d.scatter(X_ori[idxBest,0],X_ori[idxBest,1],color='b',s=60)
    
    
    acq2d.set_title('Gaussian Process Variance',fontsize=16)
    #acq2d.set_xlim(bo.bounds[0,0]-0.2, bo.bounds[0,1]+0.2)
    #acq2d.set_ylim(bo.bounds[1,0]-0.2, bo.bounds[1,1]+0.2)
             
    
    fig.colorbar(CS_acq, ax=acq2d, shrink=0.9)
    
