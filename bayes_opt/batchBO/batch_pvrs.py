# -*- coding: utf-8 -*-


from __future__ import division


import sys
sys.path.insert(0,'../../')
sys.path.insert(0,'..')


import numpy as np
from bayes_opt.acquisition_functions import AcquisitionFunction, unique_rows
#from bayes_opt import visualization

#from visualization import Visualization
from bayes_opt.gaussian_process import GaussianProcess
#from visualization import *
from bayes_opt.visualization import visualization
from bayes_opt.acquisition_maximization import acq_max,acq_max_with_name,acq_max_with_init

#from bayes_opt.visualization import vis_variance_reduction_search as viz
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import cluster
from sklearn import mixture
import matplotlib.pyplot as plt
from scipy.ndimage import filters
import time
import copy
from matplotlib import rc

#======================================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================

class BatchPVRS(object):

    def __init__(self,gp_params, func_params, acq_params):
        """      
        Input parameters
        ----------
        
        gp_params:                  GP parameters
        gp_params.thete:            to compute the kernel
        gp_params.delta:            to compute the kernel
        
        func_params:                function to optimize
        func_params.init bound:     initial bounds for parameters
        func_params.bounds:        bounds on parameters        
        func_params.func:           a function to be optimized
        
        
        acq_params:            acquisition function, 
        acq_params.acq_func['name']=['ei','ucb','poi','lei']
                            ,acq['kappa'] for ucb, acq['k'] for lei
        acq_params.opt_toolbox:     optimization toolbox 'nlopt','direct','scipy'
                            
        Returns
        -------
        dim:            dimension
        bounds:         bounds on original scale
        scalebounds:    bounds on normalized scale of 0-1
        time_opt:       will record the time spent on optimization
        gp:             Gaussian Process object
        """
        
        try:
            bounds=func_params['bounds']
        except:
            bounds=func_params['function'].bounds      
            
        self.dim = len(bounds)

        # Create an array with parameters bounds
        if isinstance(bounds,dict):
            # Get the name of the parameters
            self.keys = list(bounds.keys())
        
            self.bounds = []
            for key in list(bounds.keys()):
                self.bounds.append(bounds[key])
            self.bounds = np.asarray(self.bounds)
        else:
            self.bounds=np.asarray(bounds)

     

        scalebounds=np.array([np.zeros(self.dim), np.ones(self.dim)])
        self.scalebounds=scalebounds.T
        
        self.max_min_gap=self.bounds[:,1]-self.bounds[:,0]
                    
        # acquisition function type
        self.acq=acq_params['acq_func']
        
        if 'debug' not in self.acq:
            self.acq['debug']=0
            
        if 'optimize_gp' not in acq_params:
            self.optimize_gp=0
        else:                
            self.optimize_gp=acq_params['optimize_gp']
        
        if 'marginalize_gp' not in acq_params:
            self.marginalize_gp=0
        else:                
            self.marginalize_gp=acq_params['marginalize_gp']
            
        # Some function to be optimized
           
        self.function=func_params['function']
        try:
            self.f = func_params['function']['func']
        except:
            self.f = func_params['function'].func
            
        # optimization toolbox
        if 'opt_toolbox' not in acq_params:
            self.opt_toolbox='scipy'
        else:
            self.opt_toolbox=acq_params['opt_toolbox']

        
        # store the batch size for each iteration
        self.NumPoints=[]
        # Numpy array place holders
        self.X_original= None
        
        # scale the data to 0-1 fit GP better
        self.X = None # X=( X_original - min(bounds) / (max(bounds) - min(bounds))
        
        self.Y = None # Y=( Y_original - mean(bounds) / (max(bounds) - min(bounds))
        self.Y_original = None
        self.opt_time=0
        
        self.L=0 # lipschitz

        self.gp=GaussianProcess(gp_params)
        
        self.gp_params=gp_params

        # Acquisition Function
        #self.acq_func = None
        self.acq_func = AcquisitionFunction(acq=self.acq)
        self.accum_dist=[]
        
        # theta vector for marginalization GP
        self.theta_vector =[]
        
        if 'xstars' not in self.acq:
            self.xstars=[]
        else:
            self.xstars=self.acq['xstars']
            
        # PVRS before and after
        self.PVRS_before_after=[]
        self.xstars=[]
        
        self.Y_original_maxGP=None
        self.X_original_maxGP=None

    def posterior(self, Xnew):
        #xmin, xmax = -2, 10
        ur = unique_rows(self.X)

        self.gp.fit(self.X[ur], self.Y[ur])
        mu, sigma2 = self.gp.predict(Xnew, eval_MSE=True)
        return mu, np.sqrt(sigma2)
    
    
    def init(self, n_init_points):
        """      
        Input parameters
        ----------
        gp_params:            Gaussian Process structure      
        n_init_points:        # init points
        """

        # Generate random points
        l = [np.random.uniform(x[0], x[1], size=n_init_points) for x in self.bounds]

        # Concatenate new random points to possible existing
        # points from self.explore method.
        #self.init_points += list(map(list, zip(*l)))
        temp=np.asarray(l)
        temp=temp.T
        init_X=list(temp.reshape((n_init_points,-1)))

        # Evaluate target function at all initialization           
        y_init=self.f(init_X)

        # Turn it into np array and store.
        self.X_original=np.asarray(init_X)
        temp_init_point=np.divide((init_X-self.bounds[:,0]),self.max_min_gap)
        
        self.X_original_maxGP= np.asarray(init_X)

        
        self.X_original = np.asarray(init_X)
        self.X = np.asarray(temp_init_point)
        y_init=np.reshape(y_init,(n_init_points,1))
        
        self.Y_original = np.asarray(y_init)
        self.Y_original_maxGP=np.asarray(y_init)      

        self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))
        
        self.NumPoints=np.append(self.NumPoints,n_init_points)
        
        # Set parameters if any was passed
        #self.gp=GaussianProcess(gp_params)
        
        # Find unique rows of X to avoid GP from breaking
        #ur = unique_rows(self.X)
        #self.gp.fit(self.X[ur], self.Y[ur])
        
        #print "#Batch={:d} f_max={:.4f}".format(n_init_points,self.Y.max())

    def init_with_data(self, init_X,init_Y):
        """      
        Input parameters
        ----------
        gp_params:            Gaussian Process structure      
        x,y:        # init data observations (in original scale)
        """

        # Turn it into np array and store.
        self.X_original=np.asarray(init_X)
        temp_init_point=np.divide((init_X-self.bounds[:,0]),self.max_min_gap)
        
        self.X_original = np.asarray(init_X)
        self.X_original_maxGP= np.asarray(init_X)

        self.X = np.asarray(temp_init_point)
        
        self.Y_original = np.asarray(init_Y)
        self.Y_original_maxGP=np.asarray(init_Y)      

        self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))
        
        self.NumPoints=np.append(self.NumPoints,len(init_Y))
        
        # Set acquisition function
        self.acq_func = AcquisitionFunction(self.acq)

        # Find unique rows of X to avoid GP from breaking
        ur = unique_rows(self.X)
        self.gp.fit(self.X[ur], self.Y[ur])


    def compute_PredictiveVariance(self,Xstars,X_t):
        """
        Xstars:     locations of global optimums
        X:          existing observations
        X_t:        suggested_batch        
        """
        
        # for robustness, remove empty X_t
        X_t=np.atleast_2d(X_t)
        mask = ~np.any(np.isnan(X_t), axis=1)
        X_t = X_t[mask]
                
        X=np.vstack((self.X,X_t))
        
        var=self.gp.compute_var(X,Xstars)
        mean_variance=np.mean(var)

        return np.asarray(mean_variance)    		

    def maximize_batch_PVRS_iterative_greedy(self,B=5,first_batch=[]):
        """
        Finding a batch of points using Peak Suppression / Constant Liar approach
        
        Input Parameters
        ----------

        gp_params:          Parameters to be passed to the Gaussian Process class
        
        kappa:              constant value in UCB
              
        Returns
        -------
        X: a batch of [x_1..x_Nt]
        """
        y_max = self.Y.max()
        
        # Set parameters if any was passed
        gp=GaussianProcess(self.gp_params)
        
        # Find unique rows of X to avoid GP from breaking
        ur = unique_rows(self.X)
        gp.fit(self.X[ur], self.Y[ur])
        
        # define the number of Thompson sample M
        if 'n_xstars' in self.acq:
            numXtar=self.acq['n_xstars']
        else:
            numXtar=20*self.dim

        if self.xstars==[]:
            xstars=[]
            for ii in range(numXtar):
                mu_acq={}
                mu_acq['name']='thompson'
                mu_acq['dim']=self.dim
                mu_acq['scalebounds']=self.scalebounds     
                acq_mu=AcquisitionFunction(mu_acq)
                xt_TS = acq_max(ac=acq_mu.acq_kind,gp=gp,bounds=self.scalebounds,opt_toolbox='scipy')
                
                #temp.append(xt_TS)
                xstars.append(xt_TS)
        else:
            xstars=self.xstars
            
        # Set acquisition function
        myacq={}
        myacq['name']='pvrs'
        myacq['dim']=self.acq['dim']
        myacq['xstars']=xstars
        acq_func = AcquisitionFunction(myacq)
        
        nRepeat=8
        pred_var=[0]*nRepeat
        bestBatch=[0]*nRepeat
        for tt in range(nRepeat):
            
            # copy GP, X and Y
            temp_gp=copy.deepcopy(gp)
            temp_X=copy.deepcopy(self.X)
            temp_Y=copy.deepcopy(self.Y)
            
            start_batch=time.time()
    
            #store new_x
            if tt==0: # first iteration (repeat) use Greedy approach to fill a batch
                
                if first_batch==[]: # if the first batch is not initialized by greedy
                    new_X=[]
                    for ii in range(B):
                        # Finding argmax of the acquisition function.
                        x_max = acq_max(ac=acq_func.acq_kind,gp=temp_gp, bounds=self.scalebounds)
                        if ii==0:
                            new_X=x_max
                        else:
                            new_X= np.vstack((new_X, x_max.reshape((1, -1))))
                        temp_X = np.vstack((temp_X, x_max.reshape((1, -1))))
                        const_liar,const_liar_variance=temp_gp.predict(x_max,eval_MSE=1)
                        const_liar=np.random.rand()
                        temp_Y = np.append(temp_Y, const_liar )
                        temp_gp.fit(temp_X,temp_Y)
                else:
                    new_X=first_batch
                    #temp_X = np.vstack((temp_X, new_X.reshape((B, -1))))
                    #const_liar,const_liar_variance=temp_gp.predict(new_X,eval_MSE=1)
                    #const_liar=np.random.rand()
                    #temp_Y = np.append(temp_Y, const_liar )
                    #temp_gp.fit(temp_X,temp_Y)
                    
            else:# >=1 iteration
  
                for ii in range(B):                
                    #new_X=new_X.pop(0)
                    temp_X=copy.deepcopy(self.X)

                    if ii==0: # first element
                        temp_X = np.vstack((temp_X, new_X[ii+1:])) # remove item ii  
                    else:
                        if ii==B-1: # last element
                            temp_X = np.vstack((temp_X, new_X[0:ii-1])) # remove item ii  
                        else:
                            #temp_X = np.vstack((temp_X, new_X[0:ii]+new_X[ii+1:])) # remove item ii  
                            temp_X = np.vstack((temp_X, np.vstack((new_X[0:ii],new_X[ii+1:])))) # remove item ii  
   
                    temp_Y,const_liar_variance=temp_gp.predict(temp_X,eval_MSE=1)
                    #temp_Y=np.random.random(size=(len(temp_X),1)) # constant liar
                    temp_gp.fit(temp_X,temp_Y)
    
                    # Finding argmax of the acquisition function.
                    x_max = acq_max_with_init(ac=acq_func.acq_kind,
                                                          gp=temp_gp, y_max=y_max, 
                                                          bounds=self.scalebounds,
                                                          #init_location=np.asarray(new_X[ii]))                    
                                                          init_location=[])                    
                                                                                              
                    previous_var=self.compute_PredictiveVariance(Xstars=xstars,X_t=np.asarray(new_X))
                    
                    # back up old value
                    old_value=new_X[ii].copy()
                                       
                    new_X[ii]=x_max
                    
                    new_var=self.compute_PredictiveVariance(Xstars=xstars,X_t=np.asarray(new_X))

                    if new_var>previous_var: # keep the previous value if the uncertainty does not reduce
                        new_X[ii]=old_value
                        #print "old value"
                        
                    #new_var2=self.compute_PredictiveVariance(Xstars=xstars,X_t=np.asarray(new_X))

                    #print "prev var={:.6f}, newvar={:.6f}, newvar2={:.6f}".format(np.asscalar(previous_var),
                                    #np.asscalar(new_var),np.asscalar(new_var2))


            pred_var[tt]=self.compute_PredictiveVariance(Xstars=xstars,X_t=np.asarray(new_X))
            #print pred_var
            bestBatch[tt]=np.asarray(new_X)
            
            
        #return new_X,new_X_original
        idxBest=np.argmin(pred_var)
        
        new_X=bestBatch[idxBest]
        
        self.NumPoints=np.append(self.NumPoints,new_X.shape[0])

        self.X=np.vstack((self.X,new_X))
        # convert back to original scale
        temp_X_new_original=[val*self.max_min_gap+self.bounds[:,0] for idx, val in enumerate(new_X)]
        temp_X_new_original=np.asarray(temp_X_new_original)
        self.X_original=np.vstack((self.X_original, temp_X_new_original))
        
        # evaluate y=f(x)
        temp=self.f(temp_X_new_original)
        temp=np.reshape(temp,(-1,1))
        self.Y_original=np.append(self.Y_original,temp)
        self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))
        
        
        
        #return bestBatch[idxBest],pred_var[idxBest]
        return bestBatch[idxBest],pred_var
            
    def maximize_batch_greedy_PVRS(self,B=5):
        """
        Finding a batch of points using Peak Suppression / Constant Liar approach
        
        Input Parameters
        ----------

        gp_params:          Parameters to be passed to the Gaussian Process class
        
        kappa:              constant value in UCB
              
        Returns
        -------
        X: a batch of [x_1..x_Nt]
        """
        y_max = self.Y.max()
        
        # Set parameters if any was passed
        self.gp=GaussianProcess(self.gp_params)
        
        # Find unique rows of X to avoid GP from breaking
        ur = unique_rows(self.X)
        self.gp.fit(self.X[ur], self.Y[ur])
        
        start_opt=time.time()        

        if 'n_xstars' in self.acq:
            numXtar=self.acq['n_xstars']
        else:
            numXtar=30*self.dim
        
        #temp=[]
        # finding the xt of Thompson Sampling
        xstars=[]
            
        for ii in range(numXtar):
            mu_acq={}
            mu_acq['name']='thompson'
            mu_acq['dim']=self.dim
            mu_acq['scalebounds']=self.scalebounds     
            acq_mu=AcquisitionFunction(mu_acq)
            xt_TS = acq_max(ac=acq_mu.acq_kind,gp=self.gp,bounds=self.scalebounds,opt_toolbox='scipy')
            
            #temp.append(xt_TS)
            xstars.append(xt_TS)
                
        self.xstars=xstars

                    
        # Set acquisition function
        myacq={}
        myacq['name']='pvrs'
        myacq['dim']=self.acq['dim']
        myacq['xstars']=xstars
        
        acq_func = AcquisitionFunction(myacq)
        
        # copy GP, X and Y
        temp_gp=copy.deepcopy(self.gp)
        temp_X=copy.deepcopy(self.X)
        temp_Y=copy.deepcopy(self.Y)
        #temp_Y_original=self.Y_original
        
        start_batch=time.time()


        # check predictive variance before adding a new data points
        var_before=self.gp.compute_var(temp_X,xstars) 
        var_before=np.mean(var_before)
        
        
        #store new_x
        new_X=np.empty((0,self.dim),float)
        for ii in range(B):
            # Finding argmax of the acquisition function.
            x_max = acq_max(ac=acq_func.acq_kind,gp=temp_gp, bounds=self.scalebounds)
                                  
            new_X= np.vstack((new_X, x_max.reshape((1, -1))))
            
            temp_X = np.vstack((temp_X, x_max.reshape((1, -1))))
            
            
                    
            # check predictive variance after
            var_after=self.gp.compute_var(temp_X,xstars) 
            var_after=np.mean(var_after)
        
            if self.PVRS_before_after==[]:
                self.PVRS_before_after=np.asarray([var_before,var_after])
            else:
                temp_var=np.asarray([var_before,var_after])
                self.PVRS_before_after=np.vstack((self.PVRS_before_after, temp_var))

        
            var_before=var_after
            
            const_liar,const_liar_variance=temp_gp.predict(x_max,eval_MSE=1)
            
            const_liar=np.random.rand()
            temp_Y = np.append(temp_Y, const_liar )
            
            temp_gp.fit(temp_X,temp_Y)
        
        # for debug
        finish_batch=time.time()-start_batch        

        #return new_X,new_X_original
        
        self.NumPoints=np.append(self.NumPoints,new_X.shape[0])

        self.X=np.vstack((self.X,new_X))
        # convert back to original scale
        temp_X_new_original=[val*self.max_min_gap+self.bounds[:,0] for idx, val in enumerate(new_X)]
        temp_X_new_original=np.asarray(temp_X_new_original)
        self.X_original=np.vstack((self.X_original, temp_X_new_original))
        
        # evaluate y=f(x)
        temp=self.f(temp_X_new_original)
        temp=np.reshape(temp,(-1,1))
        self.Y_original=np.append(self.Y_original,temp)
        self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))
        
                # find the maximizer in the GP mean function
        try: 
            len(self.gp)
            x_mu_max=[]
            for j in range(self.J):
                x_mu_max_temp=acq_max_with_name(gp=self.gp[j],scalebounds=self.scalebounds[self.featIdx[j]],acq_name="mu")
                x_mu_max=np.hstack((x_mu_max,x_mu_max_temp))

        except:            
            x_mu_max=acq_max_with_name(gp=self.gp,scalebounds=self.scalebounds,acq_name="mu")

        
        x_mu_max_original=x_mu_max*self.max_min_gap+self.bounds[:,0]
        # set y_max = mu_max
        #mu_max=acq_mu.acq_kind(x_mu_max,gp=self.gp)
        self.Y_original_maxGP = np.append(self.Y_original_maxGP, self.f(x_mu_max_original))
        self.X_original_maxGP = np.vstack((self.X_original_maxGP, x_mu_max_original))
        
        return new_X

