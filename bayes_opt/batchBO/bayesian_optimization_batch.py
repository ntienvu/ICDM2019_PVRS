# -*- coding: utf-8 -*-


from __future__ import division

import sys
sys.path.insert(0,'../../')
sys.path.insert(0,'..')

import numpy as np
#from sklearn.gaussian_process import GaussianProcess
from scipy.optimize import minimize
from prada_bayes_opt.acquisition_functions import AcquisitionFunction, unique_rows
#from prada_bayes_opt import visualization

#from visualization import Visualization
from prada_bayes_opt.prada_gaussian_process import PradaGaussianProcess
#from visualization import *
from prada_bayes_opt.visualization import visualization
from prada_bayes_opt.acquisition_maximization import acq_max
from prada_bayes_opt.acquisition_maximization import *

from sklearn.metrics.pairwise import euclidean_distances
from sklearn import cluster
from sklearn import mixture
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from sklearn import linear_model

import time
import copy


#import nlopt


#======================================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================

class PradaBayOptBatch(object):

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
        
                # Find number of parameters
        pbounds=func_params['bounds']
        bounds=func_params['bounds']
        if 'init_bounds' not in func_params:
            init_bounds=bounds
        else:
            init_bounds=func_params['init_bounds']
            
        # Store the original dictionary
        self.pbounds = bounds

        # Find number of parameters
        self.dim = len(pbounds)

        # Create an array with parameters bounds

        if isinstance(pbounds,dict):
            # Get the name of the parameters
            self.keys = list(pbounds.keys())
        
            self.bounds = []
            for key in self.pbounds.keys():
                self.bounds.append(self.pbounds[key])
            self.bounds = np.asarray(self.bounds)
        else:
            self.bounds=np.asarray(pbounds)

        scalebounds=np.array([np.zeros(self.dim), np.ones(self.dim)])
        self.scalebounds=scalebounds.T
        
        self.max_min_gap=self.bounds[:,1]-self.bounds[:,0]

        # Some function to be optimized
        self.f = func_params['f']
        # optimization toolbox
        if 'opt_toolbox' not in acq_params:
            self.opt_toolbox='scipy'
        else:
            self.opt_toolbox=acq_params['opt_toolbox']
            
            
        if 'optimize_gp' not in acq_params:
            self.optimize_gp=0
        else:                
            self.optimize_gp=acq_params['optimize_gp']
            
        # acquisition function type
        self.acq=acq_params['acq_func']
        
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

        self.gp=PradaGaussianProcess(gp_params)

        # Acquisition Function
        #self.acq_func = None
        self.acq_func = AcquisitionFunction(acq=self.acq)
        self.accum_dist=[]

        # performance evaluation at the maximum mean GP (for information theoretic)
        self.Y_original_maxGP = None
        self.X_original_maxGP = None
        self.nIter=0
        
        
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
        #l=[np.linspace(x[0],x[1],num=n_init_points) for x in self.bounds]


        # Concatenate new random points to possible existing
        # points from self.explore method.
        #self.init_points += list(map(list, zip(*l)))
        temp=np.asarray(l)
        temp=temp.T
        init_X=list(temp.reshape((n_init_points,-1)))

        # Evaluate target function at all initialization           
        y_init=self.f(init_X)

        self.X_original_maxGP= np.asarray(init_X)
        self.Y_original_maxGP=np.asarray(y_init)    
        
        # Turn it into np array and store.
        self.X_original=np.asarray(init_X)
        temp_init_point=np.divide((init_X-self.bounds[:,0]),self.max_min_gap)
        
        self.X_original = np.asarray(init_X)
        self.X = np.asarray(temp_init_point)
        y_init=np.reshape(y_init,(n_init_points,1))
        
        self.Y_original = np.asarray(y_init)
        self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))
        
        self.NumPoints=np.append(self.NumPoints,n_init_points)
        
        # Set parameters if any was passed
        #self.gp=PradaGaussianProcess(gp_params)
        
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
        self.X = np.asarray(temp_init_point)
        
        self.Y_original = np.asarray(init_Y)
        self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))
        
        self.NumPoints=np.append(self.NumPoints,len(init_Y))
        
        # Set acquisition function
        self.acq_func = AcquisitionFunction(self.acq)

        
        # Find unique rows of X to avoid GP from breaking
        ur = unique_rows(self.X)
        self.gp.fit(self.X[ur], self.Y[ur])
   
        
    def smooth_the_peak(self,my_peak):
        
        # define the local bound around the estimated point
        local_bound=np.zeros((self.dim,2))
        for dd in range(self.dim):
            try:
                local_bound[dd,0]=my_peak[-1][dd]-0.005
                local_bound[dd,1]=my_peak[-1][dd]+0.005
            except:
                local_bound[dd,0]=my_peak[dd]-0.005
                local_bound[dd,1]=my_peak[dd]+0.005
                
        local_bound=np.clip(local_bound,self.scalebounds[:,0],self.scalebounds[:,1])
                 
        dim = len(local_bound)
        num_data=1000*dim
        samples = np.zeros(shape=(num_data,dim))
        #for k in range(0,dim): samples[:,k] = np.random.uniform(low=local_bound[k][0],high=local_bound[k][1],size=num_data)
        for dd in range(0,dim): samples[:,dd] = np.linspace(local_bound[dd][0],local_bound[dd][1],num_data)

        # smooth the peak
        """
        n_bins =  100*np.ones(self.dim)
        mygrid = np.mgrid[[slice(row[0], row[1], n*1j) for row, n in zip(local_bound, n_bins)]]
        mygrid=mygrid.reshape(100**self.dim, self.dim)
        utility_grid=self.acq_func.acq_kind(mygrid,self.gp,self.Y.max())        
        
        mysamples=np.vstack((mygrid,utility_grid))
        samples_smooth=filters.uniform_filter(mysamples, size=[2,2], output=None, mode='reflect', cval=0.0, origin=0)
        """

        # get the utility after smoothing
        samples_smooth=samples
        utility_smooth=self.acq_func.acq_kind(samples_smooth,self.gp,self.Y.max()) 
        
        # get the peak value y
        #peak_y=np.max(utility_smooth)
        
        # get the peak location x
        #peak_x=samples_smooth[np.argmax(utility_smooth)]        
        
        peak_x=my_peak
        # linear regression
        regr = linear_model.LinearRegression()

        regr.fit(samples_smooth, utility_smooth)
        #residual_ss=np.mean((regr.predict(samples_smooth) - utility_smooth) ** 2)
        mystd=np.std(utility_smooth)

        return peak_x,mystd

    def check_real_peak(self,my_peak,threshold=0.1):
        
        # define the local bound around the estimated point
        local_bound=np.zeros((self.dim,2))
        for dd in range(self.dim):
            try:
                local_bound[dd,0]=my_peak[-1][dd]-0.01
                local_bound[dd,1]=my_peak[-1][dd]+0.01
            except:
                local_bound[dd,0]=my_peak[dd]-0.01
                local_bound[dd,1]=my_peak[dd]+0.01
                
        #local_bound=np.clip(local_bound,self.scalebounds[:,0],self.scalebounds[:,1])
        local_bound[:,0]=local_bound[:,0].clip(self.scalebounds[:,0],self.scalebounds[:,1])
        local_bound[:,1]=local_bound[:,1].clip(self.scalebounds[:,0],self.scalebounds[:,1])
                 
        dim = len(local_bound)
        num_data=100*dim
        samples = np.zeros(shape=(num_data,dim))
        for dd in range(0,dim): samples[:,dd] = np.linspace(local_bound[dd][0],local_bound[dd][1],num_data)

        # get the utility after smoothing
        myutility=self.acq_func.acq_kind(samples,self.gp,self.Y.max()) 
        
        # linear regression
        #regr = linear_model.LinearRegression()
        #regr.fit(samples, myutility)
        #residual_ss=np.mean((regr.predict(samples_smooth) - utility_smooth) ** 2)
        
        #mystd=np.std(myutility)
        mystd=np.mean(myutility)

        IsPeak=0
        if mystd>threshold/(self.dim**2):
            IsPeak=1
        return IsPeak,mystd
        
    def estimate_L(self,bounds):
        '''
        Estimate the Lipschitz constant of f by taking maximizing the norm of the expectation of the gradient of *f*.
        '''
        def df(x,model,x0):
            mean_derivative=gp_model.predictive_gradient(self.X,self.Y,x)
            
            temp=mean_derivative*mean_derivative
            if len(temp.shape)<=1:
                res = np.sqrt( temp)
            else:
                res = np.sqrt(np.sum(temp,axis=1)) # simply take the norm of the expectation of the gradient        
            return -res

        gp_model=self.gp
                
        dim = len(bounds)
        num_data=1000*dim
        samples = np.zeros(shape=(num_data,dim))
        for k in range(0,dim): samples[:,k] = np.random.uniform(low=bounds[k][0],high=bounds[k][1],size=num_data)

        #samples = np.vstack([samples,gp_model.X])
        pred_samples = df(samples,gp_model,0)
        x0 = samples[np.argmin(pred_samples)]

        res = minimize(df,x0, method='L-BFGS-B',bounds=bounds, args = (gp_model,x0), options = {'maxiter': 100})
        
        try:
            minusL = res.fun[0][0]
        except:
            if len(res.fun.shape)==1:
                minusL = res.fun[0]
            else:
                minusL = res.fun
                
        L=-minusL
        if L<1e-6: L=0.0001  ## to avoid problems in cases in which the model is flat.
        return L  
        
    def maximize_batch_PS(self,gp_params,B=5, kappa=2):
        """
        Finding a batch of points using Peak Suppression approach
        
        Input Parameters
        ----------

        gp_params:          Parameters to be passed to the Gaussian Process class
        
        kappa:              constant value in UCB
              
        Returns
        -------
        X: a batch of [x_1..x_Nt]
        """
                
        const_liar=self.Y_original.min()
        
        # Set acquisition function
        #self.acq_func = AcquisitionFunction(kind=self.acq, kappa=kappa)
        
        y_max = self.Y.max()
        
        # Set parameters if any was passed
        self.gp=PradaGaussianProcess(gp_params)
        
        # Find unique rows of X to avoid GP from breaking
        ur = unique_rows(self.X)
        self.gp.fit(self.X[ur], self.Y[ur])
        
        start_opt=time.time()        

        # copy GP, X and Y
        temp_gp=self.gp
        temp_X=self.X
        temp_Y=self.Y
        
        #store new_x
        new_X=[]
        stdPeak=[0]*B
        IsPeak=[0]*B
        for ii in range(B):
        
            # Finding argmax of the acquisition function.
            x_max = acq_max(ac=self.acq_func.acq_kind,gp=temp_gp, y_max=y_max, bounds=self.scalebounds,opt_toolbox=self.opt_toolbox)

            # Test if x_max is repeated, if it is, draw another one at random
            if np.any((np.abs(temp_X - x_max)).sum(axis=1) <0.002*self.dim) | np.isnan(x_max.sum()):
                #x_max = np.random.uniform(self.scalebounds[:, 0], self.scalebounds[:, 1],size=self.scalebounds.shape[0])
                                          
                IsPeak[ii]=0
                stdPeak[ii]=0
                print("reject")
            else:
                IsPeak[ii],stdPeak[ii]=self.check_real_peak(x_max)               
           
            #print "IsPeak={:d} std={:.5f}".format(IsPeak[ii],stdPeak[ii])
                                                                 
            if ii==0:
                new_X=x_max
            else:
                new_X= np.vstack((new_X, x_max.reshape((1, -1))))
                
            temp_X = np.vstack((temp_X, x_max.reshape((1, -1))))
            temp_Y = np.append(temp_Y, const_liar )
            
            #temp_gp.fit(temp_X,temp_Y)
            temp_gp.fit_incremental(x_max, np.asarray([const_liar]))

            """
            toplot_bo=copy.deepcopy(self)
            toplot_bo.gp=copy.deepcopy(temp_gp)
            toplot_bo.X=temp_X
            toplot_bo.X_original=[val*self.max_min_gap+self.bounds[:,0] for idx, val in enumerate(temp_X)]
            toplot_bo.X_original=np.asarray(toplot_bo.X_original)
            toplot_bo.Y=temp_Y
            toplot_bo.Y_original=temp_Y*(np.max(self.Y_original)-np.min(self.Y_original))+np.mean(self.Y_original)
            visualization.plot_bo(toplot_bo)
            """

            
        IsPeak=np.asarray(IsPeak)

        # check if there is no real peak, then pick up the top peak (highest std)

        # rank the peak
        idx=np.sort(stdPeak)


        if np.sum(IsPeak)==0:
            top_peak=np.argmax(stdPeak)
            new_X=new_X[top_peak]
        else:
            new_X=new_X[IsPeak==1]
            
        print(new_X)

        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        self.opt_time=np.hstack((self.opt_time,elapse_opt))

        # Updating the GP.
        #new_X=new_X.reshape((-1, self.dim))

        # Test if x_max is repeated, if it is, draw another one at random
        temp_new_X=[]
        for idx,val in enumerate(new_X):
            if np.all(np.any(np.abs(self.X-val)>0.02,axis=1)): # check if a data point is already taken
                temp_new_X=np.append(temp_new_X,val)
                
        if len(temp_new_X)==0:
            temp_new_X=np.zeros((1,self.dim))
            for idx in range(0,self.dim):
                temp_new_X[0,idx]=np.random.uniform(self.scalebounds[idx,0],self.scalebounds[idx,1],1)
        else:
            temp_new_X=temp_new_X.reshape((-1,self.dim))
         

        self.X=np.vstack((self.X, temp_new_X))
        
        # convert back to original scale
        temp_X_new_original=[val*self.max_min_gap+self.bounds[:,0] for idx, val in enumerate(temp_new_X)]
        temp_X_new_original=np.asarray(temp_X_new_original)
        self.X_original=np.vstack((self.X_original, temp_X_new_original))
        
        for idx,val in enumerate(temp_X_new_original):
            self.Y_original = np.append(self.Y_original, self.f(val))

        # update Y after change Y_original
        self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))
        self.NumPoints=np.append(self.NumPoints,temp_X_new_original.shape[0])

        
        #print "#Batch={:d} f_max={:.4f}".format(temp_X_new_original.shape[0],self.Y_original.max())
        
    def maximize_batch_CL(self,gp_params,B=5):
        """
        Finding a batch of points using Constant Liar approach
        
        Input Parameters
        ----------

        gp_params:          Parameters to be passed to the Gaussian Process class
        
        kappa:              constant value in UCB
              
        Returns
        -------
        X: a batch of [x_1..x_Nt]
        """
        
        
        if self.acq['name']=='random':
            x_max = [np.random.uniform(x[0], x[1], size=B) for x in self.bounds]
            x_max=np.asarray(x_max)
            x_max=x_max.T
            self.X_original=np.vstack((self.X_original, x_max))
            # evaluate Y using original X
            
            #self.Y = np.append(self.Y, self.f(temp_X_new_original))
            self.Y_original = np.append(self.Y_original, self.f(x_max))
            
            # update Y after change Y_original
            self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))
            
            self.opt_time=np.hstack((self.opt_time,0))
            return     
            
        #const_liar=self.Y.mean()
        #const_liar=self.Y_original.mean()
        #const_liar=self.Y.max()
        
        # Set acquisition function
        self.acq_func = AcquisitionFunction(self.acq)
        
        # Set parameters if any was passed
        self.gp=PradaGaussianProcess(gp_params)
        
        if len(self.gp.KK_x_x_inv)==0: # check if empty
            self.gp.fit(self.X, self.Y)
        #else:
            #self.gp.fit_incremental(self.X[ur], self.Y[ur])
                
        # optimize GP parameters after 10 iterations
        if self.nIter%10==0:
            if self.optimize_gp=='maximize':
                newlengthscale = self.gp.optimize_lengthscale_SE_maximizing(gp_params['lengthscale'],gp_params['noise_delta'])
                gp_params['lengthscale']=newlengthscale
                #print "estimated lengthscale ={:s}".format(newlengthscale)
            elif self.optimize_gp=='loo':
                newlengthscale = self.gp.optimize_lengthscale_SE_loo(gp_params['lengthscale'],gp_params['noise_delta'])
                gp_params['lengthscale']=newlengthscale
                #print "estimated lengthscale ={:s}".format(newlengthscale)

            elif self.optimize_gp=='marginal':
                self.theta_vector = self.gp.slice_sampling_lengthscale_SE(gp_params['lengthscale'],gp_params['noise_delta'])
                gp_params['lengthscale']=self.theta_vector[0]
                self.theta_vector =np.unique(self.theta_vector)
                gp_params['newtheta_vector']=self.theta_vector 
                #print "estimated lengthscale ={:s}".format(self.theta_vector)
                
            # init a new Gaussian Process after optimizing hyper-parameter
            self.gp=PradaGaussianProcess(gp_params)
            # Find unique rows of X to avoid GP from breaking
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur])
        
        start_opt=time.time()        


        # copy GP, X and Y
        temp_gp=self.gp
        temp_X=self.X
        temp_Y=self.Y
        #temp_Y_original=self.Y_original
        
        
        start_batch=time.time()

        #store new_x
        new_X=[]
        for ii in range(B):
        
            # Finding argmax of the acquisition function.
            x_max = acq_max(ac=self.acq_func.acq_kind,gp=temp_gp,  bounds=self.scalebounds)
            #val_acq=self.acq_func.acq_kind(x_max,temp_gp,y_max)
            #print "CL alpha[x_max]={:.5f}".format(np.ravel(val_acq)[0])

            # Test if x_max is repeated, if it is, draw another one at random
            # If it is repeated, print a warning
            if np.any(np.abs((self.X - x_max)).sum(axis=1) == 0)| np.isnan(x_max.sum()):
                #x_max = np.random.uniform(self.scalebounds[:, 0], self.scalebounds[:, 1],size=self.scalebounds.shape[0])
                break
                   
                                  
            if ii==0:
                new_X=x_max
            else:
                new_X= np.vstack((new_X, x_max.reshape((1, -1))))
            
            temp_X = np.vstack((temp_X, x_max.reshape((1, -1))))
            
            const_liar,const_liar_variance=temp_gp.predict(x_max,eval_MSE=1)
            temp_Y = np.append(temp_Y, const_liar )
            
            temp_gp.fit(temp_X,temp_Y)
        
        # for debug
        finish_batch=time.time()-start_batch        
        #print "batchB={:.3f}".format(finish_batch)
             
        # Updating the GP.
        #new_X=new_X.reshape((-1, self.acq['dim']))

        self.NumPoints=np.append(self.NumPoints,len(new_X))


        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        self.opt_time=np.hstack((self.opt_time,elapse_opt))
        
        #print new_X
        self.X=np.vstack((self.X, new_X))

        
        # convert back to original scale
        temp_X_new_original=[val*self.max_min_gap+self.bounds[:,0] for idx, val in enumerate(new_X)]
        temp_X_new_original=np.asarray(temp_X_new_original)
        self.X_original=np.vstack((self.X_original, temp_X_new_original))
        
        for idx,val in enumerate(temp_X_new_original):
            self.Y_original = np.append(self.Y_original, self.f(val))

        # update Y after change Y_original
        self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))
        #print "#Batch={:d} f_max={:.4f}".format(B,self.Y_original.max())
        

        #    find the maximizer in the GP mean function
        mu_acq={}
        mu_acq['name']='mu'
        mu_acq['dim']=self.dim
        acq_mu=AcquisitionFunction(mu_acq)
        x_mu_max = acq_max(ac=acq_mu.acq_kind,gp=self.gp,bounds=self.scalebounds,opt_toolbox=self.opt_toolbox)
        
        x_mu_max_original=x_mu_max*self.max_min_gap+self.bounds[:,0]
        
        y_mu_max_original=self.f(x_mu_max_original)
        
        temp_y=[y_mu_max_original]*(B)
        temp_x=[x_mu_max_original]*B
        # set y_max = mu_max
        #mu_max=acq_mu.acq_kind(x_mu_max,gp=self.gp)
        self.Y_original_maxGP = np.append(self.Y_original_maxGP, temp_y)
        self.X_original_maxGP = np.vstack((self.X_original_maxGP, np.asarray(temp_x)))
        
        self.nIter=self.nIter+1
        
        return new_X,temp_X_new_original
        
        
    def maximize_batch_CL_incremental(self,gp_params,B=5):
        """
        Finding a batch of points using Constant Liar approach
        
        Input Parameters
        ----------

        gp_params:          Parameters to be passed to the Gaussian Process class
        
        kappa:              constant value in UCB
              
        Returns
        -------
        X: a batch of [x_1..x_Nt]
        """
        
        self.NumPoints=np.append(self.NumPoints,B)
        
        if self.acq['name']=='random':
            x_max = [np.random.uniform(x[0], x[1], size=B) for x in self.bounds]
            x_max=np.asarray(x_max)
            x_max=x_max.T
            self.X_original=np.vstack((self.X_original, x_max))
            # evaluate Y using original X
            
            #self.Y = np.append(self.Y, self.f(temp_X_new_original))
            self.Y_original = np.append(self.Y_original, self.f(x_max))
            
            # update Y after change Y_original
            self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))
            
            self.opt_time=np.hstack((self.opt_time,0))
            return     
            
        #const_liar=self.Y.mean()
        #const_liar=self.Y_original.min()
        #const_liar=self.Y.max()
        
        # Set acquisition function
        self.acq_func = AcquisitionFunction(self.acq)
        
        y_max = self.Y.max()
        
        # Set parameters if any was passed
        self.gp=PradaGaussianProcess(gp_params)
        
        # Find unique rows of X to avoid GP from breaking
        ur = unique_rows(self.X)
        self.gp.fit(self.X[ur], self.Y[ur])
        
        start_opt=time.time()        


        # copy GP, X and Y
        temp_gp=copy.deepcopy(self.gp)
        temp_X=self.X.copy()
        temp_Y=self.Y.copy()
        #temp_Y_original=self.Y_original
                
        #store new_x
        new_X=[]
        for ii in range(B):
        
            # Finding argmax of the acquisition function.
            x_max = acq_max(ac=self.acq_func.acq_kind,gp=temp_gp, y_max=y_max, bounds=self.scalebounds)
            
            # Test if x_max is repeated, if it is, draw another one at random
            if np.any(np.abs((self.X - x_max)).sum(axis=1) == 0)| np.isnan(x_max.sum()):
                print("the same location - the batch is terminanted!")
                #x_max = np.random.uniform(self.scalebounds[:, 0], self.scalebounds[:, 1],
                                          #size=self.scalebounds.shape[0]) 
                break
            
            if ii==0:
                new_X=x_max
            else:
                new_X= np.vstack((new_X, x_max.reshape((1, -1))))  
            
            const_liar,const_liar_variance=temp_gp.predict(x_max)

            #temp_X= np.vstack((temp_X, x_max.reshape((1, -1))))
            #temp_Y = np.append(temp_Y, const_liar )
            
            #temp_gp.fit(temp_X,temp_Y)
            
            # update the Gaussian Process and thus the acquisition function                         
            #temp_gp.compute_incremental_cov_matrix(temp_X,x_max)
            temp_gp.fit_incremental(x_max,np.asarray([const_liar]))
            
         
        # Updating the GP.
        new_X=new_X.reshape((-1,self.acq['dim']))

        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        self.opt_time=np.hstack((self.opt_time,elapse_opt))
        
        #print new_X
        
        self.X=np.vstack((self.X, new_X))
        
        
        # convert back to original scale
        temp_X_new_original=[val*self.max_min_gap+self.bounds[:,0] for idx, val in enumerate(new_X)]
        temp_X_new_original=np.asarray(temp_X_new_original)
        self.X_original=np.vstack((self.X_original, temp_X_new_original))
        
        for idx,val in enumerate(temp_X_new_original):
            self.Y_original = np.append(self.Y_original, self.f(val))

        # update Y after change Y_original
        self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))
        #print "#Batch={:d} f_max={:.4f}".format(B,self.Y_original.max())
 
#======================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================

    def maximize_batch_BUCB_incremental(self,gp_params, B=5):
        """
        Finding a batch of points using GP-BUCB approach
        
        Input Parameters
        ----------

        gp_params:          Parameters to be passed to the Gaussian Process class
        
        B:                  fixed batch size for all iteration
        
        kappa:              constant value in UCB
        
        IsPlot:             flag variable for visualization    
        
        
        Returns
        -------
        X: a batch of [x_1..x_B]
        """        
        self.B=B
                
        # Set acquisition function
        self.acq_func = AcquisitionFunction(self.acq)
               
        # Set parameters if any was passed
        self.gp=PradaGaussianProcess(gp_params)
        
        if len(self.gp.KK_x_x_inv)==0: # check if empty
            self.gp.fit(self.X, self.Y)
        #else:
            #self.gp.fit_incremental(self.X[ur], self.Y[ur])
        
        start_opt=time.time()
       
        y_max=self.gp.Y.max()
        # check the bound 0-1 or original bound        
        temp_X=self.X.copy()
        temp_gp=copy.deepcopy(self.gp)
        temp_gp.X_bucb=temp_X.copy()
        temp_gp.KK_x_x_inv_bucb=copy.deepcopy(self.gp.KK_x_x_inv)
        
        # finding new X
        # finding a first point using the original acquisition function (UCB, EI)
        x_max_first = acq_max(ac=self.acq_func.acq_kind, gp=self.gp, y_max=y_max, bounds=self.scalebounds)
        acq_value=self.acq_func.acq_kind(x_max_first,gp=temp_gp, y_max=y_max)
        #print "ucb(x_max)={:.2f}".format(acq_value[0][0])
               
               
        new_batch_X=[]
        new_batch_X.append(x_max_first) # append the first point
        
        # create BUCB acquisition function
        bucb_acq={}
        bucb_acq['name']='bucb_incremental'
        bucb_acq['dim']=self.dim
        if 'kappa' not in self.acq:
            bucb_acq['kappa']=2
        else:
            bucb_acq['kappa']=self.acq['kappa']

        acq_bucb=AcquisitionFunction(bucb_acq)
        
        temp_X=np.vstack((temp_X,x_max_first))
        temp_gp.X_bucb=temp_X.copy()
        # update the Gaussian Process and thus the acquisition function                         
        temp_gp.compute_incremental_cov_matrix(temp_X,x_max_first)

        for ii in range(B-1):
            # Finding argmax of the acquisition function.
            x_max = acq_max(ac=acq_bucb.acq_kind, gp=temp_gp, y_max=y_max, bounds=self.scalebounds)

            #print the max value
            acq_value=acq_bucb.acq_kind(x_max,gp=temp_gp, y_max=y_max)
            #print "bucb sigma(x_max)={:.2f}".format(acq_value[0][0])
            if np.any(np.abs((temp_X - x_max)).sum(axis=1) == 0) | np.isnan(x_max.sum()):
                #x_max = np.random.uniform(self.scalebounds[:, 0],self.scalebounds[:, 1],size=self.scalebounds.shape[0])
                #print "the same location - the batch is terminanted!"
                break

            new_batch_X= np.vstack((new_batch_X, x_max.reshape((1, -1))))
                            
            # update the Gaussian Process and thus the acquisition function                         
            temp_gp.compute_incremental_cov_matrix(temp_X,x_max)

            temp_X = np.vstack((temp_X, x_max.reshape((1, -1))))
            temp_gp.X_bucb=temp_X
        
        
        # record the optimization time
        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        
        self.opt_time=np.hstack((self.opt_time,elapse_opt))
        self.NumPoints=np.append(self.NumPoints,len(new_batch_X))
        self.X=temp_X
                    
        # convert back to original scale
        temp_X_new_original=[val*self.max_min_gap+self.bounds[:,0] for idx, val in enumerate(new_batch_X)]
        temp_X_new_original=np.asarray(temp_X_new_original)
        self.X_original=np.vstack((self.X_original, temp_X_new_original))
        
        # evaluate y=f(x)
        temp=self.f(temp_X_new_original)
        temp=np.reshape(temp,(-1,1))
        self.Y_original=np.append(self.Y_original,temp)
        self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))
        #print "#Batch={:d} f_max={:.4f}".format(new_X.shape[0],self.Y_original.max())

    def maximize_batch_BUCB_iterative(self,gp_params, B=5):
        """
        Finding a batch of points using Geometric approach
        
        Input Parameters
        ----------

        gp_params:          Parameters to be passed to the Gaussian Process class
        
        B:                  fixed batch size for all iteration
        
        Returns
        -------
        X: a batch of [x_1..x_B]
        """        

        # create Geometric acquisition function
        bucb_acq={}
        bucb_acq['name']='bucb'
        bucb_acq['kappa']=self.acq['kappa']

        bucb_acq['dim']=self.acq['dim']
        acq_bucb=AcquisitionFunction(bucb_acq)
        
        self.B=B
        # Set acquisition function
        self.acq_func = AcquisitionFunction(self.acq)
               
        # Set parameters if any was passed
        self.gp=PradaGaussianProcess(gp_params)
        
        if len(self.gp.KK_x_x_inv)==0: # check if empty
            self.gp.fit(self.X, self.Y)
        #else:
            #self.gp.fit_incremental(self.X[ur], self.Y[ur])
                
        # optimize GP parameters after 5 iterations
        if self.optimize_gp==1 and len(self.Y)>=12*self.dim and len(self.Y)%10*self.dim==0:
            newtheta = self.gp.optimize_lengthscale(gp_params['theta'],gp_params['noise_delta'])
            gp_params['theta']=newtheta
            logmarginal=self.gp.log_marginal_lengthscale(newtheta,gp_params['noise_delta'])
            
        start_gmm_opt=time.time()       
        y_max=self.gp.Y.max()
        # check the bound 0-1 or original bound        
        temp_X=self.X.copy()
        temp_gp=copy.deepcopy(self.gp ) 
        temp_gp.X_bucb=temp_X.copy()
        
        start_first_x=time.time()
        # finding a first point using the original acquisition function (UCB, EI)
        x_max_first = acq_max(ac=self.acq_func.acq_kind, gp=temp_gp, y_max=y_max, bounds=self.scalebounds)
        
        #acq_value=self.acq_func.acq_kind(x_max_first,gp=temp_gp, y_max=y_max)
        #print "ucb(x_max)={:.2f}".format(acq_value[0][0])
        
        end_first_x=time.time()-start_first_x
        
        #new_batch_X=[]
        #new_batch_X.append(x_max_first) # append the first point

        # compute accummulate sigma

        start_batch=time.time()
        # finding B-1 points using Geometric acquisition function
        
        nRepeat=10
        for tt in range (nRepeat):
            temp_X=self.X.copy()
            temp_gp=copy.deepcopy(self.gp ) 
            temp_X=np.vstack((temp_X,x_max_first))
            
            #store new_x
            if tt==0:
                new_batch_X=[0]*B
                new_batch_X[0]=x_max_first
        
            for ii in range(B): # from ii=2 : B
                
                if ii==0:
                    continue
                
                if tt>0:
                    temp_X=copy.deepcopy(self.X)
                    temp_X = np.vstack((temp_X, new_batch_X[0:ii]+new_batch_X[ii+1:])) # remove item ii
                else:
                    if ii>1:
                        temp_X = np.vstack((temp_X, x_max)) # remove item ii
                        
                temp_gp.X_bucb=temp_X.copy() # instead of fitting both X and Y, we consider only X
                temp_Y=np.random.random(size=(len(temp_X),1)) # constant liar
                temp_gp.fit(temp_X,temp_Y)

                # Finding argmax of the acquisition function.
                x_max = acq_max(ac=acq_bucb.acq_kind, gp=temp_gp, y_max=y_max, bounds=self.scalebounds)
    
                #acq_value=acq_bucb.acq_kind(x_max,gp=temp_gp, y_max=y_max)
                #self.accum_dist.append(acq_value)
    
                #print "geo(x_max)={:.2f}".format(acq_value[0])
            
                if np.any(np.abs((temp_X - x_max)).sum(axis=1) == 0) | np.isnan(x_max.sum()):
                    #x_max = np.random.uniform(self.scalebounds[:, 0],self.scalebounds[:, 1],size=self.scalebounds.shape[0])
                    print("the same location - the batch is terminated!")
                    break
                                              
                #new_batch_X= np.vstack((new_batch_X, x_max.reshape((1, -1))))
                new_batch_X[ii]=x_max
                
                #temp_X = np.vstack((temp_X, x_max.reshape((1, -1)))) # to update temp_gp
                #temp_gp.X=temp_X

        #debug
        finish_batch=time.time()-start_batch
        #print "first={:.3f}, batchB_1={:.3f}".format(end_first_x,finish_batch)

        # record the optimization time
        finished_gmm_opt=time.time()
        elapse_gmm_opt=finished_gmm_opt-start_gmm_opt
        
        self.opt_time=np.hstack((self.opt_time,elapse_gmm_opt))

        self.NumPoints=np.append(self.NumPoints,len(new_batch_X))

        #new_batch_X=new_batch_X.reshape((-1,self.acq['dim']))
        new_batch_X=np.asarray(new_batch_X)
        new_batch_X=new_batch_X.reshape(-1,self.dim)
        self.X=np.vstack((self.X,new_batch_X))
                    
        # convert back to original scale
        temp_X_new_original=[val*self.max_min_gap+self.bounds[:,0] for idx, val in enumerate(new_batch_X)]
        temp_X_new_original=np.asarray(temp_X_new_original)
        self.X_original=np.vstack((self.X_original, temp_X_new_original))
        
        # evaluate y=f(x)
        temp=self.f(temp_X_new_original)
        temp=np.reshape(temp,(-1,1))
        self.Y_original=np.append(self.Y_original,temp)
        self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))
        #print "#Batch={:d} f_max={:.4f}".format(new_batch_X.shape[0],self.Y_original.max())
        
        self.gp=temp_gp # for visualization
        
        
    def maximize_batch_BUCB(self,gp_params, B=5):
        """
        Finding a batch of points using GP-BUCB approach
        
        Input Parameters
        ----------

        gp_params:          Parameters to be passed to the Gaussian Process class
        
        B:                  fixed batch size for all iteration
        
        kappa:              constant value in UCB
        
        IsPlot:             flag variable for visualization    
        
        
        Returns
        -------
        X: a batch of [x_1..x_B]
        """        
        self.B=B
                
        # Set acquisition function
        self.acq_func = AcquisitionFunction(self.acq)
               
        # Set parameters if any was passed
        self.gp=PradaGaussianProcess(gp_params)
        
        if len(self.gp.KK_x_x_inv)==0: # check if empty
            self.gp.fit(self.X, self.Y)
        #else:
            #self.gp.fit_incremental(self.X[ur], self.Y[ur])
        
        
        # optimize GP parameters after 10 iterations
        if self.nIter%10==0:
            if self.optimize_gp=='maximize':
                newlengthscale = self.gp.optimize_lengthscale_SE_maximizing(gp_params['lengthscale'],gp_params['noise_delta'])
                gp_params['lengthscale']=newlengthscale
                #print "estimated lengthscale ={:s}".format(newlengthscale)
            elif self.optimize_gp=='loo':
                newlengthscale = self.gp.optimize_lengthscale_SE_loo(gp_params['lengthscale'],gp_params['noise_delta'])
                gp_params['lengthscale']=newlengthscale
                #print "estimated lengthscale ={:s}".format(newlengthscale)

            elif self.optimize_gp=='marginal':
                self.theta_vector = self.gp.slice_sampling_lengthscale_SE(gp_params['lengthscale'],gp_params['noise_delta'])
                gp_params['lengthscale']=self.theta_vector[0]
                self.theta_vector =np.unique(self.theta_vector)
                gp_params['newtheta_vector']=self.theta_vector 
                #print "estimated lengthscale ={:s}".format(self.theta_vector)
                
            # init a new Gaussian Process after optimizing hyper-parameter
            self.gp=PradaGaussianProcess(gp_params)
            # Find unique rows of X to avoid GP from breaking
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur])
            
            
        start_opt=time.time()
       
        y_max=self.gp.Y.max()
        # check the bound 0-1 or original bound        
        temp_X=self.X.copy()
        temp_gp=copy.deepcopy(self.gp)
        temp_gp.X_bucb=temp_X.copy()
        temp_gp.KK_x_x_inv_bucb=copy.deepcopy(self.gp.KK_x_x_inv)
        
        # finding new X
        # finding a first point using the original acquisition function (UCB, EI)
        x_max_first = acq_max(ac=self.acq_func.acq_kind, gp=temp_gp, bounds=self.scalebounds)
        #acq_value=self.acq_func.acq_kind(x_max_first,gp=temp_gp, y_max=y_max)
        #print "ucb(x_max)={:.2f}".format(acq_value[0][0])
               
        new_batch_X=[]
        new_batch_X.append(x_max_first) # append the first point
        
        # create BUCB acquisition function
        bucb_acq={}
        bucb_acq['name']='bucb'
        bucb_acq['dim']=self.dim
        if 'kappa' not in self.acq:
            bucb_acq['kappa']=2
        else:
            bucb_acq['kappa']=self.acq['kappa']

        acq_bucb=AcquisitionFunction(bucb_acq)
        
        temp_X=np.vstack((temp_X,x_max_first))
        temp_gp.X_bucb=temp_X.copy()
        
        
        for ii in range(B-1):
            # Finding argmax of the acquisition function.
            x_max = acq_max(ac=acq_bucb.acq_kind, gp=temp_gp, bounds=self.scalebounds)

            #print the max value
            #acq_value=acq_bucb.acq_kind(x_max,gp=temp_gp, y_max=y_max)
            #print "bucb sigma(x_max)={:.2f}".format(acq_value[0][0])
            #if np.any(np.abs((temp_X - x_max)).sum(axis=1) == 0) | np.isnan(x_max.sum()):
                #x_max = np.random.uniform(self.scalebounds[:, 0],self.scalebounds[:, 1],size=self.scalebounds.shape[0])
                #print "the same location - the batch is terminanted!"
                #break

            new_batch_X= np.vstack((new_batch_X, x_max.reshape((1, -1))))
                            
            # update the Gaussian Process and thus the acquisition function                         
            #temp_gp.compute_incremental_cov_matrix(temp_X,x_max)

            temp_X = np.vstack((temp_X, x_max.reshape((1, -1))))
            temp_gp.X_bucb=temp_X
        
        
        # record the optimization time
        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        
        self.opt_time=np.hstack((self.opt_time,elapse_opt))
        self.NumPoints=np.append(self.NumPoints,len(new_batch_X))
        self.X=temp_X
                    
        # convert back to original scale
        temp_X_new_original=[val*self.max_min_gap+self.bounds[:,0] for idx, val in enumerate(new_batch_X)]
        temp_X_new_original=np.asarray(temp_X_new_original)
        self.X_original=np.vstack((self.X_original, temp_X_new_original))
        
        # evaluate y=f(x)
        temp=self.f(temp_X_new_original)
        temp=np.reshape(temp,(-1,1))
        self.Y_original=np.append(self.Y_original,temp)
        self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))
        #print "#Batch={:d} f_max={:.4f}".format(new_X.shape[0],self.Y_original.max())

        self.gp=temp_gp
        
        # find the maximizer in the GP mean function
        mu_acq={}
        mu_acq['name']='mu'
        mu_acq['dim']=self.dim
        acq_mu=AcquisitionFunction(mu_acq)
        x_mu_max = acq_max(ac=acq_mu.acq_kind,gp=self.gp,bounds=self.scalebounds,opt_toolbox=self.opt_toolbox)
        
        x_mu_max_original=x_mu_max*self.max_min_gap+self.bounds[:,0]
        
        y_mu_max_original=self.f(x_mu_max_original)
        
        temp_y=[y_mu_max_original]*(B)
        temp_x=[x_mu_max_original]*B
        # set y_max = mu_max
        #mu_max=acq_mu.acq_kind(x_mu_max,gp=self.gp)
        self.Y_original_maxGP = np.append(self.Y_original_maxGP, temp_y)
        self.X_original_maxGP = np.vstack((self.X_original_maxGP, np.asarray(temp_x)))
        
        self.nIter=self.nIter+1

    def maximize_batch_UCB_PE(self,gp_params, B=5):
        """
        Finding a batch of points using GP-BUCB-PE approach
        
        Input Parameters
        ----------

        gp_params:          Parameters to be passed to the Gaussian Process class
        
        B:                  fixed batch size for all iteration      
                 
        Returns
        -------
        X: a batch of [x_1..x_B]
        """        
        self.B=B
                
        # Set acquisition function
        self.acq_func = AcquisitionFunction(self.acq)
               
        # Set parameters if any was passed
        self.gp=PradaGaussianProcess(gp_params)
        
        if len(self.gp.KK_x_x_inv)==0: # check if empty
            self.gp.fit(self.X, self.Y)
        #else:
            #self.gp.fit_incremental(self.X[ur], self.Y[ur])
        
        
        # optimize GP parameters after 10 iterations
        if self.nIter%10==0:
            if self.optimize_gp=='maximize':
                newlengthscale = self.gp.optimize_lengthscale_SE_maximizing(gp_params['lengthscale'],gp_params['noise_delta'])
                gp_params['lengthscale']=newlengthscale
                #print "estimated lengthscale ={:s}".format(newlengthscale)
            elif self.optimize_gp=='loo':
                newlengthscale = self.gp.optimize_lengthscale_SE_loo(gp_params['lengthscale'],gp_params['noise_delta'])
                gp_params['lengthscale']=newlengthscale
                #print "estimated lengthscale ={:s}".format(newlengthscale)

            elif self.optimize_gp=='marginal':
                self.theta_vector = self.gp.slice_sampling_lengthscale_SE(gp_params['lengthscale'],gp_params['noise_delta'])
                gp_params['lengthscale']=self.theta_vector[0]
                self.theta_vector =np.unique(self.theta_vector)
                gp_params['newtheta_vector']=self.theta_vector 
                #print "estimated lengthscale ={:s}".format(self.theta_vector)
                
            # init a new Gaussian Process after optimizing hyper-parameter
            self.gp=PradaGaussianProcess(gp_params)
            # Find unique rows of X to avoid GP from breaking
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur])
            
        start_gmm_opt=time.time()
       
        y_max=self.gp.Y.max()
        
        # check the bound 0-1 or original bound        
        temp_X=self.X.copy()
        temp_gp=copy.deepcopy(self.gp)
        temp_gp.X_bucb=temp_X.copy()
        temp_gp.KK_x_x_inv_bucb=copy.deepcopy(self.gp.KK_x_x_inv)
        
        # finding new X
        # finding a first point using the original acquisition function (UCB, EI)
        x_max_first = acq_max(ac=self.acq_func.acq_kind, gp=self.gp, bounds=self.scalebounds)
               
        temp_X = np.vstack((temp_X, x_max_first.reshape((1, -1))))

        new_batch_X=[]
        new_batch_X.append(x_max_first) # append the first point
        
        # finding the maximum over the lower bound
        # mu(x)-kappa x sigma(x)
        mu_acq={}
        mu_acq['name']='lcb'
        mu_acq['dim']=self.dim
        if 'kappa' not in self.acq:
            mu_acq['kappa']=2
        else:
            mu_acq['kappa']=self.acq['kappa']
        acq_mu=AcquisitionFunction(mu_acq)
        
        # obtain the argmax(lcb), make sure the scale bound vs original bound
        x_lcb_max = acq_max(ac=acq_mu.acq_kind,gp=self.gp,bounds=self.scalebounds,opt_toolbox=self.opt_toolbox)
        
        # obtain the max(lcb)
        max_lcb=acq_mu.acq_kind(x_lcb_max,gp=self.gp)
        max_lcb=np.ravel(max_lcb)
        
        
        ucb_pe_acq={}
        ucb_pe_acq['name']='ucb_pe'
        ucb_pe_acq['dim']=self.dim
        if 'kappa' not in self.acq:
            ucb_pe_acq['kappa']=2
        else:
            ucb_pe_acq['kappa']=self.acq['kappa']
        ucb_pe_acq['maxlcb']=max_lcb
        acq_ucb_pe=AcquisitionFunction(ucb_pe_acq)

        
        for ii in range(B-1):
            # Finding argmax of the acquisition function.
            x_max = acq_max(ac=acq_ucb_pe.acq_kind, gp=temp_gp, bounds=self.scalebounds)
            

            #if np.any(np.abs((temp_X - x_max)).sum(axis=1) == 0) | np.isnan(x_max.sum()):
                #x_max = np.random.uniform(self.scalebounds[:, 0],self.scalebounds[:, 1],size=self.scalebounds.shape[0])
                #print "the same location - the batch is terminanted!"
                #break
                                        
            new_batch_X= np.vstack((new_batch_X, x_max.reshape((1, -1))))
                            
            # update the Gaussian Process and thus the acquisition function                         
            #temp_gp.compute_incremental_cov_matrix(temp_X,x_max)

            temp_X = np.vstack((temp_X, x_max.reshape((1, -1))))
            temp_gp.X_bucb=temp_X
            

        # record the optimization time
        finished_gmm_opt=time.time()
        elapse_gmm_opt=finished_gmm_opt-start_gmm_opt
        
        self.opt_time=np.hstack((self.opt_time,elapse_gmm_opt))

        self.NumPoints=np.append(self.NumPoints,len(new_batch_X))

        self.X=temp_X
                    
        # convert back to original scale
        temp_X_new_original=[val*self.max_min_gap+self.bounds[:,0] for idx, val in enumerate(new_batch_X)]
        temp_X_new_original=np.asarray(temp_X_new_original)
        self.X_original=np.vstack((self.X_original, temp_X_new_original))
        
        # evaluate y=f(x)
        temp=self.f(temp_X_new_original)
        temp=np.reshape(temp,(-1,1))
        self.Y_original=np.append(self.Y_original,temp)
        self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))
        #print "#Batch={:d} f_max={:.4f}".format(new_batch_X.shape[0],self.Y_original.max())


        # find the maximizer in the GP mean function
        mu_acq={}
        mu_acq['name']='mu'
        mu_acq['dim']=self.dim
        acq_mu=AcquisitionFunction(mu_acq)
        x_mu_max = acq_max(ac=acq_mu.acq_kind,gp=self.gp,bounds=self.scalebounds,opt_toolbox=self.opt_toolbox)
        
        x_mu_max_original=x_mu_max*self.max_min_gap+self.bounds[:,0]
        
        y_mu_max_original=self.f(x_mu_max_original)
        
        temp_y=[y_mu_max_original]*(B)
        temp_x=[x_mu_max_original]*B
        # set y_max = mu_max
        #mu_max=acq_mu.acq_kind(x_mu_max,gp=self.gp)
        self.Y_original_maxGP = np.append(self.Y_original_maxGP, temp_y)
        self.X_original_maxGP = np.vstack((self.X_original_maxGP, np.asarray(temp_x)))
        
    def maximize_batch_UCB_PE_incremental(self,gp_params, B=5):
        """
        Finding a batch of points using GP-BUCB-PE approach
        
        Input Parameters
        ----------

        gp_params:          Parameters to be passed to the Gaussian Process class
        
        B:                  fixed batch size for all iteration      
                 
        Returns
        -------
        X: a batch of [x_1..x_B]
        """        
        self.B=B
                
        # Set acquisition function
        self.acq_func = AcquisitionFunction(self.acq)
               
        # Set parameters if any was passed
        self.gp=PradaGaussianProcess(gp_params)
        
        if len(self.gp.KK_x_x_inv)==0: # check if empty
            self.gp.fit(self.X, self.Y)
        #else:
            #self.gp.fit_incremental(self.X[ur], self.Y[ur])
        
        start_gmm_opt=time.time()
       
        y_max=self.gp.Y.max()
        
        # check the bound 0-1 or original bound        
        temp_X=self.X.copy()
        temp_gp=copy.deepcopy(self.gp)
        temp_gp.X_bucb=temp_X.copy()
        temp_gp.KK_x_x_inv_bucb=copy.deepcopy(self.gp.KK_x_x_inv)
        
        # finding new X
        # finding a first point using the original acquisition function (UCB, EI)
        x_max_first = acq_max(ac=self.acq_func.acq_kind, gp=self.gp, y_max=y_max, bounds=self.scalebounds)
               
        new_batch_X=[]
        new_batch_X.append(x_max_first) # append the first point
        
        # finding the maximum over the lower bound
        # mu(x)-kappa x sigma(x)
        mu_acq={}
        mu_acq['name']='lcb'
        mu_acq['dim']=self.dim
        if 'kappa' not in self.acq:
            mu_acq['kappa']=2
        else:
            mu_acq['kappa']=self.acq['kappa']
        acq_mu=AcquisitionFunction(mu_acq)
        
        # obtain the argmax(lcb), make sure the scale bound vs original bound
        x_lcb_max = acq_max(ac=acq_mu.acq_kind,gp=self.gp,bounds=self.scalebounds,opt_toolbox=self.opt_toolbox)
        
        # obtain the max(lcb)
        max_lcb=acq_mu.acq_kind(x_lcb_max,gp=self.gp)
        max_lcb=np.ravel(max_lcb)
        
        
        ucb_pe_acq={}
        ucb_pe_acq['name']='ucb_pe_incremental'
        ucb_pe_acq['dim']=self.dim
        if 'kappa' not in self.acq:
            ucb_pe_acq['kappa']=2
        else:
            ucb_pe_acq['kappa']=self.acq['kappa']
        ucb_pe_acq['maxlcb']=max_lcb
        acq_ucb_pe=AcquisitionFunction(ucb_pe_acq)

        
        for ii in range(B-1):
            # Finding argmax of the acquisition function.
            x_max = acq_max(ac=acq_ucb_pe.acq_kind, gp=temp_gp, y_max=y_max, bounds=self.scalebounds)
            
            if np.any(np.abs((temp_X - x_max)).sum(axis=1) == 0) | np.isnan(x_max.sum()):
                #x_max = np.random.uniform(self.scalebounds[:, 0],self.scalebounds[:, 1],size=self.scalebounds.shape[0])
                #print "the same location - the batch is terminanted!"
                break
                                        
            new_batch_X= np.vstack((new_batch_X, x_max.reshape((1, -1))))
                            
            # update the Gaussian Process and thus the acquisition function                         
            temp_gp.compute_incremental_cov_matrix(temp_X,x_max)

            temp_X = np.vstack((temp_X, x_max.reshape((1, -1))))
            temp_gp.X_bucb=temp_X
            

        # record the optimization time
        finished_gmm_opt=time.time()
        elapse_gmm_opt=finished_gmm_opt-start_gmm_opt
        
        self.opt_time=np.hstack((self.opt_time,elapse_gmm_opt))

        self.NumPoints=np.append(self.NumPoints,len(new_batch_X))

        self.X=temp_X
                    
        # convert back to original scale
        temp_X_new_original=[val*self.max_min_gap+self.bounds[:,0] for idx, val in enumerate(new_batch_X)]
        temp_X_new_original=np.asarray(temp_X_new_original)
        self.X_original=np.vstack((self.X_original, temp_X_new_original))
        
        # evaluate y=f(x)
        temp=self.f(temp_X_new_original)
        temp=np.reshape(temp,(-1,1))
        self.Y_original=np.append(self.Y_original,temp)
        self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))
        #print "#Batch={:d} f_max={:.4f}".format(new_batch_X.shape[0],self.Y_original.max())
                
        
#======================================================================================================