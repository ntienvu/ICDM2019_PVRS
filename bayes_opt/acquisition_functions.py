
import numpy as np
from scipy.stats import norm
from sklearn.metrics.pairwise import euclidean_distances
from scipy import stats



counter = 0


class AcquisitionFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, acq):
        
        self.acq=acq
        acq_name=acq['name']
        
        ListAcq=['bucb','ucb', 'ei', 'poi','random','thompson',  'pvrs', 'thompson', 'mu',                     
                     'pure_exploration','kov_mes','mes','kov_ei',
                         'kov_erm','kov_cbm','kov_tgp','kov_tgp_ei']
        
        # check valid acquisition function
        IsTrue=[val for idx,val in enumerate(ListAcq) if val in acq_name]
        #if  not in acq_name:
        if  IsTrue == []:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(acq_name)
            raise NotImplementedError(err)
        else:
            self.acq_name = acq_name
            
        self.dim=acq['dim']
        
        if 'scalebounds' not in acq:
            self.scalebounds=[0,1]*self.dim
            
        else:
            self.scalebounds=acq['scalebounds']
        
        # vector theta for thompson sampling
        #self.flagTheta_TS=0
        self.initialized_flag=0
        self.objects=[]
        
    def acq_kind(self, x, gp):
        
        y_max=np.max(gp.Y)
        
        if np.any(np.isnan(x)):
            return 0
        
        if self.acq_name == 'ucb':
            return self._ucb(x, gp)
        if self.acq_name=='kov_cbm':
            return self._cbm(x,gp,target=self.acq['fstar_scaled'])
        if self.acq_name == 'lcb':
            return self._lcb(x, gp)
        if self.acq_name == 'ei' or self.acq_name=='kov_tgp_ei':
            return self._ei(x, gp, y_max)
        if self.acq_name == 'kov_ei' :
            return self._ei(x, gp, y_max=self.acq['fstar_scaled'])
        if self.acq_name == 'kov_erm' or self.acq_name =='kov_tgp' or self.acq_name=='kov_ei_cb':
            return self._erm(x, gp, fstar=self.acq['fstar_scaled'])
        

        if self.acq_name == 'pure_exploration':
            return self._pure_exploration(x, gp) 
       
        if self.acq_name == 'ei_mu':
            return self._ei(x, gp, y_max)
        if self.acq_name == 'mu':
            return self._mu(x, gp)
        if self.acq_name == 'mes':
            if self.initialized_flag==0:
                self.object=AcquisitionFunction.MaxValueEntropySearch(gp,self.scalebounds,
                                                                      ystars=self.acq['ystars'])
                self.initialized_flag=1
                return self.object(x)
            else:
                return self.object(x)  
            
        if self.acq_name == 'kov_mes':
            if self.initialized_flag==0:
                self.object=AcquisitionFunction.MaxValueEntropySearch(gp,self.scalebounds,
                                                                      ystars=np.asarray([self.acq['fstar_scaled']]))
                self.initialized_flag=1
                return self.object(x)
            else:
                return self.object(x)  
        if 'pvrs' in self.acq_name: #pvrs (10d) and #pvrs_50d
            if self.initialized_flag==0:
                self.object=AcquisitionFunction.PredictiveVarianceReductionSearch(gp,self.scalebounds,
                                                                             xstars=self.acq['xstars'])
                self.initialized_flag=1
                return self.object(x,gp)
            else:
                return self.object(x,gp)
            
            
        if self.acq_name == 'thompson':
            if self.initialized_flag==0:
                self.object=AcquisitionFunction.ThompsonSampling(gp)
                self.initialized_flag=1
                return self.object(x,gp)
            else:
                return self.object(x,gp)
            
     

    @staticmethod
    def _mu(x, gp):
        mean, var = gp.predict(x, eval_MSE=True)
        mean=np.atleast_2d(mean).T
        return mean
                
    @staticmethod
    def _lcb(x, gp):
        mean, var = gp.predict(x, eval_MSE=True)
        var.flags['WRITEABLE']=True
        #var=var.copy()
        var[var<1e-10]=0
        mean=np.atleast_2d(mean).T
        var=np.atleast_2d(var).T
        #beta_t = gp.X.shape[1] * np.log(len(gp.Y))
        beta_t = 2 * np.log(len(gp.Y));

        return mean - np.sqrt(beta_t) * np.sqrt(var) 
        
    
    @staticmethod
    def _ucb(x, gp):
        mean, var = gp.predict(x, eval_MSE=True)
        var.flags['WRITEABLE']=True
        #var=var.copy()
        var[var<1e-10]=0
        mean=np.atleast_2d(mean).T
        var=np.atleast_2d(var).T                
        
        # Linear in D, log in t https://github.com/kirthevasank/add-gp-bandits/blob/master/BOLibkky/getUCBUtility.m
        #beta_t = gp.X.shape[1] * np.log(len(gp.Y))
        beta_t = np.log(len(gp.Y))
  
        #beta=300*0.1*np.log(5*len(gp.Y))# delta=0.2, gamma_t=0.1
        return mean + np.sqrt(beta_t) * np.sqrt(var) 
    
    @staticmethod
    def _cbm(x, gp, target): # confidence bound minimization
        mean, var = gp.predict(x, eval_MSE=True)
        var.flags['WRITEABLE']=True
        var[var<1e-10]=0
        mean=np.atleast_2d(mean).T
        var=np.atleast_2d(var).T                
        
        # Linear in D, log in t https://github.com/kirthevasank/add-gp-bandits/blob/master/BOLibkky/getUCBUtility.m
        #beta_t = gp.X.shape[1] * np.log(len(gp.Y))
        beta_t = np.log(len(gp.Y))
  
        #beta=300*0.1*np.log(5*len(gp.Y))# delta=0.2, gamma_t=0.1
        return -np.abs(mean-target) - np.sqrt(beta_t) * np.sqrt(var) 
    
  
   
    @staticmethod
    def _erm(x, gp, fstar):
        #y_max=np.asscalar(y_max)
        mean, var = gp.predict(x, eval_MSE=True)
        
        if mean.shape!=var.shape:
            print("bug")
            mean, var = gp.predict(x, eval_MSE=True)
        var2 = np.maximum(var, 1e-10 + 0 * var)
        z = (mean - fstar)/np.sqrt(var2)        
        out=(fstar-mean) * (1-norm.cdf(z)) + np.sqrt(var2) * norm.pdf(z)
        
        out[var2<1e-10]=0
        
        #print(out)
        if any(out)<0:
            print("out<0")
        return -1*out # for minimization problem
                    
                
    @staticmethod
    def _ei(x, gp, y_max):
        #y_max=np.asscalar(y_max)
        mean, var = gp.predict(x, eval_MSE=True)
        if gp.nGP==0:
            var2 = np.maximum(var, 1e-10 + 0 * var)
            z = (mean - y_max)/np.sqrt(var2)        
            out=(mean - y_max) * norm.cdf(z) + np.sqrt(var2) * norm.pdf(z)
            
            out[var2<1e-10]=0
            return out
        else:                 # multiple GP
            z=[None]*gp.nGP
            out=[None]*gp.nGP
            # Avoid points with zero variance
            #y_max=y_max*0.8
            for idx in range(gp.nGP):
                var[idx] = np.maximum(var[idx], 1e-9 + 0 * var[idx])
            
                z[idx] = (mean[idx] - y_max)/np.sqrt(var[idx])
            
                out[idx]=(mean[idx] - y_max) * norm.cdf(z[idx]) + np.sqrt(var[idx]) * norm.pdf(z[idx])
                
            if len(x)==1000:    
                return out
            else:
                return np.mean(out)# get mean over acquisition functions
                return np.prod(out,axis=0) # get product over acquisition functions

   
 
     
    @staticmethod
    def _pure_exploration(x, gp):
        mean, var = gp.predict(x, eval_MSE=True)
        var.flags['WRITEABLE']=True
        #var=var.copy()
        var[var<1e-10]=0
        mean=np.atleast_2d(mean).T
        var=np.atleast_2d(var).T
        return np.sqrt(var)
  

    @staticmethod      
    def _poi(x, gp,y_max): # run Predictive Entropy Search using Spearmint
        mean, var = gp.predict(x, eval_MSE=True)    
        # Avoid points with zero variance
        var = np.maximum(var, 1e-9 + 0 * var)
        z = (mean - y_max)/np.sqrt(var)        
        return norm.cdf(z)

 
    class PredictiveVarianceReductionSearch(object): # perform PVRS of x* from thompson sampling only
        def __init__(self,gp,boundaries,xstars=[]):
            self.dim=gp.X.shape[1]
            # get the suggestions x_t from EI, UCB, ES, PES
            # this step will be performed once
            
            numXtar=10*self.dim
            self.Euc_dist_train_train=[]
            #y_max=np.max(gp.Y)
            if xstars==[]:
                
                print("generate x* inside acquisition function VRS of TS")
                self.xstars=[]

                # finding the xt of Thompson Sampling
                for ii in range(numXtar):
                    xt_TS,y_xt_TS=acq_max_with_name(gp=self.gp,scalebounds=self.scalebounds,
                                                    acq_name="thompson",IsReturnY=True)
                    
                    # check if f* > y^max and ignore xt_TS otherwise
                    if y_xt_TS>=np.max(gp.Y):
                        self.xstars.append(xt_TS)
                    
            else:
                self.xstars=xstars
                
            # compute the predictive variances at these locations
            myvar=[]
            for idx,val in enumerate(self.xstars):
                #predmean,predvar=gp.predict(val)
                predvar=gp.compute_var(gp.X,val)
                myvar.append(predvar)

            # take these average for numerical stability
            self.average_predvar=np.mean(myvar)


        def compute_variance_marginal_hyperparameter(self,X,xTest,gp):
            """
            compute variance given X and xTest
            
            Input Parameters
            ----------
            X: the observed points
            xTest: the testing points 
            
            Returns
            -------
            diag(var)
            """ 
            
            # we implement for SE kernel only
            xTest=np.asarray(xTest)
            #Euc_dist=euclidean_distances(xTest,xTest)
            #KK_xTest_xTest=np.exp(-np.square(Euc_dist)/self.lengthscale)+np.eye(xTest.shape[0])*self.noise_delta
            ur = unique_rows(X)
            X=X[ur]
            
            #if xTest.shape[0]!=len(gp.lengthscale_vector):
             #   print "bug"
            Euc_dist_test_train=euclidean_distances(xTest,X)
            
            Euc_dist_train_train=euclidean_distances(X,X)
                
            var=[0]*len(gp.lengthscale_vector)
            for idx,lengthscale in enumerate(gp.lengthscale_vector):
                KK_xTest_xTrain=np.exp(-np.square(Euc_dist_test_train[idx])/lengthscale)

                KK_bucb_train_train=np.exp(-np.square(Euc_dist_train_train)/lengthscale)+np.eye(X.shape[0])*gp.noise_delta        
            
                try:
                    temp=np.linalg.solve(KK_bucb_train_train,KK_xTest_xTrain.T)
                except:
                    temp=np.linalg.lstsq(KK_bucb_train_train,KK_xTest_xTrain.T, rcond=-1)
                    temp=temp[0]
                
            #var=KK_xTest_xTest-np.dot(temp.T,KK_xTest_xTrain.T)
                var[idx]=1-np.dot(temp.T,KK_xTest_xTrain.T)

            #var.flags['WRITEABLE']=True
            var[var<1e-100]=0
            return var
        
        def compute_var_incremental_cov_matrix(self,X,newX,xTest,gp):
            """
            Compute covariance matrix incrementall for BUCB (KK_x_x_inv_bucb)
            
            Input Parameters
            ----------
            X: the observed points 
            newX: the new point
            xTest: the test point (to compute variance)
            Returns
            -------
            KK_x_x_inv_bucb: the covariance matrix will be incremented one row and one column
            """   
            
            if len(newX.shape)==1: # 1d
                newX=newX.reshape((-1,newX.shape[0]))
                
            nNew=np.shape(newX)[0]
            #K_xtest_xtrain
            Euc_dist=euclidean_distances(X,newX)
            KK_x=np.exp(-np.square(Euc_dist)*1.0/gp.lengthscale)+gp.noise_delta           
            
            #delta_star=np.dot(self.KK_x_x_inv_bucb,KK_x)
            delta_star=np.dot(gp.KK_x_x_inv,KK_x)

            sigma=np.identity(nNew)-np.dot(KK_x.T,delta_star)
            inv_sigma=np.linalg.pinv(sigma)
    
            temp=np.dot(delta_star,inv_sigma)
            #TopLeft=self.KK_x_x_inv_bucb+np.dot(temp,delta_star.T)
            TopLeft=gp.KK_x_x_inv+np.dot(temp,delta_star.T)

            TopRight=-np.dot(delta_star,np.linalg.pinv(sigma))
            BottomLeft=-np.dot(inv_sigma,delta_star.T)
            BottomRight=np.dot(np.identity(nNew),inv_sigma)   
            
            new_K_inv=np.vstack((TopLeft,BottomLeft))
            temp=np.vstack((TopRight,BottomRight))
            
            # new incremental covariance
            KK_x_x_inv_new=np.hstack((new_K_inv,temp))
                    
            xTest=np.asarray(xTest)
            Euc_dist_test=euclidean_distances(xTest,xTest)
            KK_xTest_xTest=np.exp(-np.square(Euc_dist_test)*1.0/gp.lengthscale)+np.eye(xTest.shape[0])*gp.noise_delta
            
            Euc_dist_test_train=euclidean_distances(xTest,np.vstack((X,newX)))
            KK_xTest_xTrain=np.exp(-np.square(Euc_dist_test_train)*1.0/gp.lengthscale)
            
            
            temp=np.dot(KK_xTest_xTrain,KK_x_x_inv_new)
            var=KK_xTest_xTest-np.dot(temp,KK_xTest_xTrain.T)        
            return np.diag(var)  
        
        
        def compute_var(self,X,xTest,lengthscale,noise_delta):
            """
            compute variance given X and xTest
            
            Input Parameters
            ----------
            X: the observed points
            xTest: the testing points 
            
            Returns
            -------
            diag(var)
            """ 
            xTest=np.asarray(xTest)
    
            ur = unique_rows(X)
            X=X[ur]
    
            Euc_dist_test_train=euclidean_distances(xTest,X)
            #Euc_dist_test_train=dist(xTest, X, matmul='gemm', method='ext', precision='float32')
            KK_xTest_xTrain=np.exp(-np.square(Euc_dist_test_train)/lengthscale)

            Euc_dist_train_train=euclidean_distances(X,X)
            KK_bucb_train_train=np.exp(-np.square(Euc_dist_train_train)/lengthscale)+np.eye(X.shape[0])*noise_delta        
    
            try:
                temp=np.linalg.solve(KK_bucb_train_train,KK_xTest_xTrain.T)
            except:
                temp=np.linalg.lstsq(KK_bucb_train_train,KK_xTest_xTrain.T, rcond=-1)
                temp=temp[0]
                
            #var=KK_xTest_xTest-np.dot(temp.T,KK_xTest_xTrain.T)
            var=np.eye(xTest.shape[0])-np.dot(temp.T,KK_xTest_xTrain.T)
            var=np.diag(var)
            var.flags['WRITEABLE']=True
            var[var<1e-100]=0
            return var 
    
        def __call__(self,x,gp):
            if len(x)==self.dim: # one observation
                sum_variance=0

                if gp.lengthscale_vector!=[]: #marginal
                    X=np.vstack((gp.X,x))
                    var=self.compute_variance_marginal_hyperparameter(X,self.xstars,gp)
                else:
                    #var=self.compute_var_incremental_cov_matrix(gp.X,x,self.xstars,gp)
                    X=np.vstack((gp.X,x))
                    #var=self.compute_var(X,self.xstars,gp.lengthscale,gp.noise_delta) 
                    var=gp.compute_var(X,self.xstars) 

                temp=np.mean(var)
                #sum_variance=self.average_predvar-temp
                sum_variance=-temp
        
                return np.asarray(sum_variance)      # we want to minimize not maximize
            else:
                sum_variance=[0]*len(x)
                for idx2,val2 in enumerate(x):
                    #var=gp.compute_var(X,self.xstars)
                    if gp.lengthscale_vector!=[]: #marginal
                        X=np.vstack((gp.X,val2))
                        var=self.compute_variance_marginal_hyperparameter(X,self.xstars,gp)
                    else:
                        #var=self.compute_var_incremental_cov_matrix(gp.X,val2,self.xstars,gp)
                        X=np.vstack((gp.X,val2))
                        #var=self.compute_var(X,self.xstars,gp.lengthscale,gp.noise_delta)
                        
                        # if xstars set is too large, split them into smaller chunks for ease of computation
                        var=np.array([])
                        if len(self.xstars)<=100:
                            var=gp.compute_var(X,self.xstars)    
                        else:
                            nsplit=np.ceil(len(self.xstars)*1.0/100)
                            xstars_split=np.array_split(self.xstars,nsplit)
                            for idx_split, val in enumerate(xstars_split):
                                temp_var=gp.compute_var(X,val) 
                                var=np.hstack((var,temp_var))
                            
                    temp=np.mean(var)
                    #sum_variance[idx2]=-np.log(temp)# we want to minimize not maximize
                    sum_variance[idx2]=self.average_predvar-temp# we want to minimize not maximize
                    #sum_variance[idx2]=-temp# we want to minimize not maximize

                    if np.isnan(sum_variance[idx2]): # debug
                        print("nan")
                        #var=gp.compute_var(X,self.xstars)
                        var=self.compute_var_incremental_cov_matrix(gp.X,x,self.xstars,gp.lengthscale,gp.noise_delta)

                #sum_variance=(sum_variance-np.min(sum_variance))/(np.max(sum_variance)-np.min(sum_variance))                   
                return np.asarray(sum_variance)    				               

    
    class ThompsonSampling(object):
        def __init__(self,gp):
            dim=gp.X.shape[1]
            # used for Thompson Sampling
            #self.WW_dim=200 # dimension of random feature
            
            self.WW_dim=30*dim # dimension of random feature
            self.WW=np.random.multivariate_normal([0]*self.WW_dim,np.eye(self.WW_dim),dim)/gp.lengthscale  
            self.bias=np.random.uniform(0,2*3.14,self.WW_dim)

            # computing Phi(X)^T=[phi(x_1)....phi(x_n)]
            Phi_X=np.sqrt(2.0/self.WW_dim)*np.hstack([np.sin(np.dot(gp.X,self.WW)+self.bias), np.cos(np.dot(gp.X,self.WW)+self.bias)]) # [N x M]
            
            # computing A^-1
            A=np.dot(Phi_X.T,Phi_X)+np.eye(2*self.WW_dim)*gp.noise_delta
            
            # theta ~ N( A^-1 Phi_T Y, sigma^2 A^-1
            gx=np.dot(Phi_X.T,gp.Y)
            self.mean_theta_TS=np.linalg.solve(A,gx)
            
        def __call__(self,x,gp):
            #phi_x=np.sqrt(1.0/self.UU_dim)*np.hstack([np.sin(np.dot(x,self.UU)), np.cos(np.dot(x,self.UU))])
            phi_x=np.sqrt(2.0/self.WW_dim)*np.hstack([np.sin(np.dot(x,self.WW)+self.bias), np.cos(np.dot(x,self.WW)+self.bias)])
            
            # compute the mean of TS
            gx=np.dot(phi_x,self.mean_theta_TS)    
            return gx  
        
        
    class EntropySearch(object):
        def __init__(self,gp,boundaries,xstars=[]):
            from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C

            from bayesian_optimization import ( GaussianProcessModel,
                                               UpperConfidenceBound,EntropySearch,MinimalRegretSearch)
    
            # Configure Bayesian optimizer
            #kernel = C(1.0, (0.01, 1000.0)) \
            	#* Matern(length_scale=1.0, length_scale_bounds=[(0.01, 100)])
            kernel=RBF(gp.lengthscale)
            model = GaussianProcessModel(kernel=kernel)
            model.fit(gp.X,gp.Y)
            self.acq_func_es = EntropySearch(model,n_candidates=10, n_gp_samples=200,n_samples_y=10, n_trial_points=200, rng_seed=0)
            
            if xstars==[]:
                self.acq_func_es.set_boundaries(boundaries)
            else:
                self.acq_func_es.set_boundaries(boundaries,X_candidate=np.asarray(xstars))
            self.x_stars=self.acq_func_es.X_candidate
            
        def __call__(self,x):
            return self.acq_func_es(x)                

    class MaxValueEntropySearch(object):
        def __init__(self,gp,boundaries,ystars=[]):

            self.X=gp.X
            self.Y=gp.Y
            self.gp=gp
            if ystars==[]:
                print("y_star is empty for MES")                
            self.y_stars=ystars
                
        def __call__(self,x):
            mean_x, var_x = self.gp.predict(x, eval_MSE=True)

            acq_value=0
            for idx,val in enumerate(self.y_stars):
                gamma_ystar=(val-mean_x)*1.0/var_x
                temp=0.5*gamma_ystar*norm.pdf(gamma_ystar)*1.0/norm.cdf(gamma_ystar)-np.log(norm.cdf(gamma_ystar))
                acq_value=acq_value+temp
            #acq_value=acq_value*1.0/len(self.y_stars)
            return acq_value

def unique_rows(a):
    """
    A functions to trim repeated rows that may appear when optimizing.
    This is necessary to avoid the sklearn GP object from breaking

    :param a: array to trim repeated rows from

    :return: mask of unique rows
    """

    # Sort array and kep track of where things should go back to
    order = np.lexsort(a.T)
    reorder = np.argsort(order)

    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)

    return ui[reorder]



class BColours(object):
    BLUE = '\033[94m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    MAGENTA = '\033[35m'
    RED = '\033[31m'
    ENDC = '\033[0m'
