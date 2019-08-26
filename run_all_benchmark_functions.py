import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../..')


from bayes_opt.sequentialBO.bayesian_optimization import BayesOpt

import numpy as np
from bayes_opt import auxiliary_functions

from bayes_opt.test_functions import functions,real_experiment_function
import warnings
#from bayes_opt import acquisition_maximization

import sys

from bayes_opt.utility import export_results
import itertools


np.random.seed(6789)

warnings.filterwarnings("ignore")


counter = 0


myfunction_list=[]


#myfunction_list.append(functions.branin())
myfunction_list.append(functions.eggholder())
myfunction_list.append(functions.hartman_3d())
myfunction_list.append(functions.ackley(input_dim=5))
myfunction_list.append(functions.alpine2(input_dim=5))
myfunction_list.append(functions.hartman_6d())
myfunction_list.append(functions.gSobol(a=np.array([1,1,1,1,1,1,1,1,1,1])))


acq_type_list=[]

temp={}
temp['name']='pvrs' #PVRS
acq_type_list.append(temp)



for idx, (myfunction,acq_type,) in enumerate(itertools.product(myfunction_list,acq_type_list)):
    func=myfunction.func
    
    func_params={}
    func_params['function']=myfunction

    gp_params = {'lengthscale':0.05*myfunction.input_dim,'noise_delta':1e-7} # the lengthscaled parameter will be optimized

    yoptimal=myfunction.fstar*myfunction.ismax
    
    acq_type['dim']=myfunction.input_dim
    acq_type['debug']=0
    acq_type['fstar']=myfunction.fstar

    acq_params={}
    acq_params['optimize_gp']='maximize'#maximize
    acq_params['acq_func']=acq_type
    
    nRepeat=20
    
    ybest=[0]*nRepeat
    MyTime=[0]*nRepeat
    MyOptTime=[0]*nRepeat
    marker=[0]*nRepeat

    bo=[0]*nRepeat
   
    [0]*nRepeat
    
    
    for ii in range(nRepeat):
        

        bo[ii]=BayesOpt(gp_params,func_params,acq_params,verbose=0)
  
        ybest[ii],MyTime[ii]=auxiliary_functions.run_experiment(bo[ii],gp_params,
             n_init=3*myfunction.input_dim,NN=20*myfunction.input_dim,runid=ii)                                               
        MyOptTime[ii]=bo[ii].time_opt
        print("ii={} Best Inferred Value={:.6f}".format(ii,np.max(ybest[ii])))                                              
        

    Score={}
    Score["ybest"]=ybest
    Score["MyTime"]=MyTime
    Score["MyOptTime"]=MyOptTime
    
    export_results.print_result_sequential(bo,myfunction,Score,acq_type) 