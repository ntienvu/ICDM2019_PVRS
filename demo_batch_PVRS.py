from bayes_opt import BayesOpt
from bayes_opt.batchBO.batch_pvrs import BatchPVRS

import matplotlib.pyplot as plt
from bayes_opt.test_functions import functions

import warnings
import sys

warnings.filterwarnings("ignore")




# select the function to be optimized
#myfunction=functions.branin(sd=0)
myfunction=functions.hartman_3d()
#myfunction=functions.hartman_6d()
#myfunction=functions.ackley(input_dim=5)
#myfunction=functions.gSobol(a=np.array([1,1,1,1,1]))
#myfunction=functions.gSobol(a=np.array([1,1,1,1,1,1,1,1,1,1]))



func=myfunction.func
    

# specifying the acquisition function
acq_func={}
acq_func['name']='pvrs'


acq_func['dim']=myfunction.input_dim
    
# we specify the known optimum value here
acq_func['fstar']=myfunction.fstar



func_params={}
func_params['function']=myfunction


acq_params={}
acq_params['acq_func']=acq_func
gp_params = {'kernel':'SE','lengthscale':0.04*myfunction.input_dim,'noise_delta':1e-8,'flagIncremental':0}


bo=BatchPVRS(gp_params,func_params,acq_params)
            

# initialize BO using 3*dim number of observations
bo.init(n_init_points=3*myfunction.input_dim)

# run for 10*dim iterations
NN=5*myfunction.input_dim
for index in range(0,NN):

    bo.maximize_batch_greedy_PVRS(B=3) #B is the batch size

    
    if myfunction.ismax==1:
        print("recommemded x={}, current inferred value ={:.5f}, best inferred value ={:.5f}".format(bo.X_original[-1],bo.Y_original_maxGP[-1],bo.Y_original_maxGP.max()))
    else:
        print("recommemded x={} current inferred value ={:.5f}, best inferred value={:.5f}".format(bo.X_original[-1],myfunction.ismax*bo.Y_original_maxGP[-1],myfunction.ismax*bo.Y_original_maxGP.max()))
    sys.stdout.flush()

fig=plt.figure(figsize=(6, 3))
myYbest=[bo.Y_original[:idx+1].max()*-1 for idx,val in enumerate(bo.Y_original)]
plt.plot(range(len(myYbest)),myYbest,linewidth=2,color='m',linestyle='-',marker='o')
plt.xlabel('Iteration',fontsize=14)
plt.ylabel('Best Found Value',fontsize=14)
plt.title('Performance',fontsize=16)
