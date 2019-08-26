from bayes_opt import BayesOpt
import matplotlib.pyplot as plt
from bayes_opt.test_functions import functions
from bayes_opt.visualization import vis_variance_reduction_search as viz
import warnings
import sys

warnings.filterwarnings("ignore")




# select the function to be optimized
myfunction=functions.branin(sd=0)
#myfunction=functions.hartman_3d()
#myfunction=functions.hartman_6d()
#myfunction=functions.ackley(input_dim=5)
#myfunction=functions.gSobol(a=np.array([1,1,1,1,1]))
#myfunction=functions.gSobol(a=np.array([1,1,1,1,1,1,1,1,1,1]))



func=myfunction.func
    

# specifying the acquisition function
acq_func={}
#acq_func['name']='ei'
#acq_func['name']='ucb'
acq_func['name']='pvrs'



acq_func['dim']=myfunction.input_dim
    

func_params={}
func_params['function']=myfunction


acq_params={}
acq_params['acq_func']=acq_func
gp_params = {'kernel':'SE','lengthscale':0.2*myfunction.input_dim,'noise_delta':1e-8}


bo=BayesOpt(gp_params,func_params,acq_params)
            

# initialize BO using 3*dim number of observations
bo.init(gp_params,n_init_points=1*myfunction.input_dim)

# run for 10*dim iterations
NN=3*myfunction.input_dim
for index in range(0,NN):

    bo.maximize()

    #viz.plot_bo_2d_pvrs(bo)
    viz.plot_bo_2d_pvrs_short(bo)

    
    if myfunction.ismax==1:
        print("recommemded x={}, current inferred value ={}, best inferred value ={}".format(bo.X_original[-1],bo.Y_original_maxGP[-1],bo.Y_original_maxGP.max()))
    else:
        print("recommemded x={} current inferred value y={}, best inferred value={}".format(bo.X_original[-1],myfunction.ismax*bo.Y_original_maxGP[-1],myfunction.ismax*bo.Y_original_maxGP.max()))
    sys.stdout.flush()

fig=plt.figure(figsize=(6, 3))
myYbest=[bo.Y_original[:idx+1].max()*-1 for idx,val in enumerate(bo.Y_original)]
plt.plot(range(len(myYbest)),myYbest,linewidth=2,color='m',linestyle='-',marker='o')
plt.xlabel('Iteration',fontsize=14)
plt.ylabel('Best Found Value',fontsize=14)
plt.title('Performance',fontsize=16)
