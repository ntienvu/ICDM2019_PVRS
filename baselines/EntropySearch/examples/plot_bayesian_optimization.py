"""
=====================================
Illustration of Bayesian Optimization
=====================================

Visualize Bayesian optimization on a simple 1D function.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C

from bayesian_optimization import (BayesianOptimizer, GaussianProcessModel,
    UpperConfidenceBound,EntropySearch,MinimalRegretSearch)

# Configure Bayesian optimizer
kernel = C(1.0, (0.01, 1000.0)) \
	* Matern(length_scale=1.0, length_scale_bounds=[(0.01, 100)])
    
kernel = RBF(0.1)
    
model = GaussianProcessModel(kernel=kernel)
kappa = 2.5
#acquisition_function = UpperConfidenceBound(model, kappa=kappa)
acquisition_function = EntropySearch(model, n_candidates=20, n_gp_samples=500,n_samples_y=10, n_trial_points=500, rng_seed=0)

#acquisition_function = MinimalRegretSearch(model, n_candidates=20, n_gp_samples=500,n_samples_y=10, n_trial_points=500, point=False, rng_seed=0)



bayes_opt = BayesianOptimizer(model=model,
                              acquisition_function=acquisition_function,
                              optimizer="random")

def f(X):  # target function
    return -np.linalg.norm(X)

# Perform trials
n_trials = 8
for i in range(n_trials):
    X_query = bayes_opt.select_query_point(boundaries=np.array([[-1, 1]]))
    y_query = f(X_query)
    bayes_opt.update(X_query, y_query)

    # Plot learned model and acquisition function
    X_ = np.linspace(-1, 1, 250)[:, None]
    y_pred, y_std = bayes_opt.model.predictive_distribution(X_)
    
plt.plot(X_[:, 0], np.apply_along_axis(f, 1, X_), c='k', label="Truth")
plt.plot(X_[:, 0], y_pred, c='b', label="GP mean")
plt.fill_between(X_[:, 0], y_pred - y_std, y_pred + y_std,
                   color='b', alpha=0.3, label="GP mean+-std")
plt.plot(X_[:, 0], acquisition_function(X_), color='r',
         label="Acquisition function")
plt.scatter(bayes_opt.X_, bayes_opt.y_)

plt.scatter(bayes_opt.X_[-1], bayes_opt.y_[-1],color='g',label='New Obs')

plt.legend(loc="best")
plt.xlabel("Data space")
plt.ylabel("Target value")
plt.show()

print acquisition_function(0.1,1)
