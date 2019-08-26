% Copyright (c) 2018 Binxin Ru
% Fast Information-Theoretic Bayesian optimisation
% which uses quadrature ( useMM = 0 ) or momenet matching for approximating 
% entropy of the Gaussian mixture
% The sampler used are elliptical slice sampler (ess=1) or slice sampler
% (ess=0) 
% This algorithm include the grid search followed by global optimiser to 
% find the optimal values. 
% The output of the algorithm are:
%       minimiser_FITBO - best guess of location of global minimum 
%       min_predict_FITBO - best guess of global minimum value 
%       next_eval_location_FITBO - next evaluation point recommended
%       next_eval_FITBO - value of the next evaluation point recommended
%       eta_values_FITBO - samples of the hyperparameter \eta
                
close all;path(pathdef); clc;clear all
addpath ./utility
addpath ./testfunc
addpath ./sampler

%%%%%%%%%% Define the objective function %%%%%%%%%%
%objective = @(x) branin(x);           % ojective function
%objective = @(x) branin_vu(x);           % ojective function
objective = @(x) egg(x);           % ojective function
%objective = @(x) ackley(x);           % ojective function
%objective = @(x) alpine2(x);           % ojective function

%objective = @(x) hartmann6D(x);           % ojective function
%objective = @(x) hart6(x);           % ojective function

d=2;
%d         = 5;                        % dimension of data
lb        = [-512,512];           % bounds for data input 
hb        = [512,512];

%%%%%%%%%% Specify parameters %%%%%%%%%%
options.N_seed          = 5;         % number of random initialisations
options.N_evaluation    = 10*d;        % number of evaluations
options.N_hypsample     = 50;        % number of hyperparameter samples
options.var_noise       = 1.0000e-06; % variance of true output noise level 
options.nInitialSamples = 3*d;          % number of initial observation data points
options.ess             = 1;          % use elliptical slice sampler (1) or slice sampler (0)
options.useMM           = 0;          % use Moment-Matching for approxi entropy of GMM
options.kov=-959.64;
%options.kov=0.4;

%%%%%%%%%% Run FITBO %%%%%%%%%%
FITBOacq(objective, lb, hb, options);
%FITBOacq_kov(objective, lb, hb, options);

