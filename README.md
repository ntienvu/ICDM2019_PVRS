# Efficient Bayesian Optimization for Uncertainty Reduction over Perceived Optima Locations

This is the source code for our ICDM 2019 paper entitled "Efficient Bayesian Optimization for Uncertainty Reduction over Perceived Optima Locations"
Copy right by the authors.

# Reference
```
@inproceedings{nguyen2019efficient,
  title={Efficient Bayesian Optimization for Uncertainty Reduction Over Perceived Optima Locations},
  author={Nguyen, Vu and Gupta, Sunil and Rana, Santu and Thai, My and Li, Cheng and Venkatesh, Svetha},
  booktitle={2019 IEEE International Conference on Data Mining (ICDM)},
  pages={1270--1275},
  year={2019},
  organization={IEEE}
}
```

# Scripts:
## Running the demo in the "demo" folder
```
demo_PVRS_with_plot.py # run a simple illustration
demo_PVRS.py # run a PVRS in sequential setting
demo_batch_PVRS.py # run a PVRS in a batch setting
```

## Running to reproduce experiments
```
run_all_benchmark_functions.py # benchmark functions
```

After running these scripts to reproduce experiments, the results will be stored as pickles files in "pickle_storage" folder.

# Dependencies
* Numpy
* Scipy
* Scikit-learn

Disclaimer: This project is under active development, if you find a bug, or anything that needs correction, please let me know.
