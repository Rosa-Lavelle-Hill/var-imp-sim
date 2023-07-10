
enet_param_grid = {"tol": [0.001],
                   "max_iter": [5000],
                   "l1_ratio": [0.2, 0.4, 0.6, 0.8],
                   "alpha": [0.2, 0.5, 0.7, 1, 1.5, 2]
                   }

lasso_param_grid = {"tol": [0.001],
                   "max_iter": [5000],
                   "alpha": [0.2, 0.5, 1, 1.5, 2]
                   }

rf_param_grid = {"min_samples_split": [2, 4, 6],
                 "max_depth": [5, 10, 20],
                 "n_estimators": [50],
                 "max_features": [0.3]}

tree_param_grid = {"min_samples_split": [2, 4, 6],
                 "max_depth": [5, 10, 20],
                 "max_features": [0.3]}

