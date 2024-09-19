import random
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import shap as shap
import datetime as dt
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from Functions.gen_data import add_noise, extract_coef, generate_interaction_data
from Functions.plotting import plot_impurity, plot_permutation, plot_SHAP, plot_SHAP_force, plot_PDP, plot_ICE, \
    check_corr, print_tree, plot_multiple_permutations
from Functions.pred import define_model
from PyALE import ale

# =================================
# Define the changeable parameters:
# =================================

n_samples = 1000 # number of samples in generated data
n_features = 5 # number of features (or "independent variables") in generated data
mean = 0 # mean of generated data
sd = 1 # standard deviation of generated data
iv_cor = 0.6 # the Pearson r correlation of the features with one another in the generated data
dv_r2 = 0.6 # the R-squared value representing the extent the features can explain y in the full generated dataset
test_size = 0.5 # ratio of training:test data
cv = 5 # number of cross-validation splits
scoring = "r2" # scoring used for both the training and the testing: 'r2' is prediction R-squared, for other options, see: https://scikit-learn.org/stable/modules/model_evaluation.html
permutations = 10 # number of permutations in permutation importance calculations
shap_method = "interventional" # "interventional" = true to model; "correlation_dependent" = true to data (for tree-based models "path_dependent") (Lundberg & Lee, 2017; 2020)
explain_data_instance_num = 0 # the row index indicating which instance in the data to create a local explanation for (used for SHAP and LIME)
decimal_places = 2 # integer used for rounding
seed = 93 # the random seed (used in the data generating process, splitting process, the model fitting process, and the permutation importance calculations)
results_path = "Results figure-X/Interpretation/"
outputs_path = "Outputs figure-X/"
replicate_figure_model = True
# -------------------------- Run Simulations --------------------------

start_time = dt.datetime.now()
for pred_model in ["rf"]:
    print(f"Running model: {pred_model}")
    if replicate_figure_model == True:
        np.random.seed(0) #todo: update final figure to this iteration

    # -------------------------- Generate Data --------------------------

    X, y, Z = generate_interaction_data()

    # Plot the data
    font_size = 18
    plt.figure(figsize=(5, 3.5))
    scatter = plt.scatter(X, y, c=Z, cmap='coolwarm', alpha=0.6)  # Color points by Z value
    cbar = plt.colorbar(scatter)
    cbar.set_label("Z Value", fontsize=font_size)
    plt.xlabel("X", fontsize=font_size)
    plt.ylabel("y", fontsize=font_size)
    plt.tight_layout()
    plt.savefig(outputs_path + "actual_data_interaction_Z.png")
    plt.clf()
    plt.cla()
    plt.close()

    # Plot the data
    plt.figure(figsize=(4, 4))
    plt.scatter(Z, y, alpha=0.6)
    plt.xlabel("Z", fontsize=font_size)
    plt.ylabel("y", fontsize=font_size)
    plt.tight_layout()
    plt.savefig(outputs_path + "actual_data_y_Z.png")
    plt.clf()
    plt.cla()
    plt.close()

    # Change to dataframes
    X_df = pd.DataFrame(zip(X, Z), columns=['X', 'Z'])
    n_features = n_features + 1
    y = pd.Series(y, name="y")
    vars = ['X', 'Z']

    # Check correlations
    X_and_y = pd.concat([X_df, y], axis=1)
    check_corr(X_and_y=X_and_y, X_feature_names=vars, save_path=outputs_path,
               save_name="data_cor_plot", decimal_places=decimal_places)

    # Extract coefficients
    extract_coef(X=X_df, y=y, X_feature_names=vars, decimal_places=decimal_places, file_path=outputs_path)

    # -------------------------- Train Model and Evaluate Out-Of-Sample (OOS) --------------------------

    # Split train and test (same random seed so constant stable comparison)
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, random_state=seed, test_size=test_size, shuffle=True)

    if (pred_model == "rf") and (replicate_figure_model == False):

        # Define model
        model, param_grid = define_model(pred_model=pred_model, fixed_seed=seed)

        # Perform CV on train data to tune model hyper-parameters
        grid_search = GridSearchCV(estimator=model,
                                   param_grid=param_grid,
                                   cv=cv,
                                   scoring=scoring,
                                   refit=True,
                                   verbose=0,
                                   n_jobs=2)

        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        cv_r2 = round(grid_search.best_score_, decimal_places)
        print(f"best hyper-params from training: {best_params} \n CV score: {cv_r2}")

    if (pred_model == "rf") and (replicate_figure_model == True):
        # parameters used to replicate Figure:
        if n_samples == 1000:
            best_params = {'max_depth': 20, 'max_features': 0.3, 'min_samples_split': 2, 'n_estimators': 250}
        model, _ = define_model(pred_model=pred_model, fixed_seed=seed)

    model.set_params(**best_params)
    model.fit(X_train, y_train)

    # Predict OOS test data
    y_pred = model.predict(X_test)
    test_r2 = round(metrics.r2_score(y_test, y_pred), decimal_places)
    print(f'Model performance on unseen test data: {test_r2} Prediction R2')
    # ------------------------------------------ Explanations ------------------------------------------

    ## Partial Dependence Plot (PDP)
    f1="X"
    f2="Z"
    # first 2D, then 3D plot (on same figure):
    features = [f1, (f1, f2)]
    save_path= results_path + "PDP/"
    X_test = pd.DataFrame(X_test, columns=vars)
    plot_PDP(save_path=save_path, pred_model=pred_model, model=model, X_test=X_test, features=features)
    # just 2D:
    features = [f1]
    plot_PDP(save_path=save_path, pred_model=pred_model, model=model, X_test=X_test, features=features,
             save_name="pdp_2D", ylim=(-11, 11))

    ## Individual Conditional Expectation (ICE) plot
    save_path= results_path + "ICE/"
    X_test.columns.values[0] = 'X'
    plot_ICE(save_path=save_path, pred_model=pred_model, model=model, X_test=X_test, feature='X',
             xlab='X', ylab='Partial Dependence', figsize=(5, 3.5))

    ## Accumulated Local Effects (ALE) graph
    save_path = results_path + "ALE/"
    grid = 50
    # a) one variable:
    ale_eff = ale(X=X_test, model=model, feature=['X'], grid_size=grid, include_CI=False)
    plt.savefig(save_path + f"{pred_model}_1D_ale.png")

    run_time = dt.datetime.now() - start_time
    print(f'{pred_model} finished! Run time: {run_time}')
