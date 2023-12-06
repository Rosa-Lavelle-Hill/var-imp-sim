from random import random

import seaborn as sns
import numpy as np
import pandas as pd
import shap as shap
import datetime as dt
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, GridSearchCV
from Functions.gen_data import add_noise, extract_coef
from Functions.plotting import plot_impurity, plot_permutation, plot_SHAP, plot_SHAP_force, plot_PDP, plot_ICE, \
    check_corr, print_tree, plot_multiple_permutations
from Functions.pred import define_model
from PyALE import ale

# =================================
# Define the changeable parameters:
# =================================

# supported model classes for pred_model: "enet" for elastic net regression, "lasso" for lasso regression,
# ..."tree" for a decision tree, and "rf" for a random forest

pred_model = "rf" # string defining the prediction model to use (see above for alternatives)
n_samples = 100 # number of samples in generated data
n_features = 5 # number of features (or "independent variables") in generated data
mean = 0 # mean of generated data
sd = 1 # standard deviation of generated data
iv_cor = 0.6 # the Pearson r correlation of the features with one another in the generated data
dv_r2 = 0.6 # the R-squared value representing the extent the features can explain y in the full generated dataset
test_size = 0.5 # ratio of training:test data
cv = 5 # number of cross-validation splits
scoring = "r2" # scoring used for both the training and the testing: 'r2' is prediction R-squared, for other options, see: https://scikit-learn.org/stable/modules/model_evaluation.html
permutations = 10 # number of permutations in permutation importance calculations
force_plot_data_instance_num = 0 # the row number indicating which person in the data to create a SHAP force plot for
decimal_places = 2 # integer used for rounding
seed = 93 # the random seed (used in the data generating process, splitting process, the model fitting process, and the permutation importance calculations)

# -------------------------- Run Simulations --------------------------

if __name__ == '__main__':
    start_time = dt.datetime.now()
    print("Running analysis for {} model...".format(pred_model))

    # -------------------------- Generate Data --------------------------

    # Define the mean and standard deviation for each variable
    means = [mean] * n_features
    stds = [sd] * n_features

    # Define the correlation matrix
    ones = np.ones((n_features, n_features))
    corr_matrix = iv_cor * ones + (1 - iv_cor) * np.eye(n_features)

    # Generate X
    np.random.seed(seed)
    X = np.random.multivariate_normal(mean=means, cov=corr_matrix, size=n_samples)

    X_feature_names = []
    for i in list(range(1, X.shape[1] + 1)):
        n = "X{}".format(i)
        X_feature_names.append(n)

    # Predict y from X with fixed coefficients (b=1)
    y_pred = np.dot(X, np.ones((X.shape[1],)))

    # Add noise to y_pred so that X predicts y with a given r2
    y, iters_count = add_noise(y_pred, dv_r2)

    # Check correlations
    save_path = "Outputs/"
    X_and_y = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)
    check_corr(X_and_y=X_and_y, X_feature_names=X_feature_names, save_path=save_path,
               save_name="data_cor_plot", decimal_places=decimal_places)

    # Extract coefficients
    extract_coef(X=X, y=y, X_feature_names=X_feature_names, decimal_places=decimal_places)

    # -------------------------- Train Model and Evaluate Out-Of-Sample (OOS) --------------------------

    # Split train and test (same random seed so constant stable comparison)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size, shuffle=True)

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
    print("best hyper-params from training: {}".format(best_params))
    model.set_params(**best_params)
    model.fit(X_train, y_train)
    cv_r2 = grid_search.best_score_

    # Predict OOS test data
    y_pred = model.predict(X_test)
    test_r2 = round(metrics.r2_score(y_test, y_pred), decimal_places)
    print('Model performance on unseen test data: {} Prediction R2'.format(test_r2))

    # -------------------------- Explanations --------------------------

    vars = X_feature_names.copy()

    # 1a) Permutation importance (a single imputation)
    save_path = "Results/Interpretation/Permutation/"
    result = permutation_importance(model, X_test, y_test, n_repeats=1, # a single imputation
                                    random_state=seed, n_jobs=2, scoring=scoring)
    perm_importances_mean = result.importances_mean
    dict = {'Feature': vars, "Importance": perm_importances_mean}
    perm_imp_df = pd.DataFrame(dict)
    # flip so most important at top on graph
    perm_imp_df.sort_values(by="Importance", ascending=True, inplace=True, axis=0)
    plot_permutation(perm_imp_df=perm_imp_df, save_path=save_path, save_name="{}_permutation".format(pred_model))

    # 1b) Plot multiple permutations with with variance bars:
    result = permutation_importance(model, X_test, y_test, n_repeats=permutations,
                                    random_state=seed, n_jobs=2, scoring=scoring)
    plot_multiple_permutations(result=result, vars=vars, save_path=save_path,
                               save_name=f"{pred_model}_permutation_with_bars.png")

    # 2) Tree-based impurity importance
    if (pred_model == 'rf') or (pred_model == "tree"):
        save_path = "Results/Interpretation/Impurity/"
        feature_importances = model.feature_importances_

        dict = {'Feature': vars, "Importance": feature_importances}
        impurity_imp_df = pd.DataFrame(dict)

        # flip so most important features at the top of the graph
        impurity_imp_df.sort_values(by="Importance", ascending=True, inplace=True, axis=0)

        # plot
        plot_impurity(impurity_imp_df=impurity_imp_df,
                      save_path=save_path,
                      save_name="{}_impurity".format(pred_model))

    # 3) SHAP importance
    save_path = "Results/Interpretation/SHAP/"
    if (pred_model == "rf") or (pred_model == "tree"):
        explainer = shap.TreeExplainer(model, feature_pertubation="tree_path_dependent")# "true to the data" approach (see Lundberg et al., 2020)
        shap_dict = explainer(X_test)
        shap_values = explainer.shap_values(X_test)
        shap_values_df = pd.DataFrame(shap_values, columns=vars)

    elif (pred_model == "enet") or (pred_model == "lasso"):
        explainer = shap.LinearExplainer(model, X_test, feature_pertubation="correlation_dependent") # "true to the data" approach (see Lundberg & Lee, 2017)
        shap_dict = explainer(X_test)
        shap_values = explainer.shap_values(X_test)
        shap_values_df = pd.DataFrame(shap_values, columns=vars)
    else:
        print("please enter one of the regression or tree based models: 'rf', 'tree', 'lasso, or 'enet'")

    # 3a) Plot summary "global" plots:
    plot_types = ["bar", "summary", "violin"]
    for plot_type in plot_types:
        plot_SHAP(shap_dict, col_list=vars,
                  n_features=n_features, plot_type=plot_type,
                  save_path= save_path, title="SHAP importance (test set)",
                  save_name="{}_shap_{}.png".format(pred_model, plot_type))
        if plot_type == "summary":
            plot_SHAP(shap_dict, col_list=vars,
                      n_features=n_features, plot_type=None,
                      save_path= save_path, title="SHAP importance (test set)",
                      save_name="{}_shap_{}.png".format(pred_model, plot_type))

    # 3b) Example of SHAP local force plot for data instance i:
    plot_SHAP_force(i=force_plot_data_instance_num, X_test=pd.DataFrame(X_test, columns=vars), model=model, save_path=save_path,
                    save_name="{}_shap_local".format(pred_model), pred_model=pred_model,
                    title="")

    # 4) Partial Dependence Plot (PDP)
    # select features to plot automatically based on permutation importance (most important)
    perm_imp_df.sort_values(by="Importance", ascending=False, inplace=True, axis=0)
    f1=perm_imp_df['Feature'].iloc[0] # to specify a feature, substitute for: f1='feature_name'
    f2=perm_imp_df['Feature'].iloc[1]
    # first 2D, then 3D plot (on same figure):
    features = [f1, (f1, f2)]
    save_path="Results/Interpretation/PDP/"
    X_test = pd.DataFrame(X_test, columns=vars)
    plot_PDP(save_path=save_path, pred_model=pred_model, model=model, X_test=X_test, features=features)

    # 5) Individual Conditional Expectation (ICE) plot
    save_path="Results/Interpretation/ICE/"
    plot_ICE(save_path=save_path, pred_model=pred_model, model=model, X_test=X_test, feature=f1)

    # 6) Accumulated Local Effects (ALE) graph
    save_path = "Results/Interpretation/ALE/"
    grid = 50
    # a) one variable:
    ale_eff = ale(X=X_test, model=model, feature=[f1], grid_size=grid, include_CI=False)
    plt.savefig(save_path + f"{pred_model}_1D_ale.png")
    # b) two variables:
    ale_eff_2D = ale(X=X_test, model=model, feature=[f1, f2], grid_size=grid)
    plt.savefig(save_path + f"{pred_model}_2D_ale.png")

    # 7) Print decision tree structure (here, for visual purposes only)
    if (pred_model == 'rf') or (pred_model == "tree"):
        save_path = "Results/Interpretation/Tree/"
        # pre-define depth parameter (as tree is for visual purposes only)
        print_tree(X_test=X_test, y_test=y_test, max_depth=3, feature_names=vars,
                   fontsize=8, save_path=save_path, save_name="example_dt_structure")

    run_time = dt.datetime.now() - start_time
    print(f'Finished! Run time: {run_time}')
# todo: requirements file