from random import random
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import shap as shap
import datetime as dt
import matplotlib.pyplot as plt
import statsmodels.api as sm
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

pred_models = ["lasso", "enet", "rf", "tree"] # string defining the prediction model to use (see above for alternatives)
n_samples = 1000 # number of samples in generated data
test_size = 0.5 # ratio of training:test data
cv = 5 # number of cross-validation splits
scoring = "r2" # scoring used for both the training and the testing: 'r2' is prediction R-squared, for other options, see: https://scikit-learn.org/stable/modules/model_evaluation.html
permutations = 10 # number of permutations in permutation importance calculations
explain_data_instance_num = 0 # the row index indicating which instance in the data to create a local explanation for (used for SHAP and LIME)
decimal_places = 2 # integer used for rounding
seed = 93 # the random seed (used in the data generating process, splitting process, the model fitting process, and the permutation importance calculations)
results_path = "Results confounder-sim/Interpretation/"
# -------------------------- Run Simulations --------------------------

if __name__ == '__main__':
    start_time = dt.datetime.now()

    # -------------------------- Generate Data --------------------------

    # Generate random variables X1, X2, X3
    np.random.seed(seed)
    X1 = np.random.normal(size=n_samples)
    X2 = np.random.normal(size=n_samples)
    X3 = np.random.normal(size=n_samples)

    # Y is influenced by X1 and X3 but not X2
    y = 2 * X1 + 3 * X3 + np.random.normal(size=n_samples)

    # Create a dataframe with these variables
    data = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'y': y})

    # Correlations between variables
    print("Initial correlations between variables:")
    correlation_matrix = data.corr()
    print(correlation_matrix)

    # Increase correlation between X2 and X1/X3
    data['X2'] = data['X1'] + data['X3'] + X2

    # Check correlations after manipulation
    print("Correlations between variables after manipulation:")
    correlation_matrix_modified = data.corr()
    print(correlation_matrix_modified)

    # Linear regression
    X = data[['X1', 'X2', 'X3']]
    n_features = X.shape[1]
    X = sm.add_constant(X)  # Add a constant (intercept) term
    y = data['y']

    model = sm.OLS(y, X).fit()
    print(model.summary())

    for pred_model in pred_models:
        print(f"Running analysis for {pred_model} model...")

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
        print(f"best hyper-params from training: {best_params}")
        model.set_params(**best_params)
        model.fit(X_train, y_train)
        cv_r2 = grid_search.best_score_

        # Predict OOS test data
        y_pred = model.predict(X_test)
        test_r2 = round(metrics.r2_score(y_test, y_pred), decimal_places)
        print(f'Model performance on unseen test data: {test_r2} Prediction R2')

        # ------------------------------------------ Explanations ------------------------------------------

        vars = list(X.columns)

        ## 1 Permutation importance
        # a) single permutation
        save_path = results_path + "Permutation/"
        result = permutation_importance(model, X_test, y_test, n_repeats=1, # a single imputation
                                        random_state=seed, n_jobs=2, scoring=scoring)
        perm_importances_mean = result.importances_mean
        dict = {'Feature': vars, "Importance": perm_importances_mean}
        perm_imp_df = pd.DataFrame(dict)
        # flip so most important at top on graph
        perm_imp_df.sort_values(by="Importance", ascending=True, inplace=True, axis=0)
        plot_permutation(perm_imp_df=perm_imp_df, save_path=save_path, save_name=f"{pred_model}_permutation")

        # b) multiple permutations (plot with variance bars)
        result = permutation_importance(model, X_test, y_test, n_repeats=permutations,
                                        random_state=seed, n_jobs=2, scoring=scoring)
        plot_multiple_permutations(result=result, vars=vars, save_path=save_path,
                                   save_name=f"{pred_model}_permutation_with_bars.png")

        ## 2) Tree-based impurity importance
        if (pred_model == 'rf') or (pred_model == "tree"):
            save_path = results_path + "Impurity/"
            feature_importances = model.feature_importances_

            dict = {'Feature': vars, "Importance": feature_importances}
            impurity_imp_df = pd.DataFrame(dict)

            # flip so most important features at the top of the graph
            impurity_imp_df.sort_values(by="Importance", ascending=True, inplace=True, axis=0)

            # plot
            plot_impurity(impurity_imp_df=impurity_imp_df,
                          save_path=save_path,
                          save_name=f"{pred_model}_impurity")

        ## 3) SHAP importance
        save_path = results_path + "SHAP/"
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

        # a) Plot summary "global" plots:
        plot_types = ["bar", "summary", "violin"]
        for plot_type in plot_types:
            plot_SHAP(shap_dict, col_list=vars,
                      n_features=n_features, plot_type=plot_type,
                      save_path= save_path, title="SHAP importance (test set)",
                      save_name=f"{pred_model}_shap_{plot_type}.png")
            if plot_type == "summary":
                plot_SHAP(shap_dict, col_list=vars,
                          n_features=n_features, plot_type=None,
                          save_path= save_path, title="SHAP importance (test set)",
                          save_name=f"{pred_model}_shap_{plot_type}.png")

        # b) Example of SHAP local force plot for data instance i:
        plot_SHAP_force(i=explain_data_instance_num, X_test=pd.DataFrame(X_test, columns=vars), model=model,
                        save_path=save_path, save_name=f"{pred_model}_shap_local", pred_model=pred_model,
                        title="local explanation")

        ## 4) Partial Dependence Plot (PDP)
        # look at importance of X2 (confounder)
        f1="X2"
        f2="X3"
        # first 2D, then 3D plot (on same figure):
        features = [f1, (f1, f2)]
        save_path= results_path + "PDP/"
        X_test = pd.DataFrame(X_test, columns=vars)
        plot_PDP(save_path=save_path, pred_model=pred_model, model=model, X_test=X_test, features=features)

        ## 5) Individual Conditional Expectation (ICE) plot
        save_path= results_path + "ICE/"
        plot_ICE(save_path=save_path, pred_model=pred_model, model=model, X_test=X_test, feature=f1)

        ## 6) Accumulated Local Effects (ALE) graph
        save_path = results_path + "ALE/"
        grid = 50
        # a) one variable:
        ale_eff = ale(X=X_test, model=model, feature=[f1], grid_size=grid, include_CI=False)
        plt.savefig(save_path + f"{pred_model}_1D_ale.png")
        # b) two variables:
        ale_eff_2D = ale(X=X_test, model=model, feature=[f1, f2], grid_size=grid)
        plt.savefig(save_path + f"{pred_model}_2D_ale.png")

        ## 7) Local Interpretable Model-agnostic Explanations (LIME)
        save_path = results_path + "LIME/"
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=vars, mode="regression",
                                                                verbose=False, discretize_continuous=False)
        X_instance = X_test.iloc[explain_data_instance_num, :]
        exp = lime_explainer.explain_instance(data_row=X_instance, predict_fn=model.predict, num_features=len(vars))
        # Create a plot from the explanation
        fig = exp.as_pyplot_figure()
        plt.tight_layout()
        plt.savefig(save_path + f'{pred_model}_lime.png')

        ## 8) Print decision tree structure (here, for visual purposes only)
        if (pred_model == 'rf') or (pred_model == "tree"):
            save_path = results_path + "Tree/"
            # pre-define depth parameter (as tree is for visual purposes only)
            print_tree(X_test=X_test, y_test=y_test, max_depth=3, feature_names=vars,
                       fontsize=8, save_path=save_path, save_name="example_dt_structure")

    run_time = dt.datetime.now() - start_time
    print(f'Finished! Run time: {run_time}')