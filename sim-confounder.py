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
    check_corr, print_tree, plot_multiple_permutations, plot_SHAP_ordered
from Functions.pred import define_model, define_model_confounder
from PyALE import ale

# =================================
# Define the changeable parameters:
# =================================

# supported model classes for pred_model: "enet" for elastic net regression, "lasso" for lasso regression,
# ..."tree" for a decision tree, and "rf" for a random forest

pred_models = ["rf", "lasso", "enet", "tree"] # string defining the prediction model to use (see above for alternatives)
n_samples = 1000 # number of samples in generated data
test_size = 0.5 # ratio of training:test data
cv = 5 # number of cross-validation splits
scoring = "r2" # scoring used for both the training and the testing: 'r2' is prediction R-squared, for other options, see: https://scikit-learn.org/stable/modules/model_evaluation.html
permutations = 10 # number of permutations in permutation importance calculations
explain_data_instance_num = 0 # the row index indicating which instance in the data to create a local explanation for (used for SHAP and LIME)
decimal_places = 2 # integer used for rounding
seed = 93 # the random seed (used in the data generating process, splitting process, the model fitting process, and the permutation importance calculations)
results_path = "Results confounder-sim/Interpretation/"
force_max_features_1 = True
# -------------------------- Run Simulations --------------------------

if __name__ == '__main__':
    start_time = dt.datetime.now()

    # -------------------------- Generate Data --------------------------

    # Generate random variables X1, X2, X3
    np.random.seed(seed)
    X1 = np.random.normal(size=n_samples)
    X2 = np.random.normal(size=n_samples)
    X3 = np.random.normal(size=n_samples)

    # Y is influenced by X1 and X2 but not X3
    y = 2 * X1 + 3 * X2 + np.random.normal(size=n_samples)

    # Create a dataframe with these variables
    data = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'y': y})

    # Correlations between variables
    print("Initial correlations between variables:")
    correlation_matrix = data.corr()
    print(correlation_matrix)

    # Increase correlation between X3 and X1/X2
    data['X3'] = data['X1'] + data['X2'] + X3

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
        model, param_grid = define_model_confounder(pred_model=pred_model, fixed_seed=seed)

        if force_max_features_1 == True:
            if (pred_model == "tree") or (pred_model == "rf"):
                param_grid["max_features"] = [None]
                mf = "_mf1"
            else:
                mf = ""
        else:
            mf = ""
        print(param_grid)
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
        cv_r2 = round(grid_search.best_score_, decimal_places)
        print(f'Model performance on validation data: {cv_r2} Prediction R2')

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
        plot_permutation(perm_imp_df=perm_imp_df, save_path=save_path, save_name=f"{pred_model}_permutation{mf}")

        # b) multiple permutations (plot with variance bars)
        result = permutation_importance(model, X_test, y_test, n_repeats=permutations,
                                        random_state=seed, n_jobs=2, scoring=scoring)
        plot_multiple_permutations(result=result, vars=vars, save_path=save_path, order=True,
                                   save_name=f"{pred_model}_permutation_with_bars{mf}.png", title="")

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
                          save_name=f"{pred_model}_impurity{mf}")

        ## 3) SHAP importance
        save_path = results_path + "SHAP/"
        shap_results_dict = {}
        if (pred_model == "rf") or (pred_model == "tree"):
            method_types = ["tree_path_dependent", "interventional"]# (see Lundberg et al., 2020) https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
            data_options = [None, X_test]
            for method_type, data_option in zip(method_types, data_options):
                explainer = shap.TreeExplainer(model=model, data=data_option, feature_perturbation=method_type)
                shap_values = explainer.shap_values(X_test)
                shap_values_df = pd.DataFrame(shap_values, columns=vars)
                shap_values_df_abs = abs(shap_values_df)
                shap_importance_df = shap_values_df_abs.mean(axis=0).reset_index()
                shap_importance_df.columns = ["Feature", "Importance"]
                # don't plot constant
                shap_importance_df = shap_importance_df.loc[1:3]
                shap_results_dict[method_type] = shap_importance_df

        elif (pred_model == "enet") or (pred_model == "lasso"):
            method_types = ["correlation_dependent", "interventional"] # (see Lundberg & Lee, 2017) https://shap-lrjball.readthedocs.io/en/latest/generated/shap.LinearExplainer.html
            data_options = [X_test, X_test]
            for method_type, data_option in zip(method_types, data_options):
                explainer = shap.LinearExplainer(model, data_option, feature_perturbation=method_type)
                shap_values = explainer.shap_values(X_test)
                shap_values_df = pd.DataFrame(shap_values, columns=vars)
                shap_values_df_abs = abs(shap_values_df)
                shap_importance_df = shap_values_df_abs.mean(axis=0).reset_index()
                shap_importance_df.columns = ["Feature", "Importance"]
                # don't plot constant
                shap_importance_df = shap_importance_df.loc[1:3]
                shap_results_dict[method_type] = shap_importance_df

        # a) Plot summary "global" bar plots:
        for method, shap_dict in shap_results_dict.items():
            plot_SHAP_ordered(shap_dict,
                      save_path= save_path,
                      save_name=f"{pred_model}_shap_ORDERED_{method}{mf}")

        # b) Example of SHAP local force plot for data instance i:
        plot_SHAP_force(i=explain_data_instance_num, X_test=pd.DataFrame(X_test, columns=vars), model=model,
                        save_path=save_path, save_name=f"{pred_model}_shap_local{mf}", pred_model=pred_model,
                        title="local explanation")

        ## 4) Partial Dependence Plot (PDP)
        # look at importance of X2 (confounder)
        f1="X2"
        f2="X3"
        # first 2D, then 3D plot (on same figure):
        features = [f1, (f1, f2)]
        save_path= results_path + "PDP/"
        X_test = pd.DataFrame(X_test, columns=vars)
        plot_PDP(save_path=save_path, pred_model=pred_model, model=model, X_test=X_test,
                 features=features, save_name=f"pdp{mf}")

        ## 5) Individual Conditional Expectation (ICE) plot
        save_path= results_path + "ICE/"
        save_name = f"{pred_model}_ice{mf}.png"
        plot_ICE(save_path=save_path, pred_model=pred_model, model=model, X_test=X_test,
                 feature=f1, save_name=save_name)

        ## 6) Accumulated Local Effects (ALE) graph
        save_path = results_path + "ALE/"
        grid = 50
        # a) one variable:
        ale_eff = ale(X=X_test, model=model, feature=[f1], grid_size=grid, include_CI=False)
        plt.savefig(save_path + f"{pred_model}_1D_ale{mf}.png")
        # b) two variables:
        ale_eff_2D = ale(X=X_test, model=model, feature=[f1, f2], grid_size=grid)
        plt.savefig(save_path + f"{pred_model}_2D_ale{mf}.png")

        ## 7) Local Interpretable Model-agnostic Explanations (LIME)
        save_path = results_path + "LIME/"
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=vars, mode="regression",
                                                                verbose=False, discretize_continuous=False,
                                                                random_state=seed)
        X_instance = X_test.iloc[explain_data_instance_num, :]
        exp = lime_explainer.explain_instance(data_row=X_instance, predict_fn=model.predict, num_features=len(vars))
        # Create a plot from the explanation
        fig = exp.as_pyplot_figure()
        plt.tight_layout()
        plt.savefig(save_path + f'{pred_model}_lime{mf}.png')

        ## 8) Print decision tree structure (here, for visual purposes only)
        if (pred_model == 'rf') or (pred_model == "tree"):
            save_path = results_path + "Tree/"
            # pre-define depth parameter (as tree is for visual purposes only)
            print_tree(X_test=X_test, y_test=y_test, max_depth=3, feature_names=vars,
                       fontsize=8, save_path=save_path, save_name=f"example_dt_structure{mf}")

    run_time = dt.datetime.now() - start_time
    print(f'Finished! Run time: {run_time}')
