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
from Functions.gen_data import add_noise, extract_coef
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
results_path = "Results figure-C/Interpretation/"
outputs_path = "Outputs figure-C/"
replicate_figure_model = True
# -------------------------- Run Simulations --------------------------

start_time = dt.datetime.now()
for pred_model in ["rf"]:
    print(f"Running model: {pred_model}")

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

    # Predict y from X with fixed coefficients (b=1)
    y_pred = np.dot(X, np.ones((X.shape[1],)))

    # Add noise to y_pred so that X predicts y with a given r2
    y, iters_count = add_noise(y_pred, dv_r2)

    def quadratic_function(x, a, b, c):
        return a * (x - b) ** 2 + c

    # Define parameters for the quadratic function
    a = 1  # Determines the width of the U-shape
    b = 0  # Determines the position of the vertex of the U-shape
    c = 2  # Determines the vertical shift of the U-shape

    # Calculate corresponding X values
    X_values = quadratic_function(y, a, b, c)

    # Convert the list of x values to a NumPy array
    X6 = np.array(X_values)
    plt.scatter(X6, y, alpha=0.5)
    plt.xlabel('X6', fontsize=20)
    plt.ylabel('Y', fontsize=20)
    plt.title('')
    plt.savefig(outputs_path + "U-plot.png")
    plt.close()

    X_feature_names = []
    for i in list(range(1, X.shape[1] + 1)):
        n = f"X{i}"
        X_feature_names.append(n)

    # Change to dataframes
    X = pd.DataFrame(X, columns=X_feature_names)
    X['X6'] = X6
    n_features = n_features + 1
    X_feature_names = list(X.columns)
    y = pd.Series(y, name="y")

    # Check correlations
    X_and_y = pd.concat([X, y], axis=1)
    check_corr(X_and_y=X_and_y, X_feature_names=X_feature_names, save_path=outputs_path,
               save_name="data_cor_plot", decimal_places=decimal_places)

    # Extract coefficients
    extract_coef(X=X, y=y, X_feature_names=X_feature_names, decimal_places=decimal_places, file_path=outputs_path)

    # -------------------------- Train Model and Evaluate Out-Of-Sample (OOS) --------------------------

    # Split train and test (same random seed so constant stable comparison)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size, shuffle=True)

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

    vars = X_feature_names.copy()

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
        if shap_method == "correlation_dependent":
            explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        else:
            explainer = shap.TreeExplainer(model, X_test, feature_perturbation=shap_method)
        shap_dict = explainer(X_test)
        shap_values = explainer.shap_values(X_test)
        shap_values_df = pd.DataFrame(shap_values, columns=vars)

    elif (pred_model == "enet") or (pred_model == "lasso"):
        explainer = shap.LinearExplainer(model, X_test, feature_perturbation=shap_method)
        shap_dict = explainer(X_test)
        shap_values = explainer.shap_values(X_test)
        shap_values_df = pd.DataFrame(shap_values, columns=vars)

    # a) Plot summary "global" plots:
    plot_types = ["bar", "summary", "violin"]
    for plot_type in plot_types:
        plot_SHAP(shap_dict, col_list=vars,
                  n_features=n_features, plot_type=plot_type,
                  save_path= save_path, title="SHAP importance (test set)",
                  save_name=f"{pred_model}_shap_{plot_type}_{shap_method}.png")
        if plot_type == "summary":
            plot_SHAP(shap_dict, col_list=vars,
                      n_features=n_features, plot_type=None,
                      save_path= save_path, title="SHAP importance (test set)",
                      save_name=f"{pred_model}_shap_{plot_type}_{shap_method}.png")

    # b) Example of SHAP local force plot for data instance i:
    plot_SHAP_force(i=explain_data_instance_num, X_test=pd.DataFrame(X_test, columns=vars), model=model,
                    save_path=save_path, save_name=f"{pred_model}_shap_local_{shap_method}", pred_model=pred_model,
                    title="local explanation")

    # c) Plot SHAP value against feature value:
    plt.scatter(X_test.X6, shap_values_df.X6, alpha=0.5)
    plt.xlabel('X6', fontsize=20)
    plt.ylabel('SHAP values', fontsize=20)
    plt.title('')
    plt.savefig(save_path + f"{pred_model}_U-plot_{shap_method}.png")
    plt.close()

    ## 4) Partial Dependence Plot (PDP)
    # select features to plot automatically based on permutation importance (most important)
    perm_imp_df.sort_values(by="Importance", ascending=False, inplace=True, axis=0)
    f1="X6"
    f2="X2"
    # first 2D, then 3D plot (on same figure):
    features = [f1, (f1, f2)]
    save_path= results_path + "PDP/"
    X_test = pd.DataFrame(X_test, columns=vars)
    plot_PDP(save_path=save_path, pred_model=pred_model, model=model, X_test=X_test, features=features)
    # just 2D:
    features = [f1]
    plot_PDP(save_path=save_path, pred_model=pred_model, model=model, X_test=X_test, features=features,
             save_name="pdp_2D", ylim=(-11, 11))

    ## 5) Individual Conditional Expectation (ICE) plot
    save_path= results_path + "ICE/"
    plot_ICE(save_path=save_path, pred_model=pred_model, model=model, X_test=X_test, feature=f1)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    PartialDependenceDisplay.from_estimator(model, X_test, ['X6'], kind='both',
                                            pd_line_kw={"color":"red", "linestyle":'dashed', "linewidth":3})
    ax.set_ylabel('PD', fontsize=20)
    ax.set_xlabel('X6', fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path + "rf_ice_X6_col.png", bbox_inches='tight')

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
    # initilise LIME on train data
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=vars, mode="regression",
                                                            verbose=False, discretize_continuous=False,
                                                            random_state=seed)
    # explain 3 different instances to show difference
    instances = [explain_data_instance_num, explain_data_instance_num+1, explain_data_instance_num+2]
    for instance in instances:
        # explain instance of test data
        X_instance = X_test.iloc[instance, :]
        exp = lime_explainer.explain_instance(data_row=X_instance, predict_fn=model.predict, num_features=len(vars))
        # Create a plot from the explanation
        fig = exp.as_pyplot_figure()
        plt.tight_layout()
        plt.savefig(save_path + f'{pred_model}_lime_x{instance}.png')

    ## 8) Print decision tree structure (here, for visual purposes only)
    if (pred_model == 'rf') or (pred_model == "tree"):
        save_path = results_path + "Tree/"
        # pre-define depth parameter (as tree is for visual purposes only)
        print_tree(X_test=X_test, y_test=y_test, max_depth=3, feature_names=vars,
                   fontsize=8, save_path=save_path, save_name="example_dt_structure")

    run_time = dt.datetime.now() - start_time
    print(f'{pred_model} finished! Run time: {run_time}')
