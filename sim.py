import seaborn as sns
import numpy as np
import pandas as pd
import shap as shap
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, GridSearchCV
from Functions.gen_data import add_noise, extract_coef
from Functions.plotting import plot_impurity, plot_permutation, plot_SHAP, plot_SHAP_force, plot_PDP, plot_ICE, \
    check_corr
from Functions.pred import define_model
from sklearn import tree

# Define the changeable parameters:
seed = 93
pred_model = "enet"
n_samples = 100
n_features = 5
iv_cor = 0.6
dv_r2 = 0.6
mean = 0
sd = 1
test_size = 0.5
cv = 5
scoring = "r2"
decimal_places = 2
permutations = 10
# -------------------------- Generate Data --------------------------

if __name__ == '__main__':
    print("Running analysis for {} model".format(pred_model))

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

    X_and_y = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)

    # Check correlations
    save_path = "Outputs/"
    check_corr(X_and_y=X_and_y, X_feature_names=X_feature_names, save_path=save_path,
               save_name="data_cor_plot", decimal_places=decimal_places)

    # Extract coefficients
    extract_coef(X=X, y=y, X_feature_names=X_feature_names, decimal_places=decimal_places)

    # -------------------------- Train Model and Evaluate OOS --------------------------

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

    # Predict test data
    y_pred = model.predict(X_test)
    test_r2 = round(metrics.r2_score(y_test, y_pred), decimal_places)
    print('Model performance on unseen test data: {} Prediction R2'.format(test_r2))

    # -------------------------- Interpretation --------------------------
    vars = X_feature_names.copy()

    # 1) Permutation importance
    save_path = "Results/Interpretation/Permutation/"
    result = permutation_importance(model, X_test, y_test, n_repeats=permutations,
                                    random_state=seed, n_jobs=2, scoring=scoring)
    perm_importances_mean = result.importances_mean
    dict = {'Feature': vars, "Importance": perm_importances_mean}
    perm_imp_df = pd.DataFrame(dict)
    # flip so most important at top on graph
    perm_imp_df.sort_values(by="Importance", ascending=True, inplace=True, axis=0)
    # plot
    plot_permutation(perm_imp_df=perm_imp_df,
                     save_path=save_path,
                     save_name="{}_permutation".format(pred_model))

    # Plot with bars:
    sorted_indices = result.importances_mean.argsort()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    plt.barh(range(len(vars)), result.importances_mean[sorted_indices],
             xerr=result.importances_std[sorted_indices], color="dodgerblue")
    plt.yticks(range(len(vars)), np.array(vars)[sorted_indices])
    plt.xlabel('Importance')
    plt.title('Permutation Importances (test set)')
    plt.tight_layout()
    plt.savefig(save_path +'{}_bars_permutation.png'.format(pred_model))
    plt.clf()
    plt.cla()
    plt.close()

    # 2) Tree-based impurity importance
    if (pred_model == 'rf') or (pred_model == "tree"):
        save_path = "Results/Interpretation/Impurity/"
        feature_importances = model.feature_importances_

        dict = {'Feature': vars, "Importance": feature_importances}
        impurity_imp_df = pd.DataFrame(dict)

        # flip so most important at top on graph
        impurity_imp_df.sort_values(by="Importance", ascending=True, inplace=True, axis=0)

        # plot
        plot_impurity(impurity_imp_df=impurity_imp_df,
                      save_path=save_path,
                      save_name="{}_impurity".format(pred_model))

    # 3) SHAP importance
    save_path = "Results/Interpretation/SHAP/"
    if (pred_model == "rf") or (pred_model == "tree"):
        explainer = shap.TreeExplainer(model, feature_pertubation="tree_path_dependent")
        shap_dict = explainer(X_test)
        shap_values = explainer.shap_values(X_test)
        shap_values_df = pd.DataFrame(shap_values, columns=vars)

    elif (pred_model == "enet") or (pred_model == "lasso"):
        explainer = shap.LinearExplainer(model, X_test, feature_pertubation="correlation_dependent")
        shap_dict = explainer(X_test)
        shap_values = explainer.shap_values(X_test)
        shap_values_df = pd.DataFrame(shap_values, columns=vars)
    else:
        print("please enter one of the regression or tree based models: 'rf', 'tree', 'lasso, or 'enet'")

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

    # example of local plot:
    plot_SHAP_force(i=0, X_test=pd.DataFrame(X_test, columns=vars), model=model, save_path=save_path,
                    save_name="{}_shap_local".format(pred_model), pred_model=pred_model,
                    title="")

    # 4) Partial Dependence Plot (PDP)
    # select features to plot based on permutation importance
    perm_imp_df.sort_values(by="Importance", ascending=False, inplace=True, axis=0)
    f1=perm_imp_df['Feature'].iloc[0]
    f2=perm_imp_df['Feature'].iloc[1]
    # first 2D, then 3D plot:
    features = [f1, (f1, f2)]
    save_path="Results/Interpretation/PDP/"
    X_test = pd.DataFrame(X_test, columns=vars)
    plot_PDP(save_path=save_path, pred_model=pred_model, model=model, X_test=X_test, features=features)

    # 5) Individual Conditional Expectation (ICE)
    save_path="Results/Interpretation/ICE/"
    plot_ICE(save_path=save_path, pred_model=pred_model, model=model, X_test=X_test, features=features)

    # 6) Print Decision Tree structure (for visual purposes only)
    if (pred_model == 'rf') or (pred_model == "tree"):
        # define meta-parameters (for visual purposes only)
        max_depth = 3

        # define tree
        dec_tree = tree.DecisionTreeRegressor(max_depth=max_depth)

        # fit tree
        dec_tree.fit(X_test, y_test)

        fig = plt.figure(figsize=(10, 4))
        _ = tree.plot_tree(dec_tree,
                           feature_names=vars,
                           class_names=True,
                           filled=True,
                           fontsize=8,
                           precision=1,
                           rounded=True)
        save_path = "Results/Interpretation/Tree/"
        plt.tight_layout()
        plt.savefig(save_path + "example_dt_structure.png")


    print('done')