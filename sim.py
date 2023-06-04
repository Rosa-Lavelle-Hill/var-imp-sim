import numpy as np
import datetime as dt
import numpy as np
import pandas as pd
import shap as shap
from sklearn import metrics
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, GridSearchCV
from Functions.gen_data import add_noise

# Define the parameters
from Functions.plotting import plot_impurity, plot_permutation, plot_SHAP
from Functions.pred import define_model

pred_model = "rf"
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
seed = 93
permutations = 10
# -------------------------- Generate Data --------------------------

# Define the mean and standard deviation for each variable
means = [mean] * n_features
stds = [sd] * n_features

# Define the correlation matrix
ones = np.ones((n_features, n_features))
corr_matrix = iv_cor * ones + (1 - iv_cor) * np.eye(n_features)

# Generate X
X = np.random.multivariate_normal(mean=means, cov=corr_matrix, size=n_samples)

X_col_names = []
for i in list(range(1, X.shape[1] + 1)):
    n = "X{}".format(i)
    X_col_names.append(n)

# Predict y from X with fixed coefficients (b=1)
y_pred = np.dot(X, np.ones((X.shape[1],)))

# Add noise to y_pred so that X predicts y with a given r2
y, iters_count = add_noise(y_pred, dv_r2)

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
model.set_params(**best_params)
model.fit(X_train, y_train)
cv_r2 = grid_search.best_score_

# Predict test data
y_pred = model.predict(X_test)
test_r2 = round(metrics.r2_score(y_test, y_pred), decimal_places)

# -------------------------- Interpretation --------------------------
vars = X_col_names.copy()

# 1) Permutation importance
save_path = "Results/Interpretation/Permutation/"
result = permutation_importance(model, X_test, y_test, n_repeats=permutations,
                                random_state=seed, n_jobs=2, scoring=scoring)
perm_importances = result.importances_mean
dict = {'Feature': vars, "Importance": perm_importances}
perm_imp_df = pd.DataFrame(dict)
# just get most important x
perm_imp_df.sort_values(by="Importance", ascending=False, inplace=True, axis=0)
# flip so most important at top on graph
perm_imp_df.sort_values(by="Importance", ascending=True, inplace=True, axis=0)
# plot
plot_permutation(perm_imp_df=perm_imp_df,
                 save_path=save_path,
                 save_name="{}_permutation".format(pred_model))
# todo: add variance in perms to plot

# 2) Tree-based impurity importance
if pred_model == 'rf':
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
if pred_model == 'rf':
    explainer = shap.TreeExplainer(model, feature_pertubation="tree_path_dependent")
    shap_dict = explainer(X_test)
    shap_values = explainer.shap_values(X_test)
    shap_values_df = pd.DataFrame(shap_values, columns=vars)

if pred_model == 'enet':
    explainer = shap.LinearExplainer(model, X_test, feature_pertubation="correlation_dependent")
    shap_dict = explainer(X_test)
    shap_values = explainer.shap_values(X_test)
    shap_values_df = pd.DataFrame(shap_values, columns=vars)

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

# PDP


# ICE

print('done!')