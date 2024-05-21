import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay

# font_size = 16
# label_size = 20
font_size = 12
label_size = 14
figsize = (2.7, 1.5)

def plot_impurity(impurity_imp_df, save_path, save_name, figsize=(8, 3.5)):
    """
    Plots impurity importance from dataframe
    :param impurity_imp_df: dataframe of importance values with a "Feature" and "Importance" column
    :param save_path: path where plot should be saved
    :param save_name: name of plot to be saved
    :param figsize: size of figure as a tuple
    """
    y_ticks = np.arange(0, impurity_imp_df.shape[0])
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(y_ticks, impurity_imp_df["Importance"], color="dodgerblue")
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(impurity_imp_df["Feature"])
    ax.set_title("Impurity Reduction Importance (training set)")
    ax.tick_params(axis='y', labelsize=label_size)
    ax.set_xlabel('Importance', fontsize=font_size)
    fig.tight_layout()
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()
    return



def plot_permutation(perm_imp_df, save_path, save_name, figsize=(8, 3.5)):
    """
    Plots permutation importance from dataframe
    :param perm_imp_df: dataframe of importance values with a "Feature" and "Importance" column
    :param save_path: path where plot should be saved
    :param save_name: name of plot to be saved
    :param figsize: size of figure as a tuple
    """
    y_ticks = np.arange(0, perm_imp_df.shape[0])
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(y_ticks, perm_imp_df["Importance"], color="dodgerblue")
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(perm_imp_df["Feature"])
    ax.set_title("Permutation Importance (test set)")
    ax.set_xlabel('Importance', fontsize=font_size)
    ax.tick_params(axis='y', labelsize=label_size)
    fig.tight_layout()
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()
    return


def plot_SHAP_ordered(shap_values, save_path, save_name, variable_order=['X3', 'X2', 'X1'], figsize=figsize):
    """
    Plots ordered SHAP importance from dataframe
    :param shap_values: dataframe of importance values with a "Feature" and "Importance" column
    :param save_path: path where plot should be saved
    :param save_name: name of plot to be saved
    :param figsize: size of figure as a tuple
    """
    shap_values["Feature"] = pd.Categorical(shap_values["Feature"], categories=variable_order, ordered=True)
    # Sort the DataFrame based on the categorical order
    shap_values = shap_values.sort_values("Feature")
    fig, ax = plt.subplots(figsize=figsize)
    plt.barh(shap_values["Feature"], shap_values["Importance"], color="dodgerblue", )
    ax.set_yticklabels(shap_values["Feature"])
    ax.set_title(" ")
    ax.set_xlim(0, 2.5)
    ax.set_xlabel('', fontsize=font_size)
    ax.tick_params(axis='y', labelsize=label_size)
    fig.tight_layout()
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()
    return


def plot_multiple_permutations(result, save_name, save_path, vars, figsize= figsize,
                               order=False, title= 'Permutation Importances (test set)',
                               variable_order=['X3', 'X2', 'X1']):
    """
    :param result: output from sklearn's permutation importance calculation
    :param save_path: path where plot should be saved
    :param save_name: name of plot to be saved
    :param figsize: size of figure as a tuple
    :param vars: list of variable names
    :param order: if True, variables will be ordered as stated in list variable_order
    :return:
    """
    sorted_indices = result.importances_mean.argsort()
    fig, ax = plt.subplots(figsize=figsize)
    if order == True:
        # Filter out variables present in data and sort them according to the order
        sorted_indices = [vars.index(var) for var in variable_order if var in vars]
        ax.set_xlim(0, 1.5)

    plt.barh(range(len(sorted_indices)), result.importances_mean[sorted_indices],
             xerr=result.importances_std[sorted_indices], color="dodgerblue")
    plt.yticks(range(len(sorted_indices)), np.array(vars)[sorted_indices])
    ax.set_xlabel('Importance', fontsize=font_size)
    ax.tick_params(axis='y', labelsize=label_size)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path + save_name)
    plt.clf()
    plt.cla()
    plt.close()
    return



def plot_SHAP(shap_dict, col_list, plot_type, n_features,
              save_path, save_name, title="", figsize= figsize,
                     order=False, variable_order=['X3', 'X2', 'X1', 'const']):
    """
    Plots SHAP values from a dictionary of values
    :param shap_dict: dictionary output from the SHAP function
    :param col_list: list of column/feature names
    :param plot_type: the type of SHAP plot (e.g., summary, bar, violin)
    :param n_features: the number of features to display (plots most important first)
    :param save_path: path where plot should be saved
    :param save_name: name of plot to be saved
    :param figsize: size of figure as a tuple
    :param title: title of plot
    """
    plt.figure(figsize=figsize)
    if order == True:
        shap.plots.bar(shap_dict, show=False, max_display=n_features, order=variable_order)
        ax = plt.gca()
        ax.set_xlim(0, 2.5)
    else:
        shap.summary_plot(shap_dict, feature_names=col_list, show=False,
                          plot_type=plot_type, max_display=n_features)
    plt.title(title)
    plt.tick_params(axis='y', labelsize=label_size)
    plt.tight_layout()
    plt.savefig(save_path + save_name)
    plt.clf()
    plt.cla()
    plt.close()
    return



def check_corr(X_and_y, save_path, save_name, X_feature_names, decimal_places):
    """
    Plots a heatmap of Pearson r correlation coefficients
    :param X_and_y: dataframe containing X and y features
    :param save_name: name of plot to be saved
    :param X_feature_names: list of feature names
    :param decimal_places: number (int) of decimal places to round to
    """
    cor = round(X_and_y.corr(method='pearson'), decimal_places)
    cor_cols = X_feature_names + ["y"]
    cor.columns = cor_cols
    cor.index = cor_cols
    cor.to_csv(save_path + "data_correlations.csv")
    ax = sns.heatmap(cor, linewidth=0.1, cmap="Oranges")
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()
    return



def plot_PDP(pred_model, model, X_test, features, save_path, save_name = "pdp", figsize=(8, 3.5)):
    """
    Plots two a partial dependence plots with either one or two variables depending on the list of features
    :param pred_model: string containing name of prediction model
    :param model: the specified model
    :param X_test: dataframe of data to be explained
    :param features: a list containing either single features or a tuple of two features
    :param figsize: size of figure as a tuple
    :param save_path: file path where plot should be saved
    :return:
    """
    fig, ax = plt.subplots(figsize=figsize)
    g = PartialDependenceDisplay.from_estimator(model, X_test, features)
    g.plot()
    if len(features) == 1:
        ax.set_ylabel('PD', fontsize=label_size)
        ax.set_xlabel(features[0], fontsize=label_size)
    plt.tight_layout()
    plt.savefig(save_path + f'{pred_model}_{save_name}.png', bbox_inches='tight')
    return



def plot_ICE(pred_model, model, X_test, feature, save_path, figsize=(8, 3.5),
             save_name=None):
    """
    :param pred_model: string containing name of prediction model
    :param model: the specified model
    :param X_test: dataframe of data to be explained
    :param feature: a string containing a single feature name
    :param save_path: file path where plot should be saved
    :return:
    """
    if save_name is None:
        save_name = '{}_ice.png'.format(pred_model)

    fig, ax = plt.subplots(figsize=figsize)
    g = PartialDependenceDisplay.from_estimator(model, X_test, [feature], kind='both')
    g.plot()
    plt.tight_layout()
    plt.savefig(save_path + save_name, bbox_inches='tight')
    return



def plot_SHAP_force(i, X_test, model, save_path, save_name,
                    pred_model, title, figsize=(8, 4)):
    """

    :param i: index (integer) of the data instance to plot explanation for
    :param X_test: dataframe of data to be explained
    :param model: the specified model
    :param save_path: string file path where plot should be saved
    :param pred_model: string containing name of prediction model
    :param title: string containing the title of the plot
    :param figsize: size of figure as a tuple
    :return:
    """
    if (pred_model == "rf") or (pred_model == "tree"):
        explainerModel = shap.TreeExplainer(model)
    elif (pred_model == "enet") or (pred_model == "lasso"):
        masker = shap.maskers.Independent(data=X_test)
        explainerModel = shap.LinearExplainer(model, masker=masker)
    # todo: needs a mask!!
    else:
        print("please enter one of the regression or tree based models: 'rf', 'tree', 'lasso, or 'enet'")
        breakpoint()
    shap_values_Model = explainerModel.shap_values(X_test).round(2)
    plt.figure(figsize=figsize)
    p = shap.force_plot(explainerModel.expected_value.round(2), shap_values_Model[i],
                        round(X_test.iloc[[i]], 2), matplotlib=True, show=False)
    plt.gcf().set_size_inches(figsize)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path + save_name + '.png')
    plt.close()
    plt.clf()
    plt.cla()
    plt.close()
    return(p)



def print_tree(max_depth, X_test, y_test, feature_names, save_name, save_path,
               fontsize=8, figsize=(10, 4)):
    """
    :param max_depth: integer to control the number of tree levels
    :param X_test: dataframe of predictor variables
    :param y_test: dataframe or array of outcome variables
    :param feature_names: list of strings
    :param save_name: string
    :param fontsize: integer
    :param figsize: tuple to control size
    :param save_path: string file path where plot should be saved
    :return:
    """
    # define a decision tree for visual puposes only
    from sklearn import tree
    dec_tree = tree.DecisionTreeRegressor(max_depth=max_depth)
    # fit tree
    dec_tree.fit(X_test, y_test)
    fig = plt.figure(figsize=figsize)
    _ = tree.plot_tree(dec_tree,
                       feature_names=feature_names,
                       filled=True,
                       fontsize=fontsize,
                       precision=1,
                       rounded=True,
                       class_names=True)
    plt.tight_layout()
    plt.savefig(save_path + save_name + ".png")
