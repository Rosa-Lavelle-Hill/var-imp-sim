import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay

def plot_impurity(impurity_imp_df, save_path, save_name, figsize=(8, 3.5)):
    y_ticks = np.arange(0, impurity_imp_df.shape[0])
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(y_ticks, impurity_imp_df["Importance"], color="dodgerblue")
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(impurity_imp_df["Feature"])
    ax.set_title("Impurity Reduction Importance (training set)")
    ax.set_xlabel("Impurity Reduction")
    fig.tight_layout()
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()
    return



def plot_permutation(perm_imp_df, save_path, save_name, figsize=(8, 3.5)):
    y_ticks = np.arange(0, perm_imp_df.shape[0])
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(y_ticks, perm_imp_df["Importance"], color="dodgerblue")
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(perm_imp_df["Feature"])
    ax.set_title("Permutation Importance (test set)")
    ax.set_xlabel("Permutation Importance")
    fig.tight_layout()
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()
    return



def plot_SHAP(shap_dict, col_list, plot_type, n_features,
              save_path, save_name, title="", figsize=(8, 3.5)):
    plt.figure(figsize=figsize)
    shap.summary_plot(shap_dict, feature_names=col_list, show=False,
                      plot_type=plot_type, max_display=n_features)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path + save_name)
    plt.clf()
    plt.cla()
    plt.close()
    return



def check_corr(X_and_y, save_path, save_name, X_feature_names, decimal_places):
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



def plot_PDP(save_path, pred_model, model, X_test, features, figsize=(8, 3.5)):
    fig, ax = plt.subplots(figsize=figsize)
    g = PartialDependenceDisplay.from_estimator(model, X_test, features)
    g.plot()
    plt.tight_layout()
    plt.savefig(save_path + '{}_pdp.png'.format(pred_model), bbox_inches='tight')
    return



def plot_ICE(save_path, pred_model, model, X_test, features, figsize=(8, 3.5)):
    fig, ax = plt.subplots(figsize=figsize)
    g = PartialDependenceDisplay.from_estimator(model, X_test, [features[0]], kind='both')
    g.plot()
    plt.tight_layout()
    plt.savefig(save_path + '{}_ice.png'.format(pred_model), bbox_inches='tight')
    return



def plot_SHAP_force(i, X_test, model, save_path, save_name,
                    pred_model, title, figsize=(8, 4)):
    if (pred_model == "rf") or (pred_model == "tree"):
        explainerModel = shap.TreeExplainer(model)
    elif (pred_model == "enet") or (pred_model == "lasso"):
        masker = shap.maskers.Independent(data=X_test)
        explainerModel = shap.LinearExplainer(model, masker=masker)
    # todo: needs a mask
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
