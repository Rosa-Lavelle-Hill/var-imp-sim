import numpy as np
import matplotlib.pyplot as plt
import shap

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