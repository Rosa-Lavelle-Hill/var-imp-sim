# ====================================================================================================
# plot results from R script for random forest:
# ====================================================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

save_path = "Results confounder/R_plots/"
font_size = 12
label_size = 14
figsize = (2.7, 1.5)

# 1. MF = 1.0
# a. Unconditional permutation importance
perm_imp_df = pd.DataFrame({'Feature': ['X3', 'X2', 'X1'], 'Importance': [0.04449965, 0.70969087, 0.33943808]})

y_ticks = np.arange(0, perm_imp_df.shape[0])
fig, ax = plt.subplots(figsize=(figsize))
ax.barh(y_ticks, perm_imp_df["Importance"], color="dodgerblue")
ax.set_yticks(y_ticks)
ax.set_yticklabels(perm_imp_df["Feature"])
# ax.set_title("Permutation Importance (OOB)")
ax.set_title("")
ax.set_xlabel('Importance', fontsize=font_size)
ax.tick_params(axis='y', labelsize=label_size)
plt.xticks([0.0, 0.5, 1.0, 1.5])
fig.tight_layout()
plt.savefig(save_path + "unconditional_perm_imp_mf1.0.png")
plt.clf()
plt.cla()
plt.close()

#  b. Conditional permutation importance
perm_imp_df = pd.DataFrame({'Feature': ['X3', 'X2', 'X1'], 'Importance': [0.00009671041, 0.02668361, 0.009719316]})

y_ticks = np.arange(0, perm_imp_df.shape[0])
fig, ax = plt.subplots(figsize=(figsize))
ax.barh(y_ticks, perm_imp_df["Importance"], color="dodgerblue")
ax.set_yticks(y_ticks)
ax.set_yticklabels(perm_imp_df["Feature"])
# ax.set_title("Permutation Importance (OOB)")
ax.set_title("")
ax.set_xlabel('Importance', fontsize=font_size)
ax.tick_params(axis='y', labelsize=label_size)
plt.xticks([0.0, 0.5, 1.0, 1.5])
fig.tight_layout()
plt.savefig(save_path + "conditional_perm_imp_mf1.0.png")
plt.clf()
plt.cla()
plt.close()

# ----------------------------------------------------------------------------------------------------------------------
# 2. MF = 0.3
# a. Unconditional permutation importance

perm_imp_df = pd.DataFrame({'Feature': ['X3', 'X2', 'X1'], 'Importance': [0.1874128, 0.5550347, 0.2475288]})

y_ticks = np.arange(0, perm_imp_df.shape[0])
fig, ax = plt.subplots(figsize=(figsize))
ax.barh(y_ticks, perm_imp_df["Importance"], color="dodgerblue")
ax.set_yticks(y_ticks)
ax.set_yticklabels(perm_imp_df["Feature"])
ax.set_title("")
ax.set_xlabel('Importance', fontsize=font_size)
ax.tick_params(axis='y', labelsize=label_size)
plt.xticks([0.0, 0.5, 1.0, 1.5])
fig.tight_layout()
plt.savefig(save_path + "unconditional_perm_imp_mf0.3.png")
plt.clf()
plt.cla()
plt.close()

#  b. Conditional permutation importance

perm_imp_df = pd.DataFrame({'Feature': ['X3', 'X2', 'X1'], 'Importance': [0.00007744431, 0.02141398, 0.00742788]})

y_ticks = np.arange(0, perm_imp_df.shape[0])
fig, ax = plt.subplots(figsize=(figsize))
ax.barh(y_ticks, perm_imp_df["Importance"], color="dodgerblue")
ax.set_yticks(y_ticks)
ax.set_yticklabels(perm_imp_df["Feature"])
# ax.set_title("Permutation Importance (OOB)")
ax.set_title("")
ax.set_xlabel('Importance', fontsize=font_size)
ax.tick_params(axis='y', labelsize=label_size)
plt.xticks([0.0, 0.5, 1.0, 1.5])
fig.tight_layout()
plt.savefig(save_path + "conditional_perm_imp_mf0.3.png")
plt.clf()
plt.cla()
plt.close()

print('done')