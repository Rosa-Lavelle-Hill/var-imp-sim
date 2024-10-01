# var-imp-sim
To accompany the article **"An Explainable AI Handbook for Psychologists: Methods, Opportunities, and Challenges"**

Rosa Lavelle-Hill (University of Copenhagen), Gavin Smith (University of Nottingham), Hannah Deininger (University of Tübingen), and Kou Murayama (University of Tübingen)

Abstract

With more researchers in psychology using machine learning to model large datasets, many are also looking to eXplainable AI (XAI) to understand how their model works and to gain insights into the most important predictors. However, the methodological approach for explaining a machine learning model is not as straightforward or as well-established as with traditional statistical models. Not only are there a large number of potential XAI methods to choose from, but there are also a number of unresolved challenges when using XAI to understand psychological data. This article aims to provide an introduction to the field of XAI for psychologists. We first introduce explainability from a machine learning perspective. Then we provide an overview of commonly used XAI approaches, namely permutation importance, impurity-based feature importance, Individual Conditional Expectation (ICE) graphs, Partial Dependence Plots (PDP), Accumulated Local Effect (ALE), Local Interpretable Model-agnostic Explanations (LIME), SHapley Additive exPlanations (SHAP), and Deep Learning Important FeaTures (DeepLIFT). Finally, we highlight and discuss some of the practical challenges that psychologists can encounter when using XAI metrics to understand predictor importance. 

![Figure 1. A Categorization of XAI Methods](https://github.com/Rosa-Lavelle-Hill/var-imp-sim/blob/master/Fig1_1stOct.png?raw=true)

The GitHub repository can be downloaded and the code interacted with, in that key parameters (i.e., the mean and SD of the data, the extent of multicollinearity (Pearson r), number of samples (N), number of features, the effect size (R^2), the model class (i.e., tree-based models compared to regression-based models), and the random seed) can be changed, the code re-run, and the effects these parameters have on the XAI outputs can be examined.

The main script is **sim.py**, where researchers can change key parameters mentioned above (e.g., the prediction model, size of data, multicollinearity, effect size) and run the code to see how the XAI outputs change.

**sim-replicate-figure1.py** was used to create Figure 1.

**sim-figure-C.py** was used to create Figure 2.

**sim-figure-interaction_U.py** was used to create Figure 3.

**sim-confounder.py** and **sim-confounder-plotR.py** were used to create Figure 4.
