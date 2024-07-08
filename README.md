# var-imp-sim
To accompany the article **"An Explainable AI Handbook for Psychologists: Methods, Opportunities, and Challenges"**

Rosa Lavelle-Hill (University of Copenhagen), Gavin Smith (University of Nottingham), and Kou Murayama (University of Tübingen)

Abstract

With more researchers in psychology using machine learning to model large datasets, many are also looking to eXplainable AI (XAI) to understand how their model works and to gain insights into the most important predictors. However, the methodological approach for explaining a machine learning model is not as straightforward or as well-established as with inferential statistical models. Not only are there a large number of potential XAI methods, but there are also a number of unresolved challenges when using XAI for psychological research. This article aims to provide an introduction the the field of XAI for psychologists. We first introduce explainability from a machine learning perspective and consider what makes a good explanation for different settings. An overview of commonly used XAI approaches and their use cases is then provided. We categorize methods along two dimensions: model-specific vs. model-agnostic and producing local vs. global explanations. We then highlight and discuss some of the practical challenges that psychologists can encounter when using XAI metrics to understand predictor importance, namely, how to attribute importance when there are dependencies between features, when there are complex (non-linear) interactions, and/or multiple possible solutions to the prediction problem. This tutorial is accompanied by a Python code repository where readers can interact with the code to understand better how changing characteristics of the data (e.g., multicollinearity) or key parameters in the modeling process (e.g., selecting different machine learning algorithms or hyper-parameters) can influence different XAI outputs. 

The main script is **sim.py**, where researchers can change key parameters (e.g., the prediction model, size of data, multicollinearity, effect size) and run the code to see how the XAI outputs change.

![MainFig8thJul](https://github.com/Rosa-Lavelle-Hill/var-imp-sim/assets/51444424/1887e370-f876-472b-90cc-e280e75dd731)

**sim-replicate-figure1.py** was used to create Figure 1.

**sim-figure-C.py** was used to create Figure 2.

**sim-figure-interaction_U.py** was used to create Figure 3.

**sim-confounder.py** and **sim-confounder-plotR** were used to create Figure 4.
