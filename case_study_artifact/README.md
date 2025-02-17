# Artifact Composition

This folder contain artifact that created during our case study experiment from `Q5_case_study.ipynb` that fascillitate that experiment.

- **dataset** folders contain dataset CSV files. Based on their functionality, they are organised into two folders:
    - **original**: Most of our datasets are hosted on UCI repository and can be grabbed online with `fetch_ucirepo` method. However, some of them are hosted elsewhere and needs to be loaded locally. This folder contain all datasets that require to be loaded from disk.
    - **metafeatures**: They contain metafeature values calculated from original datasets using `pymfe.mfe` modules. As the computational and memory requirements of calculating datasets' metafeatures grow exponentially with dataset's size, we tends to run seperate sessions from performing actual case study experiment.

- **Models** contains binary pickle files storing pre-trained predictors for our case study experiments. They may also be used for other research project as they have been evaluated to be the best meta-learning models from our previous experiments
    - **indie_models_3bs.pkl** store a collections of single-target predictors. For each of 6 dataset-level bias metrics, one single-target predictors is trained to predict said bias metrics.
    - **indie_chain_models_{..}.pkl** store a Regressor Chain models that predictor all dataset-level bias metrics except Group Fairness. Since its output is without lables, it also store the order of its target bias metrics for labeling after its prediction. Ensemble models are trained after loading these chain models.
        - There are 3 versions of the chain models. No appendix means there are no hyperparameter tuning for the chain models. Having appendix of *paramstuned* means the chain models have their hyperparameter tuned but not the base predictors inside them. Having appendix of *paramstuned_indiv* means the hyperparameters of these base predictors are tuned as well.