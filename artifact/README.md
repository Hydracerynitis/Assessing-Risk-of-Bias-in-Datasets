# Artifact Composition

This folder contain artifact that created during our research and are used to fascillitate our experiments.

- **bias_score.py** contains codes that are responsible for calculating the dataset-level bias metrics for synthesised datasets.
    - `square_average`, `average`, `z_average`. Methods that calculate weighted average of bias metrics of each sensitive subspaces of a datasets into an aggregaed values for dataset.
        - `square_average` use the metrics themsevles as the weight. `average` all use the weight of - `z_average` use Z-score calculated with bias metrics as weight. 
    - `gsd` and `prev` calculate Group Size Disparity and Prevalence Disparity of the dataset using information from the datasets themselves.
    - `equal_opportunity`, `disparate_parity` and `generalized_entropy_index` calculate Equal Opportunity, Disparate Parity, and Individual Fairness / Group Fairness respectively. They require an external landmarking models' prediction on the dataset to calculate
    - `generate_lankmarking_score` train random-node decision tree and linear regression landmarking models  to calculate all landmarking bias metrics
        - The random-node decision tree has a `max_depth=5` parameter due to we are applying to datasets with more than 2 labels.

- **sensitive_ssr** specifies the sensitive SSR framework, which is under the hierachy of sensitive Ssubspace, sensitive group and sensitive rule. They descirbe a portion of datasets entries that can be considered sensitive at different levels
    - `SensitiveRule` specify a filtering rule regardig a certain feature that whether or not entries can be considered sensitive.
    - `SensitiveShape` is a collection of sensitive Rules to construct a continous shape in the data space for a sensitive group.
    - `SensitiveSubspace` is a collection of sensitive Shape to provide a disconnected description of the sensitive group it represent.
    - `InsensitiveSubsapce` is a special sensitive Subsapce that contain all entries that have yet to be included in any Sensitive Subspace, which represent individuals that are not affected by any discrimination
    - `CatagoricalSubspace` is the adaptiation of traditional sensitive group framework into sensitive SSR framework. It use predicate method for filtering entries instead of SensitiveRule to allow using categorical data.

- **sensitive_subspace_generation** contains codes that are responsible for synthesising several sensitive subspaces given a continous dataset. The generation process also allows for specifiying the resultant Group Size Dispairty and Prevalence Disparity of the generation. We have developed two algorithms handling the synthesizing of sensitive subspaces: Patch Subspace generation and Continous Subspace generation.
    - `group_score`. Helper method used to calculate the Group Size Disparity and Prevalence Disparity of generated sensitive subspaces on a given dataset.
    - `calculate_goal` calculate the target Group Size Disparity and Prevalence Disparity based on the given weight matrix. The weight matrix is a 2xN matrix that specify the size of Group Size and Prevalence of each to-be-generated sensitive subspace in relation to others.
    - Patch Subspace generation synthesize sensitive subspaces by spliting the data space into many smaller equal-size patches of sensitive groups, and then search for the combinaton of patches that satisfies the target Group Size Disparity and Prevalence Disparity.
        - `generate_candidate` is responsible for generating sensitive patches, where `move_pointer` is used to help iterate generated groups. 
        - `search_group` is responsible to group sensitive patches together based on target Group Size Disaprity and Prevalence Disparity using Greedy search strategy.
    - Continous Subspace generation synthesize a continous sensitive subspace by searching through the sensitive shape that best approximate the target Group Size Disparity and Prevalence Disparity. 
        - `generate_rule` generates intial sensitive rules that to be considered for possible candidate.
        - `join_candidate` is responsible for examining snesitive rules and join adjacent rules together to avoid repetitions.
        - `exist_similar` is repsonsible for checking two sensitive shapes that have differently 
        - `get_candidate_shape` is responsible for combining previous candidate sensitive shape with newly generated sensitive rules.
        - `continous_serach` is responsible for iterating possible candidates and searching for best apporximate shape. For each features, it will keep best 5 candidats and combine them with rules generated with next feature. It will stop when all specified features have been iterated.

- **pipeline_components.py** facilitate our pipeline synthesise new entries in our meta-dataset by generating new datasets and their sensitive subspaces to record their bias metrics and meta-features. The output of each pipeline run is predetermined by the input config settings consist of various parameters values.
    - The pipeline has the followings steps which is handled by individual methods with the same name:
        - `pipeline_generate_dataset`: This step invokes `scikit-learn.make_classification` to synthesise a new dataset using parameters from the config settings. Config setting also has parameter *class_imbalanced* to control whether the synthesised dataset have a balanced distribution of label values or not. If *True*, `random_class_distribution` helper method is used to generate the distribution of label values.
        - `pipeline_setup_searching_algorithm`: This step will output information necessary for sensitive subspaces generation algorithm to run. More specifically, the feature dimension for generated sensitive subspaces to reside on and target Group Size Disparity and Prevalence Disparity. Config setting has parameter *sensitive_imbalanced* to control whether the generated sensitive subspaces have balanced distribution in relation to each other. If *True*, `random_class_distribution` helper method is used  to generate randomised weight matrix which is used to calculate target Group Size Disparity and Prevalence Disparity.
        - `pipeline_generate_sensitive_groups`: This step invoke sensitie subspace generation algorithm based on config settings *continous_groups*. If *True*, invoke Continous Subspace generation with `pipeline_continous_group_generation` method. Otherwise, invoke Patch Subspace generation with `pipeline_patch_group_generation` method.
        - `pipeline_calculate_biasScore`: This step calculate dataset-level bias metrics of generated sensitive subspaces from previous steps.
        - `pipeline_calculate_metaFeature`: This step calculate the metafeatures of synthesised dataset using `pymfe.mfe` library 
    - For each pipeline run, the output will be recorded to **metadataset.csv**. It will contain the config settings of the run, the metafeatures of the generated dataset and bias metrics of their generated synthesised sensitive subspaces.
    - We use **synthesize_dataset.py** as the master file to run the pipeline. It has three command line parameters: *num* indicate numbers of entries to be outputed by the pipeline in this execution of the file. *small* determines wether or not datasets generated by pipeline have less data size and data complexity or not.  *output* specifies the output file path that will store the meta-dataset.
        - Config settings for generating smaller datasets is produced by its helper method `generate_smaller_cadidate` and config settings for generating larger datasets is produced by `generate_dataset_cadidate` instead.
        - The master file will make pipeline generate four sets of sensitive subspaces  for each unique generated dataset. 

- **util.py** contains michelleneous codes that are not components of our Sensisitve SSR (Subspace, Shape, Rule) framework or dataset-level bias metrics. However, these codes are helpful for streamlining our development and experiment procedures:
    - `reload_module`: Helper method that allow us the load newer version of our python artifact into experiment notebooks.
    - `set_seed`: Initialise random seeds for numpy randomness
    - `df_data`: Representation that records information of synthesised datasets and their sensitive subspaces.
        - `X`: Features of the dataset.
        - `Y`: Label of the dataset.
        - `desired`: The list of all positive labels.
        - `encoder` (optional): Any relevant encoders that encode original positive labels into numerical values for model trainings. It is used when the original dataset contain categorical labels