import argparse
import math
from pathlib import Path
import pandas as pd
import numpy as np
import time

from artifact.util import set_seed
from artifact.pipeline_components import pipeline_generate_sensitive_groups,pipeline_calculate_biasScore, pipeline_generate_dataset,pipeline_calculate_metaFeature

start_time = time.time()

def aggregate_config(list_of_dict):
    config={}
    for d in list_of_dict:
        config.update(d)
    return config

def sensitive_evaluation(config,df_data):
    sen_groups,group_scores=pipeline_generate_sensitive_groups(config,df_data)
    config.update({"sensitive_group_scores":group_scores})
    bs_dict=pipeline_calculate_biasScore(config,df_data,sen_groups)
    return bs_dict

def generate_dataset_cadidate(seed):
    set_seed((seed+1)*(seed+1)*7)
    n_samples=np.random.randint(100,201)*100
    n_classes=np.random.randint(2,7)
    n_features=np.random.randint(10,51)
    feature_distribution=np.random.rand(4)
    n_repeated_max=int(n_features/5)
    n_informative, n_redundant, n_repeated, _= [int(n/sum(feature_distribution)*n_features) for n in feature_distribution]
    n_informative=max(int(n_features/10),n_informative,math.ceil(np.log2(n_classes)))
    n_repeated=min(n_repeated_max,n_repeated)
    if n_informative+n_redundant+n_repeated>n_features:
        n_repeated-=n_informative+n_redundant+n_repeated-n_features
    n_clusters_per_class=max(min(np.random.randint(1,4),int((2**n_informative)/n_classes)),1)
    positive_class_num=np.random.randint(1,int(n_classes/2)+1)
    return {
        "dataset_random_state":(seed+1)*70,
        "n_samples":n_samples,
        "n_features":n_features,
        "n_informative":n_informative,
        "n_redundant":n_redundant,
        "n_repeated":n_repeated,
        "n_classes":n_classes,
        "class_imbalanced":np.random.rand()>0.5, 
        "n_clusters_per_class":n_clusters_per_class,   
        "positive_class_num":positive_class_num,
    }

def generate_smaller_cadidate(seed):
    set_seed((seed+1)*(seed+1)*7)
    n_samples=np.random.randint(20,101)*100
    n_features=np.random.randint(2,11)
    n_classes=min(np.random.randint(2,7),2**n_features)
    n_informative=np.random.randint(1,n_features)
    n_informative=min(max(n_informative,math.ceil(np.log2(n_classes))),n_features)
    n_redundant=n_features-n_informative
    n_clusters_per_class=max(min(np.random.randint(1,4),int((2**n_informative)/n_classes)),1)
    positive_class_num=np.random.randint(1,int(n_classes/2)+1)
    return {
        "dataset_random_state":(seed+1)*70,
        "n_samples":n_samples,
        "n_features":n_features,
        "n_informative":n_informative,
        "n_redundant":n_redundant,
        "n_repeated":0,
        "n_classes":n_classes,
        "class_imbalanced":np.random.rand()>0.5, 
        "n_clusters_per_class":n_clusters_per_class,   
        "positive_class_num":positive_class_num,
    }

def generate_sensitive_candidates(seed):
    set_seed((seed+1)*(seed+1)*7)
    seeds=np.random.randint(seed*1000,(seed+1)*1000,size=4)
    balanced_num=np.random.randint(3)
    sensitive_imbalanced=[False]*balanced_num+[True]*(4-balanced_num)
    np.random.shuffle(sensitive_imbalanced)
    continous_groups=[True if np.random.rand()<0.5 else False for _ in range(4)]
    n_groups=[np.random.randint(2,6) for _ in range(4)]
    group_dimensions=[np.random.randint(1,4) for _ in range(4)]
    return [{
        "sensitive_random_state":seeds[i],
        "continous_groups":continous_groups[i],  
        "sensitive_imbalanced":sensitive_imbalanced[i], 
        "n_group":n_groups[i],
        "group_dimension":group_dimensions[i]
    } for i in range(4)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-num", type=int, required=True, help="Numbers of rows to be generated."
    )
    parser.add_argument(
        "-small",type=bool,required=False,help="Generate smaller datasets if true"
    )
    parser.add_argument(
        "-output", type=Path, required=True, help="Output csv File"
    )
    args = parser.parse_args()
    start_time = time.time()

    df=pd.DataFrame()
    previous_row=0
    hyperparameter={
        "k":10,  
        "prev_weight":1.5, 
        "random_walk":0.2,  
        "coefficient":1.2,
        "landmarking_random_state":100
    }
    if Path(args.output).is_file():
        df=pd.read_csv(args.output,header=0)
        previous_row=int(df.shape[0]/4)
    for i in range(args.num):
        seed=previous_row+i
        if args.small:
            dataset_config=generate_smaller_cadidate(seed)
        else:
            dataset_config=generate_dataset_cadidate(seed)
        print(dataset_config)
        df_data=pipeline_generate_dataset(dataset_config)
        mf_dict=pipeline_calculate_metaFeature(hyperparameter,df_data)
        print(f"{i+1}/{args.num}'s metafeature is evaluated. Time taken: {(time.time() - start_time)}")
        start_time=time.time()
        sensitive_configs=generate_sensitive_candidates(seed)
        for idx,sc in enumerate(sensitive_configs):
            config=aggregate_config([dataset_config,sc,hyperparameter])
            print(sc)
            bs_dict=sensitive_evaluation(config,df_data)
            row_data=aggregate_config([config,mf_dict,bs_dict])
            row=pd.DataFrame([row_data])
            df=pd.concat([df,row],axis=0)
            df.to_csv(args.output,index=False)
            print(f"{idx+1}/4 sensitive group variants out of {i+1}/{args.num} datasets has been generated. Time taken: {(time.time() - start_time)}")
            start_time=time.time()

if __name__ == "__main__":
    main()
