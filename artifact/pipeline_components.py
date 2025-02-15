from sklearn.datasets import make_classification
from pymfe.mfe import MFE
import numpy as np
import pandas as pd

from artifact.util import set_seed,df_data
from artifact.bias_score import gsd,prevd,generate_lankmarking_score
from artifact.sensitive_subspace_generation import calculate_goal,continous_serach,generate_candidate, search_group,group_score

def random_class_distribution(n_class,random_state=None):
    set_seed(random_state)
    weight=np.random.rand(n_class)
    weight/=sum(weight)
    return weight

def random_sensitive_weight(n_group,random_state=None):
    set_seed(random_state)
    weight=np.random.rand(2,n_group)
    return weight

def pipeline_generate_dataset(config):
    if config["class_imbalanced"]==True:
        class_weight=random_class_distribution(config["n_classes"],config["dataset_random_state"])
    else:
        class_weight=np.full(config["n_classes"],1/config["n_classes"])
    config.update({"class_weight":class_weight})
    X,Y=make_classification(n_samples=config["n_samples"],n_features=config["n_features"],n_informative=config["n_informative"],
                        n_redundant=config["n_redundant"],n_repeated=config["n_repeated"],n_classes=config["n_classes"],
                        n_clusters_per_class=config["n_clusters_per_class"],weights=class_weight,random_state=config["dataset_random_state"])
    X,Y=pd.DataFrame(X),pd.Series(Y)
    unique_class=Y.unique()
    set_seed(config["dataset_random_state"])
    desired=np.random.choice(unique_class,min(config["positive_class_num"],config["n_classes"]))
    dfd=df_data(X,Y,desired)
    return dfd

def pipeline_setup_searching_algorithm(config,df_data):
    if config["sensitive_imbalanced"]==True:
        sensitive_wieght_matrix=random_sensitive_weight(config["n_group"],config["sensitive_random_state"])
    else:
        sensitive_wieght_matrix=np.full((2,config["n_group"]),1)
    set_seed(config["sensitive_random_state"])
    sen_features=np.random.choice(df_data.X.columns,min(config["group_dimension"],config["n_features"]),replace=False)
    goals=calculate_goal(sensitive_wieght_matrix,df_data)
    print(goals)
    return sen_features,goals

def pipeline_continous_group_generation(config,df_data,sen_features,goals):
    sen_groups=[]
    group_scores=[]
    for g in goals:
        remaining_features=list(sen_features)
        current_shape=None
        while len(remaining_features)>0:
            set_seed(config["sensitive_random_state"]*(len(sen_groups)+1))
            feature=remaining_features.pop(np.random.randint(len(remaining_features)))
            current_shape=continous_serach(df_data,config["k"],feature,g,current_shape,config["coefficient"])
            current_shape=sorted(current_shape,key=lambda s:abs(g[0]-s[1][0])+abs(g[1]-s[1][1])*config["prev_weight"])[:config["k"]]
        if len(current_shape)<=0:
            continue
        result=current_shape[np.random.randint(len(current_shape))]
        sen_groups.append(result[0])
        group_scores.append(result[1])
    return sen_groups, group_scores

def pipeline_patch_group_generation(config,df_data,sen_features,goals):
    candidates=generate_candidate(df_data.X,config["k"],sen_features)
    candidates=[c for c in candidates if c.select(df_data.X).sum()>0]
    sensitive_groups=[]
    group_scores=[]
    for g in goals:
        if len(candidates)<=0:
            break
        sen_gp=search_group(candidates,df_data,g,config["prev_weight"],config["random_walk"])
        sensitive_groups.append(sen_gp)
        group_scores.append(group_score(sen_gp,df_data))
    return sensitive_groups, group_scores
   
def pipeline_generate_sensitive_groups(config,df_data):
    sen_features,goals=pipeline_setup_searching_algorithm(config,df_data)
    if config["continous_groups"]:
        sen_groups,group_scores=pipeline_continous_group_generation(config,df_data,sen_features,goals)
    else:
        sen_groups,group_scores=pipeline_patch_group_generation(config,df_data,sen_features,goals)
    if len(sen_groups)<=0:
        config["sensitive_random_state"]+=1
        return pipeline_generate_sensitive_groups(config,df_data)
    else:
    	return sen_groups,group_scores

def pipeline_calculate_biasScore(config,df_data,sen_groups):
    bs_dict={"Group Size Disparity":gsd(df_data,sen_groups),"Prevalence Disparity":prevd(df_data,sen_groups)}
    bs_dict.update(generate_lankmarking_score(df_data,sen_groups,config["landmarking_random_state"]))
    return bs_dict

def pipeline_calculate_metaFeature(config,df_data):
    mfe = MFE(groups=["complexity",'landmarking','info-theory',"model-based"],summary=["mean"],measure_time="total_summ",suppress_warnings=True,random_state=config["landmarking_random_state"])
    mfe.fit(df_data.X.to_numpy(),df_data.Y.to_numpy(),verbose=True,suppress_warnings=True)
    ft = mfe.extract()
    mf_dict={ft[0][i]:ft[1][i] for i in range(len(ft[0]))}
    return mf_dict 	
