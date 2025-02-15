import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

from artifact.sensitive_ssr import InsensitiveSubspace

def square_average(data):
    if sum(data)==0:
        return 0
    return sum([d*d for d in data])/sum(data)

def avergae(data):
    return sum(data)/len(data)

def z_average(data):
    mean=np.mean(data)
    std=np.std(data)
    if std==0:
        std=1
    z_score=np.array([d-mean for d in data])/std
    return np.sum(z_score * np.array(data))

def gsd(df_data,sensitive_groups,flush=False):
    group_size=[]
    X=df_data.X
    background=InsensitiveSubspace(X,sensitive_groups)
    sensitive_groups=sensitive_groups+[background]
    for sg in sensitive_groups:
        sg_df=X[sg.select(X)]
        gs=len(sg_df)/len(X)
        group_size.append(gs)
    if flush:
        print(group_size)
    return z_average(group_size)

def prevd(df_data,sensitive_groups,flush=False):
    prevs=[]
    X,Y,desired=df_data.X,df_data.Y,df_data.desired
    background=InsensitiveSubspace(X,sensitive_groups)
    sensitive_groups=sensitive_groups+[background]
    ideal=len(Y[Y.isin(desired)])/len(Y) # df[df[label].isin(desired)]
    for sg in sensitive_groups:
        split_df=X[sg.select(X)] #df[df[sensitive_groups]==i]
        if len(split_df)==0:
            prevs.append(0)
            continue
        prevs.append(len(split_df.loc[Y.isin(desired)])/len(split_df))
    disparity=[abs(p-ideal) for p in prevs]
    if flush:
        print(ideal)
        print(prevs)
        print(disparity)
    return avergae(disparity)

def equal_opportunity(df_X,df_Y,desired,sensitive_groups,pred):
    equal_opportunity=np.array([])
    for sen_g in sensitive_groups:
        sg_df=df_X[sen_g.select(df_X) & pred.isin(desired)]
        if sg_df.shape[0]==0:
            tpr=0
        else:
            tpr=sg_df.loc[df_Y.isin(desired)].shape[0]/sg_df.shape[0]
        equal_opportunity=np.append(equal_opportunity,tpr)
    background_group=InsensitiveSubspace(df_X,sensitive_groups)
    background=df_X[background_group.select(df_X) & pred.isin(desired)]
    if background.shape[0]==0:
        background_tpr=0
    else:
        background_tpr=background.loc[df_Y.isin(desired)].shape[0]/background.shape[0]
    equal_opportunity=abs(equal_opportunity-background_tpr)
    return avergae(equal_opportunity)

def disparate_parity(df_X,desired,sensitive_groups,pred):
    disparate_parity=np.array([])
    for sen_g in sensitive_groups:
        sg_df=df_X[sen_g.select(df_X)]
        sg_pos=sg_df.loc[pred.isin(desired)]
        if sg_df.shape[0]==0:
            disparate_parity=np.append(disparate_parity,0)
        else:
            disparate_parity=np.append(disparate_parity,sg_pos.shape[0]/sg_df.shape[0])
    background=InsensitiveSubspace(df_X,sensitive_groups)
    background_df=df_X[background.select(df_X)]
    background_pos=background_df.loc[pred.isin(desired)]
    if background_df.shape[0]==0:
        disparate_parity=abs(disparate_parity-0)
    else:
        disparate_parity=abs(disparate_parity-background_pos.shape[0]/background_df.shape[0])
    return avergae(disparate_parity)

def generalized_entropy_index(df_X,df_Y,sensitive_groups,pred):
    benifit=pred-df_Y+1
    mean_benifit=benifit.mean()
    individual_benifit_gain=np.power(benifit/mean_benifit,2)-1 
    individual_fairness=individual_benifit_gain.sum()/(benifit.shape[0]*2)
    within_group_fairness,between_group_fairness=[],[]
    background_g=InsensitiveSubspace(df_X,sensitive_groups)
    sensitive_groups=sensitive_groups+[background_g]
    for sen_g in sensitive_groups:
        sen_benifit=benifit[sen_g.select(df_X)]
        if sen_benifit.shape[0]<=0:
            continue
        sen_mean_benifit=sen_benifit.mean()
        sen_benifit_gain=np.power(sen_benifit/sen_mean_benifit,2)-1 
        sen_fairness=sen_benifit_gain.sum()/(sen_benifit.shape[0]*2)
        sen_group_fairnes=np.power(sen_mean_benifit/mean_benifit,2)
        sen_group_size=sen_benifit.shape[0]/benifit.shape[0]
        within_group_fairness.append(sen_group_fairnes*sen_group_size*sen_fairness)
        between_group_fairness.append((sen_group_fairnes-1)*sen_group_size/2)
    group_fairness=sum(within_group_fairness)+sum(between_group_fairness)
    return individual_fairness,group_fairness


def generate_lankmarking_score(df_data,sensitive_groups,random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    df_X,df_Y,desired=df_data.X,df_data.Y,df_data.desired
    metrics={"equal_opportunity":[],"disparate_parity":[],"individual_fairness":[],"group_fairness":[]}
    for _,(train_index,test_index) in enumerate(StratifiedKFold(shuffle=True,random_state=random_state).split(df_X,df_Y)):
        train_X,test_X=df_X.iloc[train_index],df_X.iloc[test_index]
        train_Y,test_Y=df_Y.iloc[train_index],df_Y.iloc[test_index]
        random_node=DecisionTreeClassifier(max_depth=5, random_state=random_state)
        linear=LinearRegression()
        random_node.fit(train_X,train_Y)
        linear.fit(train_X,train_Y)
        node_pred=pd.Series(random_node.predict(test_X),index=test_X.index)
        linear_pred=pd.Series(linear.predict(test_X),index=test_X.index)
        individual_fairness,group_fairness= generalized_entropy_index(test_X,test_Y,sensitive_groups,linear_pred)
        metrics["individual_fairness"].append(individual_fairness)
        metrics["group_fairness"].append(group_fairness)
        metrics["equal_opportunity"].append(equal_opportunity(test_X,test_Y,desired,sensitive_groups,node_pred))
        metrics["disparate_parity"].append(disparate_parity(test_X,desired,sensitive_groups,node_pred))
    for k,v in metrics.items():
        metrics[k]=sum(v)/len(v)
    return metrics
