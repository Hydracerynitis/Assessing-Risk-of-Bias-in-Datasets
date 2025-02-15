import numpy as np
import random

from artifact.util import set_seed
from artifact.sensitive_ssr import SensitiveShape, SensitiveRule, SensitiveSubspace

def group_score(sg,df_data):
    X,Y,desired=df_data.X,df_data.Y,df_data.desired
    group_df=X[sg.select(X)]
    if group_df.shape[0]==0:
        return np.array([0,0])
    group_size=group_df.shape[0]/X.shape[0]
    prev=group_df.loc[Y.isin(desired)].shape[0]/X.shape[0]
    return np.array([group_size,prev])

def calculate_goal(weight_matrix,df_data):
    Y,desired=df_data.Y,df_data.desired
    gs=weight_matrix[0]/sum(weight_matrix[0])
    prev_weight=np.prod(weight_matrix,axis=0)
    prev_weight=prev_weight/sum(prev_weight)
    i=Y[Y.isin(desired)].shape[0]/df_data.Y.shape[0]
    prevs=(prev_weight*i)
    return np.array([gs[:-1],prevs[:-1]]).T


# Patch Generation
def move_pointer(pointer, index,range):
    pointer[index]+=1
    if pointer[index]>=range:
        if index+1<len(pointer):
            pointer[index]=0
            move_pointer(pointer,index+1,range)

def generate_candidate(df,k,features):
    features_range=[]
    for f in features:
        f_range=np.linspace(min(df[f]),max(df[f]),k+1)
        features_range.append(f_range)
    pointer=[0 for _ in range(len(features))]
    candidate=[]
    while pointer[-1]<k:
        sensitive_group=SensitiveShape()
        for f_i,i in enumerate(pointer):
            sensitive_group.add(SensitiveRule(features[f_i],features_range[f_i][i],features_range[f_i][i+1]))
        candidate.append(sensitive_group)
        move_pointer(pointer,0,k)
    return candidate

def search_group(candidate,dfd,goal,prev_weight=1,random_walk=0,random_state=None):
    def sort_group(sg):
        score=group_score(sg,dfd)
        return abs(goal[0]-score[0])+abs(goal[1]-score[1])*prev_weight
    sg=SensitiveSubspace()
    current_score=sort_group(sg)
    while (goal>0).any():
        if len(candidate)<=0:
            return sg
        candidate.sort(key=sort_group)
        set_seed(random_state)
        if np.random.random()>random_walk:
            next_best_choice=candidate.pop(0)
        else:
            next_best_choice=candidate.pop(random.randrange(len(candidate)))
        next_score=sort_group(SensitiveSubspace(sg.shapes+[next_best_choice]))
        if current_score-next_score<0:
            return sg
        sg.add(next_best_choice)
        goal=goal-group_score(next_best_choice,dfd)
    return sg

# Continous Groups
def generate_rule(df,k,f_x):
    feature_x=np.linspace(min(df[f_x]),max(df[f_x]),k+1)

    candidate=[]
    for i in range(k):
        sensitive_rule=SensitiveRule(f_x,feature_x[i],feature_x[i+1])
        candidate.append(sensitive_rule)
    return candidate

def join_candidate(base_candidate, candidate_pool=None):
    if candidate_pool==None:
        join_target=base_candidate
    else:
        join_target=candidate_pool
    return {i.join(j) for i in join_target for j in base_candidate if j.joinable(i)}

def filter_candidate(shapes,df_data,goal,coefficient=1.5):
    result=[]
    for c in shapes:
        score=group_score(c,df_data)
        if (score[0]<=goal[0]*coefficient and score[1]<=goal[1]*coefficient) and not exist_similar(c,score,result):
            result.append((c,score))
    return result

def get_candidate_shape(base_rules,candidate_rule):
    if base_rules==None:
        return candidate_rule
    if isinstance(base_rules,SensitiveRule):
        base_rules=[base_rules]
    if isinstance(base_rules,SensitiveShape):
        base_rules=base_rules.rules
    return [SensitiveShape(base_rules+[c]) for c in candidate_rule]

def continous_serach(df_data,k,f,goal,prev_iteration=None,coefficient=1.5):
    candidate=generate_rule(df_data.X,k,f)
    candidate_pool=list(candidate)
    acceptable=[]
    while len(candidate_pool)>1:
        if prev_iteration!=None:
            for (a,_) in prev_iteration:
                acceptable+=filter_candidate(get_candidate_shape(a,candidate_pool),df_data,goal,coefficient)
        else:
            acceptable+=filter_candidate(candidate_pool,df_data,goal,coefficient)
        candidate_pool=join_candidate(candidate_pool,candidate)
    return acceptable

def exist_similar(a,s,new_acceptable):
    for (n_a,n_s) in new_acceptable:
        if s[0]==n_s[0] and s[1]==n_s[1]:
            similar_count=0
            if isinstance(a,SensitiveShape):
                for r in a.rules:
                    n_r=n_a.find(r)
                    if n_r==None:
                        break
                    else:
                        if n_r.bottom==r.bottom:
                            similar_count+=1
                        if n_r.top==r.top:
                            similar_count+=1
                    if similar_count>=len(a.rules)*2-1:
                        return True
            elif isinstance(a,SensitiveRule):
                if a.bottom==n_a.bottom:
                    similar_count+=1
                if a.top==n_a.top:
                    similar_count+=1
                if similar_count>=1:
                    return True
    return False
