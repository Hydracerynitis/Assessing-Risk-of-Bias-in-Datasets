import pandas as pd
import numpy as np
import importlib
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder

def reload_module(module):
    importlib.reload(module)


def set_seed(random_state):
    if random_state!=None:
        np.random.seed(random_state)

class df_data:
    def __init__(self,X,Y,desired,encoder=None) -> None:
        self.X=X
        self.Y=Y
        if encoder is not None:
            if isinstance(encoder,OrdinalEncoder):
                self.desired=encoder.transform(pd.DataFrame({encoder.feature_names_in_[0]:desired})).T[0]
            elif isinstance(encoder,LabelEncoder):
                self.desired=encoder.transform(desired)
            else:
                self.desired=[encoder.get_loc(d) for d in desired]
        else:
            self.desired=desired


            LabelEncoder