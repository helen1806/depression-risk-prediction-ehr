import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from typing import Tuple, List

def easy_undersample(X: pd.DataFrame, y: pd.Series, n_subsamples: int = 5) -> List[Tuple[pd.DataFrame, pd.Series]]:
    """
    EasyEnsemble: repeatedly sample minority-matched majority subsets.
    """
    X_arr = np.array(X)
    y_arr = np.array(y)

    pos_idx = np.where(y_arr == 1)[0]
    neg_idx = np.where(y_arr == 0)[0]

    n_pos = len(pos_idx)
    subsamples = []

    for _ in range(n_subsamples):

        neg_sub = np.random.choice(neg_idx, size=n_pos, replace=False)

        selected_idx = np.concatenate([pos_idx, neg_sub])
        np.random.shuffle(selected_idx)

        X_sub = pd.DataFrame(X_arr[selected_idx], columns=X.columns)
        y_sub = pd.Series(y_arr[selected_idx], name=y.name)

        subsamples.append((X_sub, y_sub))

    return subsamples

def smote_oversample(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    SMOTE: Oversample minority class.
    """
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)
    return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)
