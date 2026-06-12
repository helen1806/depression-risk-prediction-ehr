import pandas as pd
from sklearn.impute import SimpleImputer
from typing import Tuple

def impute_missing_values(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, strategy: str = "mean") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Imputes missing values in the datasets.
    """
    imputer = SimpleImputer(strategy=strategy)

    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_val_imp = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    return X_train_imp, X_val_imp, X_test_imp
