import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

def split_data(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits data into Train, Validation, and Test sets.
    If test_size=0.15 and val_size=0.15, train_size will be 0.70.
    """

    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    val_prop = val_size / (1.0 - test_size)

    train_df, val_df = train_test_split(train_val_df, test_size=val_prop, random_state=random_state)

    return train_df, val_df, test_df
