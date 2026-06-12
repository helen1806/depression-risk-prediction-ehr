import pytest
import pandas as pd
import numpy as np
from preprocessing.splitting import split_data
from preprocessing.imputation import impute_missing_values
from models.ensemble import EnsembleVoter

def test_split_data():
    df = pd.DataFrame({
        'f1': range(100),
        'f2': range(100, 200),
        'target': np.random.randint(0, 2, 100)
    })
    train, val, test = split_data(df, test_size=0.2, val_size=0.2)
    assert len(train) == 60
    assert len(val) == 20
    assert len(test) == 20

def test_imputation():
    train = pd.DataFrame({'f1': [1, 2, np.nan], 'f2': [4, 5, 6]})
    val = pd.DataFrame({'f1': [np.nan, 3], 'f2': [4, 5]})
    test = pd.DataFrame({'f1': [1, np.nan], 'f2': [4, np.nan]})

    t_imp, v_imp, te_imp = impute_missing_values(train, val, test, strategy='mean')
    assert not t_imp.isnull().values.any()
    assert not v_imp.isnull().values.any()
    assert not te_imp.isnull().values.any()

    assert t_imp.loc[2, 'f1'] == 1.5

class MockModel:
    def predict_proba(self, X):
        return np.array([[0.2, 0.8], [0.9, 0.1]])

def test_ensemble_voter():
    m1 = MockModel()
    m2 = MockModel()

    ensemble = EnsembleVoter({'m1': m1, 'm2': m2})
    X = np.array([[0], [0]])
    probs = ensemble.predict_proba(X)

    assert np.allclose(probs, [0.8, 0.1])
    preds = ensemble.predict(X, threshold=0.5)
    assert np.all(preds == [1, 0])
