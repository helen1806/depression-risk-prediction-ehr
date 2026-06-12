from xgboost import XGBClassifier

def create_xgboost_model(n_estimators: int = 100, max_depth: int = 3, learning_rate: float = 0.1, random_state: int = 42):
    """
    Creates an XGBoost Classifier replacing the placeholder proxy block.
    """
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        eval_metric='logloss',
        n_jobs=-1
    )
    return model
