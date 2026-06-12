from sklearn.ensemble import RandomForestClassifier

def create_rf_model(n_estimators: int = 100, max_depth: int = 10, random_state: int = 42):
    """
    Creates a Random Forest Classifier.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    return model
