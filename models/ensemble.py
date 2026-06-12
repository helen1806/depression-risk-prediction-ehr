import numpy as np

class EnsembleVoter:
    def __init__(self, models: dict):
        """
        Initializes the soft voting ensemble model.
        models: Dict mapping model name to the model instance/wrapper that implements `predict_proba`.
        """
        self.models = models

    def predict_proba(self, X) -> np.ndarray:
        """
        Computes the average probability across all models.
        X is assumed to be appropriately preprocessed and convertable for each model.
        """
        probas = []
        for name, model_wrapper in self.models.items():
            prob = model_wrapper.predict_proba(X)

            if prob.ndim == 2:
                prob = prob[:, 1]
            probas.append(prob)

        return np.mean(probas, axis=0)

    def predict(self, X, threshold=0.5) -> np.ndarray:
        """
        Hard predictions based on soft voting average.
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)
