import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class PyTorchWrapper:
    """
    Wraps PyTorch models to provide a scikit-learn like API (predict, predict_proba).
    """
    def __init__(self, model, epochs=50, lr=0.01, batch_size=32, device="cpu", is_lasso=False, l1_lambda=1e-4):
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.is_lasso = is_lasso
        self.l1_lambda = l1_lambda

    def fit(self, X, y):
        self.model = self.model.to(self.device)

        X_tensor = torch.tensor(X.values if hasattr(X, 'values') else X, dtype=torch.float32)
        y_tensor = torch.tensor(y.values if hasattr(y, 'values') else y, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        self.model.train()

        for epoch in range(self.epochs):
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch).squeeze()
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)

                loss = criterion(outputs, y_batch)

                if self.is_lasso:
                    loss += self.model.lasso_penalty(self.l1_lambda)

                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X.values if hasattr(X, 'values') else X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            probs = self.model(X_tensor).squeeze().cpu().numpy()

        if np.isscalar(probs):
            probs = np.array([probs])

        return np.column_stack([1 - probs, probs])

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

def train_model(model_type, X_train, y_train, **kwargs):
    """
    Unified training function for both scikit-learn and PyTorch wrappers.
    """
    if model_type == 'rf':
        from models.rf_model import create_rf_model
        model = create_rf_model(**kwargs)
        model.fit(X_train, y_train)
        return model
    elif model_type == 'xgboost':
        from models.xgboost_model import create_xgboost_model
        model = create_xgboost_model(**kwargs)
        model.fit(X_train, y_train)
        return model
    else:
        raise ValueError(f"Unsupported basic model type: {model_type}")
