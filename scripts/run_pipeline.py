import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import hydra
from omegaconf import DictConfig
from preprocessing.ingestion import load_data
from preprocessing.features import extract_features_target, scale_features
from preprocessing.imputation import impute_missing_values
from preprocessing.splitting import split_data
from preprocessing.sampling import easy_undersample, smote_oversample
from training.trainers import PyTorchWrapper, train_model
from models.lasso import LassoLogisticRegression
from models.mlp import RFApproximationMLP
from models.ensemble import EnsembleVoter
from evaluation.holdout_eval import evaluate_on_holdout
from explainability.shap_utils import generate_shap_explanations
from datasets.mock_data import generate_mock_ehr_data

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print("=== EHR Depression Prediction Pipeline ===")

    if not os.path.exists(cfg.dataset.data_path):
        print("Mock data not found, generating...")
        generate_mock_ehr_data(output_path=cfg.dataset.data_path)

    df = load_data(cfg.dataset.data_path)

    X, y = extract_features_target(df, cfg.dataset.target_col)

    train_df, val_df, test_df = split_data(df, test_size=cfg.dataset.test_size, val_size=cfg.dataset.val_size)
    X_train, y_train = extract_features_target(train_df, cfg.dataset.target_col)
    X_val, y_val = extract_features_target(val_df, cfg.dataset.target_col)
    X_test, y_test = extract_features_target(test_df, cfg.dataset.target_col)

    X_train_imp, X_val_imp, X_test_imp = impute_missing_values(X_train, X_val, X_test, strategy=cfg.preprocessing.imputation_strategy)

    X_train_scl, X_val_scl, X_test_scl = scale_features(X_train_imp, X_val_imp, X_test_imp)

    print(f"Applying Sampling Method: {cfg.preprocessing.sampling_method}")
    if cfg.preprocessing.sampling_method == "smote":
        X_train_res, y_train_res = smote_oversample(X_train_scl, y_train)
    else:
        print("Using first EasyEnsemble subsample for demonstration.")
        subsamples = easy_undersample(X_train_scl, y_train, n_subsamples=1)
        X_train_res, y_train_res = subsamples[0]

    input_dim = X_train_res.shape[1]

    print("Training Random Forest...")
    rf_model = train_model('rf', X_train_res, y_train_res,
                           n_estimators=cfg.models.rf.n_estimators,
                           max_depth=cfg.models.rf.max_depth)

    print("Training XGBoost...")
    xgb_model = train_model('xgboost', X_train_res, y_train_res,
                            n_estimators=cfg.models.xgboost.n_estimators,
                            max_depth=cfg.models.xgboost.max_depth,
                            learning_rate=cfg.models.xgboost.learning_rate)

    print("Training LASSO Logistic Regression...")
    lasso_pt = LassoLogisticRegression(input_dim)
    lasso_model = PyTorchWrapper(lasso_pt, epochs=cfg.training.epochs, lr=cfg.training.lr,
                                 batch_size=cfg.training.batch_size, device=cfg.training.device,
                                 is_lasso=True, l1_lambda=cfg.training.l1_lambda)
    lasso_model.fit(X_train_res, y_train_res)

    print("Training MLP Approximation...")
    mlp_pt = RFApproximationMLP(input_dim, hidden_dims=cfg.models.mlp.hidden_dims, dropout=cfg.models.mlp.dropout)
    mlp_model = PyTorchWrapper(mlp_pt, epochs=cfg.training.epochs, lr=cfg.training.lr,
                               batch_size=cfg.training.batch_size, device=cfg.training.device)
    mlp_model.fit(X_train_res, y_train_res)

    print("Building Ensemble...")
    ensemble = EnsembleVoter({
        'RandomForest': rf_model,
        'XGBoost': xgb_model,
        'LASSO': lasso_model,
        'MLP': mlp_model
    })

    print("Evaluating Ensemble on Holdout Test Set...")
    metrics = evaluate_on_holdout(ensemble, X_test_scl, y_test, output_dir="outputs")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    if cfg.explainability.use_shap:
        print("Generating SHAP Explanations...")
        generate_shap_explanations(xgb_model, X_train_res.sample(min(100, len(X_train_res))), X_test_scl.sample(min(50, len(X_test_scl))), model_type="tree")
        print("Pipeline Complete! Check 'outputs' folder for figures.")

if __name__ == "__main__":
    main()
