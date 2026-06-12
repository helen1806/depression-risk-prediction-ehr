def evaluate_on_holdout(model, X_test, y_test, output_dir="outputs"):

    from evaluation.metrics import calculate_metrics, plot_confusion_matrix, plot_roc_curve, plot_pr_curve
    import json
    import os

    y_prob = model.predict_proba(X_test)
    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]

    y_pred = model.predict(X_test)

    metrics = calculate_metrics(y_test, y_pred, y_prob)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    plot_confusion_matrix(y_test, y_pred, output_dir)
    plot_roc_curve(y_test, y_prob, output_dir)
    plot_pr_curve(y_test, y_prob, output_dir)

    return metrics
