import numpy as np


def generate_bootstrap_predictions(model_func, X, params, n_samples=100, confidence=0.95):
    """Generate bootstrap predictions with confidence intervals"""
    predictions = []
    for _ in range(n_samples):
        # Add random noise to parameters for bootstrapping
        noise = np.random.normal(0, 0.05, len(params))
        bootstrap_params = params * (1 + noise)
        predictions.append(model_func(X, *bootstrap_params))

    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    ci_lower = np.percentile(predictions, (100 - confidence * 100) / 2, axis=0)
    ci_upper = np.percentile(predictions, (100 + confidence * 100) / 2, axis=0)

    return mean_pred, ci_lower, ci_upper
