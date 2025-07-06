import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from .growth_models import exponential_model, logistic_model, richards_model, gompertz_model


def fit_individual_models(X, y, cv_splits):
    """Fits individual growth models and returns their metrics and parameters."""
    models = {
        "Exponential": (
            exponential_model,
            ([np.max(y) / 2, 0.005, min(y)], [0, 0.0001, -np.inf], [np.max(y) * 2, 0.1, np.inf]),
        ),
        "Logistic": (
            logistic_model,
            (
                [np.max(y) * 1.2, 0.01, np.median(X), min(y)],
                [np.max(y), 0.001, X.min(), -np.inf],
                [np.max(y) * 2, 0.1, X.max(), np.inf],
            ),
        ),
        "Richards": (
            richards_model,
            (
                [np.max(y) * 1.2, 0.01, np.median(X), 1, min(y)],
                [np.max(y), 0.001, X.min(), 0.1, -np.inf],
                [np.max(y) * 2, 0.1, X.max(), 10, np.inf],
            ),
        ),
        "Gompertz": (
            gompertz_model,
            (
                [np.max(y) * 1.2, 2, 0.01, min(y)],
                [np.max(y), 0.1, 0.001, -np.inf],
                [np.max(y) * 2, 5, 0.1, np.inf],
            ),
        ),
    }

    fitted_models = {}
    model_metrics = {}

    for name, (model_func, bounds) in models.items():
        print(f"\nFitting {name} model...")
        initial_params = bounds[0]
        cv_test_r2 = []
        best_popt = None
        best_test_r2 = -np.inf

        for i, (train_index, test_index) in enumerate(cv_splits):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            try:
                popt, _ = curve_fit(
                    model_func,
                    X_train,
                    y_train,
                    p0=initial_params,
                    bounds=(bounds[1], bounds[2]),
                    maxfev=10000,
                )
                y_test_pred = model_func(X_test, *popt)
                test_r2 = r2_score(y_test, y_test_pred)
                cv_test_r2.append(test_r2)

                if test_r2 > best_test_r2:
                    best_test_r2 = test_r2
                    best_popt = popt
            except RuntimeError:
                print(f"  Split {i + 1}: Could not find optimal parameters.")
                cv_test_r2.append(-1)

        if best_popt is not None:
            final_popt, _ = curve_fit(
                model_func, X, y, p0=best_popt, bounds=(bounds[1], bounds[2]), maxfev=10000
            )
            y_pred = model_func(X, *final_popt)

            fitted_models[name] = {"function": model_func, "parameters": final_popt}
            model_metrics[name] = {
                "r2": r2_score(y, y_pred),
                "rmse": np.sqrt(mean_squared_error(y, y_pred)),
                "mae": mean_absolute_error(y, y_pred),
                "test_r2": np.mean(cv_test_r2),  # Store CV test R2
            }
            print(f"{name} Model CV Avg R²: {np.mean(cv_test_r2):.4f}")

    return fitted_models, model_metrics
