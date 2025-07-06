import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from hiv_enugu.config import MODELS_DIR
import joblib
import os


def build_ensemble_models(X, y, fitted_models, model_metrics, cv_splits):
    """Builds and evaluates ensemble models."""
    ensemble_models = {}
    ensemble_metrics = {}

    # Prepare base features from fitted individual models
    base_features = np.column_stack(
        [model["function"](X, *model["parameters"]) for model in fitted_models.values()]
    )

    # Simple Average Ensemble
    simple_avg_pred = np.mean(base_features, axis=1)
    ensemble_models["Simple Average"] = {
        "predict": lambda x: np.mean(
            np.column_stack([m["function"](x, *m["parameters"]) for m in fitted_models.values()]),
            axis=1,
        )
    }
    ensemble_metrics["Simple Average"] = {
        "rmse": np.sqrt(mean_squared_error(y, simple_avg_pred)),
        "r2": r2_score(y, simple_avg_pred),
        "mae": mean_absolute_error(y, simple_avg_pred),
    }
    print("\nSimple Average Ensemble Metrics:")
    print(
        f"  RMSE: {ensemble_metrics['Simple Average']['rmse']:.2f}, R²: {ensemble_metrics['Simple Average']['r2']:.4f}, MAE: {ensemble_metrics['Simple Average']['mae']:.2f}"
    )

    # Weighted Average Ensemble (weights based on R2 from individual models)
    model_weights = []
    # Weights are based on the test R2 score from cross-validation
    for name in fitted_models.keys():
        model_weights.append(model_metrics[name]["test_r2"])
    model_weights = np.array(model_weights)
    if np.sum(model_weights) > 0:
        weights = model_weights / np.sum(model_weights)
    else:
        weights = np.ones(len(fitted_models)) / float(len(fitted_models))  # Fallback to equal weights

    weighted_avg_pred = np.sum(base_features * weights.reshape(1, -1), axis=1)
    ensemble_models["Weighted Average"] = {
        "predict": lambda x: np.sum(
            np.column_stack([m["function"](x, *m["parameters"]) for m in fitted_models.values()])
            * weights.reshape(1, -1),
            axis=1,
        )
    }
    ensemble_metrics["Weighted Average"] = {
        "rmse": np.sqrt(mean_squared_error(y, weighted_avg_pred)),
        "r2": r2_score(y, weighted_avg_pred),
        "mae": mean_absolute_error(y, weighted_avg_pred),
    }
    print("\nWeighted Average Ensemble Metrics:")
    print(
        f"  RMSE: {ensemble_metrics['Weighted Average']['rmse']:.2f}, R²: {ensemble_metrics['Weighted Average']['r2']:.4f}, MAE: {ensemble_metrics['Weighted Average']['mae']:.2f}"
    )

    # Machine Learning Ensembles (Random Forest, Gradient Boosting)
    # Feature engineering for ML models
    time_idx_norm = (X - X.min()) / (X.max() - X.min() + 1e-9)
    ml_features = np.column_stack(
        [
            base_features,
            time_idx_norm,
            np.sin(2 * np.pi * time_idx_norm),
            np.cos(2 * np.pi * time_idx_norm),
        ]
    )

    scaler = StandardScaler()
    ml_features_scaled = scaler.fit_transform(ml_features)

    # Random Forest
    rf_model = RandomForestRegressor(random_state=42)
    rf_param_grid = {"n_estimators": [100, 200], "max_depth": [10, 20]}
    rf_grid = GridSearchCV(
        rf_model, rf_param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1
    )
    rf_grid.fit(ml_features_scaled, y)
    best_rf = rf_grid.best_estimator_
    rf_pred = best_rf.predict(ml_features_scaled)
    ensemble_models["Random Forest"] = {
        "model": best_rf,
        "scaler": scaler,
        "predict": lambda x_new: best_rf.predict(
            scaler.transform(
                np.column_stack(
                    [
                        np.column_stack(
                            [
                                m["function"](x_new, *m["parameters"])
                                for m in fitted_models.values()
                            ]
                        ),
                        (x_new - X.min()) / (X.max() - X.min() + 1e-9),
                        np.sin(2 * np.pi * ((x_new - X.min()) / (X.max() - X.min() + 1e-9))),
                        np.cos(2 * np.pi * ((x_new - X.min()) / (X.max() - X.min() + 1e-9))),
                    ]
                )
            )
        ),
    }
    ensemble_metrics["Random Forest"] = {
        "rmse": np.sqrt(mean_squared_error(y, rf_pred)),
        "r2": r2_score(y, rf_pred),
        "mae": mean_absolute_error(y, rf_pred),
    }
    print("\nRandom Forest Ensemble Metrics:")
    print(
        f"  RMSE: {ensemble_metrics['Random Forest']['rmse']:.2f}, R²: {ensemble_metrics['Random Forest']['r2']:.4f}, MAE: {ensemble_metrics['Random Forest']['mae']:.2f}"
    )

    # Gradient Boosting
    gb_model = GradientBoostingRegressor(random_state=42)
    gb_param_grid = {"n_estimators": [100, 200], "max_depth": [5, 10]}
    gb_grid = GridSearchCV(
        gb_model, gb_param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1
    )
    gb_grid.fit(ml_features_scaled, y)
    best_gb = gb_grid.best_estimator_
    gb_pred = best_gb.predict(ml_features_scaled)
    ensemble_models["Gradient Boosting"] = {
        "model": best_gb,
        "scaler": scaler,
        "predict": lambda x_new: best_gb.predict(
            scaler.transform(
                np.column_stack(
                    [
                        np.column_stack(
                            [
                                m["function"](x_new, *m["parameters"])
                                for m in fitted_models.values()
                            ]
                        ),
                        (x_new - X.min()) / (X.max() - X.min() + 1e-9),
                        np.sin(2 * np.pi * ((x_new - X.min()) / (X.max() - X.min() + 1e-9))),
                        np.cos(2 * np.pi * ((x_new - X.min()) / (X.max() - X.min() + 1e-9))),
                    ]
                )
            )
        ),
    }
    ensemble_metrics["Gradient Boosting"] = {
        "rmse": np.sqrt(mean_squared_error(y, gb_pred)),
        "r2": r2_score(y, gb_pred),
        "mae": mean_absolute_error(y, gb_pred),
    }
    print("\nGradient Boosting Ensemble Metrics:")
    print(
        f"  RMSE: {ensemble_metrics['Gradient Boosting']['rmse']:.2f}, R²: {ensemble_metrics['Gradient Boosting']['r2']:.4f}, MAE: {ensemble_metrics['Gradient Boosting']['mae']:.2f}"
    )

    # Save ML models and scaler
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_rf, f"{MODELS_DIR}/rf_model.pkl")
    joblib.dump(best_gb, f"{MODELS_DIR}/gb_model.pkl")
    joblib.dump(scaler, f"{MODELS_DIR}/feature_scaler.pkl")

    return ensemble_models, ensemble_metrics
