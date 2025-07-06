import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from hiv_enugu.modeling.features import create_ml_features
import joblib
import os


def build_ensemble_models(X, y, fitted_models, model_metrics, cv_splits):
    """Builds and evaluates ensemble models with a focus on clarity and robustness."""
    ensemble_models = {}
    ensemble_metrics = {}

    # Base features from individual models
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

    # Weighted Average Ensemble (weights from individual model R2 scores)
    weights = np.array([model_metrics[name].get("r2", 0) for name in fitted_models.keys()])
    if weights.sum() > 0:
        weights /= weights.sum()
    else:
        weights = np.ones(len(fitted_models)) / len(fitted_models)

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

    # ML-based Ensembles

    ml_features = create_ml_features(X, base_features, X_full=X)
    scaler = StandardScaler().fit(ml_features)
    ml_features_scaled = scaler.transform(ml_features)

    # Hyperparameter tuning (simplified)
    rf_grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid={"n_estimators": [100, 200], "max_depth": [10, 20]},
        cv=3, scoring="neg_mean_squared_error", n_jobs=-1
    ).fit(ml_features_scaled, y)
    best_rf = rf_grid.best_estimator_

    gb_grid = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid={"n_estimators": [100, 200], "max_depth": [5, 10]},
        cv=3, scoring="neg_mean_squared_error", n_jobs=-1
    ).fit(ml_features_scaled, y)
    best_gb = gb_grid.best_estimator_

    # Unified prediction function for ML models
    def make_ml_predictor(model, scaler, x_min, x_max):
        def predictor(x_new):
            base_preds_new = np.column_stack(
                [m["function"](x_new, *m["parameters"]) for m in fitted_models.values()]
            )
            ml_features_new = create_ml_features(x_new, base_preds_new, X_full=x_new)
            return model.predict(scaler.transform(ml_features_new))
        return predictor

    # Finalizing ML models
    for name, model in [("Random Forest", best_rf), ("Gradient Boosting", best_gb)]:
        predictor = make_ml_predictor(model, scaler, X.min(), X.max())
        y_pred = predictor(X)
        ensemble_models[name] = {"predict": predictor, "model": model, "scaler": scaler}
        ensemble_metrics[name] = {
            "rmse": np.sqrt(mean_squared_error(y, y_pred)),
            "r2": r2_score(y, y_pred),
            "mae": mean_absolute_error(y, y_pred),
        }

    # Save models
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(best_rf, "saved_models/rf_model.pkl")
    joblib.dump(best_gb, "saved_models/gb_model.pkl")
    joblib.dump(scaler, "saved_models/feature_scaler.pkl")

    return ensemble_models, ensemble_metrics