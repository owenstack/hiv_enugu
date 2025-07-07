import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from hiv_enugu.modeling.features import create_ml_features
from hiv_enugu.config import MODELS_DIR
import joblib
import os


def build_ensemble_models_with_cv(X, y, cv_fitted_models, cv_splits):
    """Builds and evaluates ensemble models using cross-validation."""
    ensemble_metrics = {}

    for model_name in ["Random Forest", "Gradient Boosting"]:
        cv_test_rmse, cv_test_r2, cv_test_mae = [], [], []

        for i, (train_index, test_index) in enumerate(cv_splits):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Create features using growth models from the current fold
            fold_fitted_models = {name: cv_fitted_models[name][i] for name in cv_fitted_models}

            # Check if all models for the fold are valid
            if not all(fold_fitted_models.values()):
                print(f"Skipping fold {i + 1} for {model_name} due to missing growth model.")
                continue

            train_features = create_ml_features(X_train, fold_fitted_models, X_train)
            test_features = create_ml_features(X_test, fold_fitted_models, X_train)

            scaler = StandardScaler().fit(train_features)
            train_features_scaled = scaler.transform(train_features)
            test_features_scaled = scaler.transform(test_features)

            if model_name == "Random Forest":
                model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10)
            else:
                model = GradientBoostingRegressor(random_state=42, n_estimators=100, max_depth=5)

            model.fit(train_features_scaled, y_train)
            y_pred = model.predict(test_features_scaled)

            cv_test_rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))
            cv_test_r2.append(r2_score(y_test, y_pred))
            cv_test_mae.append(mean_absolute_error(y_test, y_pred))

        ensemble_metrics[model_name] = {
            "test_rmse": np.mean(cv_test_rmse),
            "test_r2": np.mean(cv_test_r2),
            "test_mae": np.mean(cv_test_mae),
        }

    return {}, ensemble_metrics


def build_ensemble_models(X, y, fitted_models, model_metrics, cv_splits):
    """Builds and evaluates ensemble models with a focus on clarity and robustness."""
    ensemble_models = {}
    ensemble_metrics = {}
    # Store weights for later reporting
    # This will be a dict like: {'Logistic': {'r2_weight': 0.x, 'inv_mse_weight': 0.y}, ...}
    global_model_weights_for_reporting = {}

    # Base features from individual models (predictions on the full X)
    base_features_dict = {
        name: model["function"](X, *model["parameters"]) for name, model in fitted_models.items()
    }
    base_features_array = np.column_stack(list(base_features_dict.values()))
    model_names_in_order = list(fitted_models.keys())

    # Simple Average Ensemble
    simple_avg_pred = np.mean(base_features_array, axis=1)
    ensemble_models["Simple Average"] = {
        "predict": lambda x_new: np.mean(
            np.column_stack(
                [m["function"](x_new, *m["parameters"]) for m in fitted_models.values()]
            ),
            axis=1,
        ),
        "type": "simple_average",
    }
    ensemble_metrics["Simple Average"] = {
        "test_rmse": np.sqrt(mean_squared_error(y, simple_avg_pred)),
        "test_r2": r2_score(y, simple_avg_pred),
        "test_mae": mean_absolute_error(y, simple_avg_pred),
    }

    # --- Weighted Average Ensembles ---
    # Weights from R2 scores
    weights_r2 = np.array(
        [model_metrics[name].get("test_r2", 0) for name in model_names_in_order], dtype=np.float64
    )
    weights_r2 = np.maximum(weights_r2, 0)  # Ensure non-negative weights (R2 can be negative)
    if weights_r2.sum() > 0:
        weights_r2 /= weights_r2.sum()
    else:
        weights_r2 = np.ones(len(model_names_in_order), dtype=np.float64) / len(
            model_names_in_order
        )

    for i, name in enumerate(model_names_in_order):
        if name not in global_model_weights_for_reporting:
            global_model_weights_for_reporting[name] = {}
        global_model_weights_for_reporting[name]["r2_weight"] = weights_r2[i]

    weighted_avg_pred_r2 = np.sum(base_features_array * weights_r2.reshape(1, -1), axis=1)
    ensemble_models["Weighted Average (R2)"] = {
        "predict": lambda x_new: np.sum(
            np.column_stack(
                [
                    fitted_models[name]["function"](x_new, *fitted_models[name]["parameters"])
                    for name in model_names_in_order
                ]
            )
            * weights_r2.reshape(1, -1),
            axis=1,
        ),
        "weights_type": "R2",
        "weights": dict(zip(model_names_in_order, weights_r2)),
        "type": "weighted_average",
    }
    ensemble_metrics["Weighted Average (R2)"] = {
        "test_rmse": np.sqrt(mean_squared_error(y, weighted_avg_pred_r2)),
        "test_r2": r2_score(y, weighted_avg_pred_r2),
        "test_mae": mean_absolute_error(y, weighted_avg_pred_r2),
    }

    # Weights from Inverse MSE
    # Using test_rmse from model_metrics, squaring it to get MSE
    mse_values = np.array(
        [model_metrics[name].get("test_rmse", np.inf) ** 2 for name in model_names_in_order],
        dtype=np.float64,
    )
    # Inverse MSE: 1/MSE. If MSE is 0 or very small, this can be problematic. Add epsilon.
    epsilon = 1e-9
    weights_inv_mse = 1.0 / (mse_values + epsilon)
    if weights_inv_mse.sum() > 0:
        weights_inv_mse /= weights_inv_mse.sum()
    else:  # Fallback if all MSEs were huge or inf
        weights_inv_mse = np.ones(len(model_names_in_order), dtype=np.float64) / len(
            model_names_in_order
        )

    for i, name in enumerate(model_names_in_order):
        if name not in global_model_weights_for_reporting:
            global_model_weights_for_reporting[name] = {}
        global_model_weights_for_reporting[name]["inv_mse_weight"] = weights_inv_mse[i]

    weighted_avg_pred_inv_mse = np.sum(
        base_features_array * weights_inv_mse.reshape(1, -1), axis=1
    )
    ensemble_models["Weighted Average (InvMSE)"] = {
        "predict": lambda x_new: np.sum(
            np.column_stack(
                [
                    fitted_models[name]["function"](x_new, *fitted_models[name]["parameters"])
                    for name in model_names_in_order
                ]
            )
            * weights_inv_mse.reshape(1, -1),
            axis=1,
        ),
        "weights_type": "InverseMSE",
        "weights": dict(zip(model_names_in_order, weights_inv_mse)),
        "type": "weighted_average",
    }
    ensemble_metrics["Weighted Average (InvMSE)"] = {
        "test_rmse": np.sqrt(mean_squared_error(y, weighted_avg_pred_inv_mse)),
        "test_r2": r2_score(y, weighted_avg_pred_inv_mse),
        "test_mae": mean_absolute_error(y, weighted_avg_pred_inv_mse),
    }

    # Store the collected weights in a way the pipeline can access it if needed, e.g., by returning it
    # For now, it's in global_model_weights_for_reporting, which is not ideal for return.
    # Let's add it to the ensemble_models dict for the weighted models themselves.
    ensemble_models["Weighted Average (R2)"]["component_weights"] = (
        global_model_weights_for_reporting
    )
    ensemble_models["Weighted Average (InvMSE)"]["component_weights"] = (
        global_model_weights_for_reporting
    )

    # ML-based Ensembles

    ml_features = create_ml_features(X, fitted_models, X)
    scaler = StandardScaler().fit(ml_features)
    ml_features_scaled = scaler.transform(ml_features)

    # Hyperparameter tuning (simplified)
    rf_grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid={"n_estimators": [100, 200], "max_depth": [10, 20]},
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    ).fit(ml_features_scaled, y)
    best_rf = rf_grid.best_estimator_

    gb_grid = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid={"n_estimators": [100, 200], "max_depth": [5, 10]},
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    ).fit(ml_features_scaled, y)
    best_gb = gb_grid.best_estimator_

    # Unified prediction function for ML models
    def make_ml_predictor(model, scaler, x_min, x_max):
        def predictor(x_new):
            base_preds_new = np.column_stack(
                [m["function"](x_new, *m["parameters"]) for m in fitted_models.values()]
            )
            ml_features_new = create_ml_features(x_new, fitted_models, x_new)
            return model.predict(scaler.transform(ml_features_new))

        return predictor

        # Finalizing ML models

    feature_names = list(
        pd.DataFrame(ml_features).columns
    )  # Get feature names from the DataFrame returned by create_ml_features

    for name, model_obj in [("Random Forest", best_rf), ("Gradient Boosting", best_gb)]:
        # The make_ml_predictor already exists and correctly uses the scaler and create_ml_features
        # which ensures that the features used for prediction are consistent with training.
        # We need to ensure the `model_obj` (best_rf, best_gb) is the one used for feature importance.

        predictor = make_ml_predictor(
            model_obj, scaler, X.min(), X.max()
        )  # X here is the original time_idx
        y_pred_on_full_X_for_metrics = model_obj.predict(
            ml_features_scaled
        )  # Predict on the same features used for training for overall metrics

        ensemble_models[name] = {
            "predict": predictor,  # This predictor can take new time_idx values
            "model": model_obj,  # The actual trained model object
            "scaler": scaler,
            "feature_names": feature_names,  # Store feature names
            "type": name.lower().replace(" ", "_"),  # e.g. "random_forest"
        }
        ensemble_metrics[name] = {
            "test_rmse": np.sqrt(mean_squared_error(y, y_pred_on_full_X_for_metrics)),
            "test_r2": r2_score(y, y_pred_on_full_X_for_metrics),
            "test_mae": mean_absolute_error(y, y_pred_on_full_X_for_metrics),
        }

    # Save models
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_rf, MODELS_DIR / "rf_model.pkl")
    joblib.dump(best_gb, MODELS_DIR / "gb_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "feature_scaler.pkl")

    return ensemble_models, ensemble_metrics
