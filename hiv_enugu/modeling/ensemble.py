import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from hiv_enugu.config import MODELS_DIR
from hiv_enugu.modeling.features import create_ml_features


# Renamed from build_ensemble_models
def build_average_ensembles(X, y, cv_fitted_growth_models, cv_splits):
    """
    Builds Simple Average and Weighted Average ensemble models using final fitted growth models.
    """
    avg_ensemble_models = {}
    avg_ensemble_metrics = {}

    # Use the final models (fitted on full data, which are the last ones produced by fit_growth_models)
    # The cv_fitted_growth_models has one model per fold. We need the final one.
    # The structure passed from fit.py is a dict of lists. Let's get the final models from the main pipeline's fitted_models_global
    # A cleaner way is to just refit here or pass them correctly. Let's assume cv_fitted_growth_models contains fold-specific models.
    # We will use the models from the *last* CV split as a proxy for the final models for calculating weights.

    final_fold_growth_models = {
        name: models[-1]
        for name, models in cv_fitted_growth_models.items()
        if models and models[-1] is not None
    }

    if not final_fold_growth_models:
        print("No fitted growth models provided for average ensembles. Skipping.")
        return avg_ensemble_models, avg_ensemble_metrics

    base_features_dict = {
        name: model["function"](X, *model["parameters"])
        for name, model in final_fold_growth_models.items()
    }
    base_features_array = np.column_stack(list(base_features_dict.values()))

    # --- Simple Average Ensemble ---
    def create_simple_predict(models):
        def simple_predict(x_new):
            predictions = np.column_stack(
                [m["function"](x_new.ravel(), *m["parameters"]) for m in models.values()]
            )
            return np.mean(predictions, axis=1)

        return simple_predict

    simple_avg_pred = np.mean(base_features_array, axis=1)
    avg_ensemble_models["Simple Average"] = {
        "predict": create_simple_predict(final_fold_growth_models),
        "type": "simple_average",
    }
    avg_ensemble_metrics["Simple Average"] = {
        "test_rmse": np.sqrt(mean_squared_error(y, simple_avg_pred)),
        "test_r2": r2_score(y, simple_avg_pred),
        "test_mae": mean_absolute_error(y, simple_avg_pred),
    }

    # --- Weighted Average Ensembles ---
    # Calculate weights based on the performance on the last CV test set
    _, test_index = cv_splits[-1]
    X_test, y_test = X[test_index], y[test_index]

    model_names, r2_weights, inv_mse_weights = [], [], []
    component_weights_for_reporting = {}

    for name, model in final_fold_growth_models.items():
        test_pred = model["function"](X_test, *model["parameters"])
        r2 = r2_score(y_test, test_pred)
        mse = mean_squared_error(y_test, test_pred)

        # Only use models with positive R-squared for weighting
        if r2 > 0:
            model_names.append(name)
            r2_weights.append(r2)
            inv_mse_weights.append(1 / (mse + 1e-9))  # Add epsilon for stability

            # **FIX**: Populate the reporting dictionary
            component_weights_for_reporting[name] = {
                "r2_score": r2,
                "mse": mse,
            }

    if not model_names:
        print("Warning: No models with positive R2 found. Skipping weighted average ensembles.")
        return avg_ensemble_models, avg_ensemble_metrics

    # Normalize weights
    r2_weights = np.array(r2_weights) / np.sum(r2_weights)
    inv_mse_weights = np.array(inv_mse_weights) / np.sum(inv_mse_weights)

    # **FIX**: Add weights to the reporting dict
    for i, name in enumerate(model_names):
        component_weights_for_reporting[name]["r2_weight"] = r2_weights[i]
        component_weights_for_reporting[name]["inv_mse_weight"] = inv_mse_weights[i]

    # Create closures for prediction
    def create_weighted_predict(weights, models_dict, model_order):
        def weighted_predict(x_new):
            predictions = np.column_stack(
                [
                    models_dict[name]["function"](x_new.ravel(), *models_dict[name]["parameters"])
                    for name in model_order
                ]
            )
            return np.average(predictions, axis=1, weights=weights)

        return weighted_predict

    # Weighted Average (R2)
    wa_r2_pred = np.average(
        np.column_stack([base_features_dict[name] for name in model_names]),
        axis=1,
        weights=r2_weights,
    )
    avg_ensemble_models["Weighted Average (R2)"] = {
        "predict": create_weighted_predict(r2_weights, final_fold_growth_models, model_names),
        "weights": dict(zip(model_names, r2_weights)),
        "component_weights": component_weights_for_reporting,  # Pass the populated dict
        "type": "weighted_average",
    }
    avg_ensemble_metrics["Weighted Average (R2)"] = {
        "test_rmse": np.sqrt(mean_squared_error(y, wa_r2_pred)),
        "test_r2": r2_score(y, wa_r2_pred),
        "test_mae": mean_absolute_error(y, wa_r2_pred),
    }

    # **FIX**: Implement Weighted Average (InvMSE)
    wa_invmse_pred = np.average(
        np.column_stack([base_features_dict[name] for name in model_names]),
        axis=1,
        weights=inv_mse_weights,
    )
    avg_ensemble_models["Weighted Average (InvMSE)"] = {
        "predict": create_weighted_predict(inv_mse_weights, final_fold_growth_models, model_names),
        "weights": dict(zip(model_names, inv_mse_weights)),
        "component_weights": component_weights_for_reporting,
        "type": "weighted_average",
    }
    avg_ensemble_metrics["Weighted Average (InvMSE)"] = {
        "test_rmse": np.sqrt(mean_squared_error(y, wa_invmse_pred)),
        "test_r2": r2_score(y, wa_invmse_pred),
        "test_mae": mean_absolute_error(y, wa_invmse_pred),
    }

    return avg_ensemble_models, avg_ensemble_metrics


def build_ensemble_models_with_cv(
    X,
    y,
    cv_fitted_growth_models,
    cv_splits,
):
    """
    Builds and evaluates ML ensemble models using cross-validation for feature generation,
    and then builds average-based ensembles using final fitted growth models.
    """
    ml_ensemble_models = {}
    ml_ensemble_metrics = {}

    # --- Part 1: ML-based Ensembles (Random Forest, Gradient Boosting) with CV-derived features ---
    # These models are trained on features derived from out-of-fold predictions of growth models.
    # Their metrics are also CV-based.

    # Create a placeholder for the scaler and feature names, to be determined from the full data training
    # This scaler will be fit on features from the *entire* X data using *final* growth models
    # to ensure consistency for predictions and feature importance.

    # First, create ML features using the *final* fitted growth models on the *entire* dataset X
    # This is for training the *final* ML models and getting a consistent scaler.
    final_fitted_growth_models = {
        name: models[0] for name, models in cv_fitted_growth_models.items() if models
    }
    final_ml_features = create_ml_features(
        X, final_fitted_growth_models, X
    )  # X_train_context=X ensures it uses X as context
    final_scaler = StandardScaler().fit(final_ml_features)
    final_ml_features_scaled = final_scaler.transform(final_ml_features)

    # Store feature names from these final features
    # Ensure ml_features is a DataFrame to get .columns
    if isinstance(final_ml_features, pd.DataFrame):
        ml_feature_names = list(final_ml_features.columns)
    elif (
        isinstance(final_ml_features, np.ndarray) and final_ml_features.shape[1] > 0
    ):  # if it's a numpy array
        ml_feature_names = [f"feature_{i}" for i in range(final_ml_features.shape[1])]
    else:
        print(
            "Warning: Could not determine ML feature names. Defaulting to generic names if possible."
        )
        # Attempt to get number of features from a sample if final_ml_features is problematic
        try:
            sample_fold_models = {
                name: cv_fitted_growth_models[name][0] for name in cv_fitted_growth_models
            }
            if any(sample_fold_models.values()):
                sample_features = create_ml_features(
                    X[cv_splits[0][0]], sample_fold_models, X[cv_splits[0][0]]
                )
                ml_feature_names = [f"feature_{i}" for i in range(sample_features.shape[1])]
            else:
                ml_feature_names = []
        except Exception:  # Specify Exception
            ml_feature_names = []

    for model_name in ["Random Forest", "Gradient Boosting"]:
        cv_test_rmse, cv_test_r2, cv_test_mae = [], [], []

        # This loop is for calculating CV metrics for RF and GB
        for i, (train_index, test_index) in enumerate(cv_splits):
            X_train_fold, X_test_fold = X[train_index], X[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]

            fold_specific_growth_models = {
                name: cv_fitted_growth_models[name][i] for name in cv_fitted_growth_models
            }

            if not all(fold_specific_growth_models.values()):
                print(f"Skipping fold {i + 1} for {model_name} CV due to missing growth model.")
                continue

            # Features for CV are created using growth models from *this specific fold*
            train_features_fold = create_ml_features(
                X_train_fold, fold_specific_growth_models, X_train_fold
            )
            test_features_fold = create_ml_features(
                X_test_fold, fold_specific_growth_models, X_train_fold
            )  # Context is train_fold

            # Scaler for CV fold is fit only on this fold's training data
            scaler_fold = StandardScaler().fit(train_features_fold)
            train_features_fold_scaled = scaler_fold.transform(train_features_fold)
            test_features_fold_scaled = scaler_fold.transform(test_features_fold)

            if model_name == "Random Forest":
                model_fold = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10)
            else:  # Gradient Boosting
                model_fold = GradientBoostingRegressor(
                    random_state=42, n_estimators=100, max_depth=5
                )

            model_fold.fit(train_features_fold_scaled, y_train_fold)
            y_pred_fold = model_fold.predict(test_features_fold_scaled)

            cv_test_rmse.append(np.sqrt(mean_squared_error(y_test_fold, y_pred_fold)))
            cv_test_r2.append(r2_score(y_test_fold, y_pred_fold))
            cv_test_mae.append(mean_absolute_error(y_test_fold, y_pred_fold))

        ml_ensemble_metrics[model_name] = {
            "test_rmse": np.mean(cv_test_rmse) if cv_test_rmse else np.nan,
            "test_r2": np.mean(cv_test_r2) if cv_test_r2 else np.nan,
            "test_mae": np.mean(cv_test_mae) if cv_test_mae else np.nan,
        }

    # Now, train final RF and GB models on the *entire dataset X* using features from *final_fitted_growth_models*
    # This uses final_ml_features_scaled and final_scaler defined earlier.

    # Hyperparameter tuning for final models (can be simplified or expanded)
    rf_grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid={"n_estimators": [100, 200], "max_depth": [10, 20]},
        cv=3,  # Inner CV for hyperparameter tuning on the full training set
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    ).fit(final_ml_features_scaled, y)
    best_rf_final = rf_grid.best_estimator_

    gb_grid = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid={"n_estimators": [100, 200], "max_depth": [5, 10]},
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    ).fit(final_ml_features_scaled, y)
    best_gb_final = gb_grid.best_estimator_

    # Unified prediction function for final ML models
    def make_final_ml_predictor(
        model, scaler, growth_models_for_feature_creation, original_X_context
    ):
        def predictor(x_new):
            # When predicting on new data x_new, features must be created using the same context (original_X_context)
            # that was used for training the growth_models_for_feature_creation, if those features depend on it.
            # create_ml_features uses X_train_context for things like residuals or interactions with historical values.
            ml_features_new = create_ml_features(
                x_new, growth_models_for_feature_creation, original_X_context
            )
            return model.predict(scaler.transform(ml_features_new))

        return predictor

    ml_ensemble_models["Random Forest"] = {
        "predict": make_final_ml_predictor(
            best_rf_final, final_scaler, final_fitted_growth_models, X
        ),
        "model": best_rf_final,
        "scaler": final_scaler,
        "feature_names": ml_feature_names,
        "type": "random_forest",
    }

    ml_ensemble_models["Gradient Boosting"] = {
        "predict": make_final_ml_predictor(
            best_gb_final, final_scaler, final_fitted_growth_models, X
        ),
        "model": best_gb_final,
        "scaler": final_scaler,
        "feature_names": ml_feature_names,
        "type": "gradient_boosting",
    }

    # Save final ML models and the scaler
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_rf_final, MODELS_DIR / "rf_model.pkl")
    joblib.dump(best_gb_final, MODELS_DIR / "gb_model.pkl")
    joblib.dump(final_scaler, MODELS_DIR / "feature_scaler.pkl")

    # --- Part 2: Average-based Ensembles (Simple Average, Weighted Average) ---
    # These use the *final* fitted growth models and their overall metrics.
    avg_ensemble_models, avg_ensemble_metrics = build_average_ensembles(
        X, y, cv_fitted_growth_models, cv_splits
    )

    # --- Part 3: Combine models and metrics ---
    combined_ensemble_models = {**ml_ensemble_models, **avg_ensemble_models}
    combined_ensemble_metrics = {**ml_ensemble_metrics, **avg_ensemble_metrics}

    return combined_ensemble_models, combined_ensemble_metrics
