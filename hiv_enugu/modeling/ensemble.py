import numpy as np
import pandas as pd # Not directly used in the function but good for context
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit # TimeSeriesSplit might be needed if cv_splits isn't passed directly
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings

warnings.filterwarnings('ignore') # As in the provided code

# Ensemble Model Building with Improved Feature Engineering and Weight Calculation
def build_ensemble_models(X, y, fitted_models, cv_splits):
    """Build ensemble models with improved feature engineering and weighting"""

    if not fitted_models:
        print("Warning: fitted_models is empty. Cannot build ensemble models.")
        return {}, {}
    if len(X) == 0 or len(y) == 0:
        print("Warning: X or y is empty. Cannot build ensemble models.")
        return {}, {}

    # Create base features from model predictions on the full X range
    # These are used for training the final ensemble model and for creating features during prediction.
    base_features_full = np.column_stack([
        model['function'](X, *model['parameters'])
        for model_name, model in fitted_models.items() # Iterate through items to keep order if Python < 3.7
    ])

    # Add engineered features for the full X range
    time_idx_norm_full = (X - X.min()) / (X.max() - X.min() + 1e-9) # Add epsilon to avoid division by zero if X is constant

    feature_matrix_full = np.column_stack([
        base_features_full,
        time_idx_norm_full,
        np.sin(2 * np.pi * time_idx_norm_full),
        np.cos(2 * np.pi * time_idx_norm_full)
    ])

    # Initialize metrics accumulators for CV
    ensemble_cv_metrics = {
        'Simple Average': {'train_rmse': [], 'test_rmse': [], 'train_r2': [], 'test_r2': [], 'test_mae': []},
        'Weighted Average': {'train_rmse': [], 'test_rmse': [], 'train_r2': [], 'test_r2': [], 'test_mae': []},
        'Random Forest': {'train_rmse': [], 'test_rmse': [], 'train_r2': [], 'test_r2': [], 'test_mae': []},
        'Gradient Boosting': {'train_rmse': [], 'test_rmse': [], 'train_r2': [], 'test_r2': [], 'test_mae': []}
    }

    rf_param_grid = {
        'n_estimators': [100, 200], # Reduced for speed, original: [100, 200, 300]
        'max_depth': [None, 10, 15],   # Reduced, original: [None, 10, 15, 20]
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2'] # Reduced, original: ['sqrt', 'log2', None]
    }

    gb_param_grid = {
        'n_estimators': [100, 200], # Reduced, original: [100, 200, 300]
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5],       # Reduced, original: [3, 5, 7]
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'subsample': [0.8, 1.0]    # Reduced, original: [0.8, 0.9, 1.0]
    }

    # Initialize models for hyperparameter tuning
    rf_model_for_tuning = RandomForestRegressor(random_state=42)
    gb_model_for_tuning = GradientBoostingRegressor(random_state=42)

    # Store best params from GridSearchCV
    best_rf_params = None
    best_gb_params = None

    # Store weights for the Weighted Average model from the last CV fold
    final_model_weights = np.ones(base_features_full.shape[1]) / base_features_full.shape[1]

    if not cv_splits:
        print("Warning: cv_splits is empty. Ensemble models will be trained on full data without CV metrics.")
        # Handle training on full data if no CV splits are provided (simplified)
        # This part would need careful implementation if full no-CV path is desired
        # For now, we assume cv_splits will be provided. If not, ML models won't be tuned via CV.
    else:
        # Process each CV split
        for i, (train_index, test_index) in enumerate(cv_splits):
            X_train_fold, X_test_fold = X[train_index], X[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]

            # Create base features for this fold
            train_base_features_fold = np.column_stack([
                model['function'](X_train_fold, *model['parameters'])
                for model in fitted_models.values()
            ])
            test_base_features_fold = np.column_stack([
                model['function'](X_test_fold, *model['parameters'])
                for model in fitted_models.values()
            ])

            # --- Simple Average ---
            simple_avg_train_pred = np.mean(train_base_features_fold, axis=1)
            simple_avg_test_pred = np.mean(test_base_features_fold, axis=1)
            for metric, val_train, val_test in [
                ('train_rmse', np.sqrt(mean_squared_error(y_train_fold, simple_avg_train_pred)), np.sqrt(mean_squared_error(y_test_fold, simple_avg_test_pred))),
                ('test_rmse', np.sqrt(mean_squared_error(y_train_fold, simple_avg_train_pred)), np.sqrt(mean_squared_error(y_test_fold, simple_avg_test_pred))), # test_rmse uses test data
                ('train_r2', r2_score(y_train_fold, simple_avg_train_pred), r2_score(y_test_fold, simple_avg_test_pred)),
                ('test_r2', r2_score(y_train_fold, simple_avg_train_pred), r2_score(y_test_fold, simple_avg_test_pred)), # test_r2 uses test data
                ('test_mae', mean_absolute_error(y_train_fold, simple_avg_train_pred), mean_absolute_error(y_test_fold, simple_avg_test_pred)) # test_mae uses test data
            ]:
                ensemble_cv_metrics['Simple Average'][metric].append(val_test if 'test' in metric else val_train)


            # --- Weighted Average (Exponential Weighting) ---
            current_fold_weights = []
            for j_model_idx in range(train_base_features_fold.shape[1]):
                model_train_pred = train_base_features_fold[:, j_model_idx]
                model_test_pred = test_base_features_fold[:, j_model_idx]

                train_r2_val = float(max(0.0, r2_score(y_train_fold, model_train_pred)))
                test_r2_val = float(max(0.0, r2_score(y_test_fold, model_test_pred)))
                weight = np.exp(train_r2_val + test_r2_val) # As per user's code
                current_fold_weights.append(weight)

            current_fold_weights = np.array(current_fold_weights)
            if sum(current_fold_weights) > 1e-9: # Avoid division by zero
                current_fold_weights_norm = current_fold_weights / sum(current_fold_weights)
            else: # Fallback to equal weights if all weights are zero
                current_fold_weights_norm = np.ones(len(current_fold_weights)) / len(current_fold_weights)

            if i == len(cv_splits) - 1: # Store weights from the last fold for the final model
                final_model_weights = current_fold_weights_norm

            weighted_avg_train_pred = np.sum(train_base_features_fold * current_fold_weights_norm.reshape(1, -1), axis=1)
            weighted_avg_test_pred = np.sum(test_base_features_fold * current_fold_weights_norm.reshape(1, -1), axis=1)
            for metric, val_train, val_test in [
                ('train_rmse', np.sqrt(mean_squared_error(y_train_fold, weighted_avg_train_pred)), np.sqrt(mean_squared_error(y_test_fold, weighted_avg_test_pred))),
                ('test_rmse', np.sqrt(mean_squared_error(y_train_fold, weighted_avg_train_pred)), np.sqrt(mean_squared_error(y_test_fold, weighted_avg_test_pred))),
                ('train_r2', r2_score(y_train_fold, weighted_avg_train_pred), r2_score(y_test_fold, weighted_avg_test_pred)),
                ('test_r2', r2_score(y_train_fold, weighted_avg_train_pred), r2_score(y_test_fold, weighted_avg_test_pred)),
                ('test_mae', mean_absolute_error(y_train_fold, weighted_avg_train_pred), mean_absolute_error(y_test_fold, weighted_avg_test_pred))
            ]:
                ensemble_cv_metrics['Weighted Average'][metric].append(val_test if 'test' in metric else val_train)

            # --- ML Models (RF, GB) ---
            # Create full features for this fold (base + engineered)
            train_time_idx_norm_fold = (X_train_fold - X.min()) / (X.max() - X.min() + 1e-9)
            test_time_idx_norm_fold = (X_test_fold - X.min()) / (X.max() - X.min() + 1e-9)

            train_features_fold = np.column_stack([
                train_base_features_fold, train_time_idx_norm_fold,
                np.sin(2 * np.pi * train_time_idx_norm_fold), np.cos(2 * np.pi * train_time_idx_norm_fold)
            ])
            test_features_fold = np.column_stack([
                test_base_features_fold, test_time_idx_norm_fold,
                np.sin(2 * np.pi * test_time_idx_norm_fold), np.cos(2 * np.pi * test_time_idx_norm_fold)
            ])

            scaler_fold = StandardScaler()
            train_features_scaled_fold = scaler_fold.fit_transform(train_features_fold)
            test_features_scaled_fold = scaler_fold.transform(test_features_fold)

            # Hyperparameter tuning on the first split only
            if i == 0:
                print("\nTuning Random Forest hyperparameters on the first CV split...")
                rf_grid = GridSearchCV(rf_model_for_tuning, rf_param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, error_score='raise')
                rf_grid.fit(train_features_scaled_fold, y_train_fold)
                best_rf_params = rf_grid.best_params_
                print(f"Best Random Forest parameters: {best_rf_params}")

                print("\nTuning Gradient Boosting hyperparameters on the first CV split...")
                gb_grid = GridSearchCV(gb_model_for_tuning, gb_param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, error_score='raise')
                gb_grid.fit(train_features_scaled_fold, y_train_fold)
                best_gb_params = gb_grid.best_params_
                print(f"Best Gradient Boosting parameters: {best_gb_params}")

            # Instantiate models with best_params_ (or default if first split)
            current_rf_model = RandomForestRegressor(**(best_rf_params or {}), random_state=42)
            current_gb_model = GradientBoostingRegressor(**(best_gb_params or {}), random_state=42)

            current_rf_model.fit(train_features_scaled_fold, y_train_fold)
            current_gb_model.fit(train_features_scaled_fold, y_train_fold)

            for model_name, model_instance in [('Random Forest', current_rf_model), ('Gradient Boosting', current_gb_model)]:
                train_pred = model_instance.predict(train_features_scaled_fold)
                test_pred = model_instance.predict(test_features_scaled_fold)
                for metric, val_train, val_test in [
                    ('train_rmse', np.sqrt(mean_squared_error(y_train_fold, train_pred)), np.sqrt(mean_squared_error(y_test_fold, test_pred))),
                    ('test_rmse', np.sqrt(mean_squared_error(y_train_fold, train_pred)), np.sqrt(mean_squared_error(y_test_fold, test_pred))),
                    ('train_r2', r2_score(y_train_fold, train_pred), r2_score(y_test_fold, test_pred)),
                    ('test_r2', r2_score(y_train_fold, train_pred), r2_score(y_test_fold, test_pred)),
                    ('test_mae', mean_absolute_error(y_train_fold, train_pred), mean_absolute_error(y_test_fold, test_pred))
                ]:
                    ensemble_cv_metrics[model_name][metric].append(val_test if 'test' in metric else val_train)

    # Compute final average metrics from CV
    final_ensemble_metrics = {}
    for name, metrics_dict in ensemble_cv_metrics.items():
        final_ensemble_metrics[name] = {
            metric_name: np.mean(values) if values else np.nan
            for metric_name, values in metrics_dict.items()
        }
        print(f"\n{name} CV Avg Metrics:")
        for mn, mv in final_ensemble_metrics[name].items():
            print(f"  {mn}: {mv:.4f}")

    # --- Train final ML models on the full dataset using best hyperparameters ---
    final_scaler = StandardScaler()
    full_features_scaled = final_scaler.fit_transform(feature_matrix_full)

    # Use default params if tuning wasn't performed (e.g. no CV splits)
    if best_rf_params is None: best_rf_params = {'n_estimators': 100, 'random_state': 42} # Default
    if best_gb_params is None: best_gb_params = {'n_estimators': 100, 'random_state': 42} # Default

    final_rf_model = RandomForestRegressor(**best_rf_params, random_state=42)
    final_rf_model.fit(full_features_scaled, y)

    final_gb_model = GradientBoostingRegressor(**best_gb_params, random_state=42)
    final_gb_model.fit(full_features_scaled, y)

    # Save final models and scaler
    output_model_dir = 'saved_models' # As per user's code structure
    os.makedirs(output_model_dir, exist_ok=True)
    joblib.dump(final_rf_model, os.path.join(output_model_dir, 'rf_model.pkl'))
    joblib.dump(final_gb_model, os.path.join(output_model_dir, 'gb_model.pkl'))
    joblib.dump(final_scaler, os.path.join(output_model_dir, 'feature_scaler.pkl'))

    # --- Define final ensemble models structure for prediction ---
    # Helper for ML model prediction, encapsulating feature creation and scaling
    def _create_ml_features_for_predict(X_new_time_idx, fitted_mdls, x_min_train, x_max_train):
        base_feats = np.column_stack([
            mdl['function'](X_new_time_idx, *mdl['parameters']) for mdl in fitted_mdls.values()
        ])
        time_idx_norm_new = (X_new_time_idx - x_min_train) / (x_max_train - x_min_train + 1e-9)
        return np.column_stack([
            base_feats, time_idx_norm_new,
            np.sin(2 * np.pi * time_idx_norm_new), np.cos(2 * np.pi * time_idx_norm_new)
        ])

    # Store X.min() and X.max() from training for consistent normalization at prediction time
    # These are from the original X passed to build_ensemble_models
    train_X_min = X.min()
    train_X_max = X.max()

    final_ensemble_models_for_prediction = {
        'Simple Average': {
            'predict': lambda x_new_time_idx: np.mean(np.column_stack([
                model['function'](x_new_time_idx, *model['parameters']) for model in fitted_models.values()
            ]), axis=1)
        },
        'Weighted Average': {
            'predict': lambda x_new_time_idx: np.sum(np.column_stack([
                model['function'](x_new_time_idx, *model['parameters']) for model in fitted_models.values()
            ]) * final_model_weights.reshape(1, -1), axis=1),
            'weights': final_model_weights # Store the weights used
        },
        'Random Forest': {
            'model': final_rf_model,
            'scaler': final_scaler,
            'create_features_func': lambda x_new_time_idx: _create_ml_features_for_predict(x_new_time_idx, fitted_models, train_X_min, train_X_max),
            'predict': lambda x_new_time_idx: final_rf_model.predict(
                final_scaler.transform(
                    _create_ml_features_for_predict(x_new_time_idx, fitted_models, train_X_min, train_X_max)
                )
            )
        },
        'Gradient Boosting': {
            'model': final_gb_model,
            'scaler': final_scaler,
            'create_features_func': lambda x_new_time_idx: _create_ml_features_for_predict(x_new_time_idx, fitted_models, train_X_min, train_X_max),
            'predict': lambda x_new_time_idx: final_gb_model.predict(
                final_scaler.transform(
                    _create_ml_features_for_predict(x_new_time_idx, fitted_models, train_X_min, train_X_max)
                )
            )
        }
    }

    return final_ensemble_models_for_prediction, final_ensemble_metrics
