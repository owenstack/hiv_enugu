import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from .growth_models import exponential_model, logistic_model, richards_model, gompertz_model
from hiv_enugu.plotting.diagnostics import plot_residuals, plot_qq, plot_residuals_histogram


def fit_growth_models(X, y, cv_splits):
    """Fit individual growth models with cross-validation and improved convergence"""
    models = {
        'Exponential': (exponential_model, ([np.max(y)/2 if np.max(y) > 0 else 1, 0.005, np.min(y) if len(y) > 0 else 0],
                                          [0, 0.0001, -np.inf],
                                          [np.max(y)*2 if np.max(y) > 0 else 2, 0.1, np.inf])),
        'Logistic': (logistic_model, ([np.max(y)*1.2 if np.max(y) > 0 else 1.2, 0.01, np.median(X) if len(X)>0 else 0.5, np.min(y) if len(y) > 0 else 0],
                                    [np.max(y) if np.max(y) > 0 else 1, 0.001, X.min() if len(X)>0 else 0, -np.inf],
                                    [np.max(y)*2 if np.max(y) > 0 else 2, 0.1, X.max() if len(X)>0 else 1, np.inf])),
        'Richards': (richards_model, ([np.max(y)*1.2 if np.max(y) > 0 else 1.2, 0.01, np.median(X) if len(X)>0 else 0.5, 1, np.min(y) if len(y) > 0 else 0],
                                    [np.max(y) if np.max(y) > 0 else 1, 0.001, X.min() if len(X)>0 else 0, 0.1, -np.inf],
                                    [np.max(y)*2 if np.max(y) > 0 else 2, 0.1, X.max() if len(X)>0 else 1, 10, np.inf])),
        'Gompertz': (gompertz_model, ([np.max(y)*1.2 if np.max(y) > 0 else 1.2, 2, 0.01, np.min(y) if len(y) > 0 else 0],
                                    [np.max(y) if np.max(y) > 0 else 1, 0.1, 0.001, -np.inf],
                                    [np.max(y)*2 if np.max(y) > 0 else 2, 10, 0.1, np.inf])) # Using 10 from new code
    }

    fitted_models = {}
    model_metrics = {}

    # Handle cases with empty X or y
    if len(X) == 0 or len(y) == 0:
        print("Error: X or y is empty. Cannot fit models.")
        return fitted_models, model_metrics
    if np.max(y) <= 0 : # Handles cases where y might be all zeros or negative
        print("Warning: Max value of y is not positive. Model fitting might be problematic.")
        # Adjust bounds to be non-zero if they depend on max(y) being positive.
        # The ternary operators in model definitions attempt to handle this.

    for name, (model_func, bounds_config) in models.items():
        # Dynamic bounds based on data, with safeguards for empty/non-positive y
        p0_initial = list(bounds_config[0])
        bounds_low = list(bounds_config[1])
        bounds_high = list(bounds_config[2])

        # Ensure p0 is within bounds
        for i in range(len(p0_initial)):
            p0_initial[i] = np.clip(p0_initial[i], bounds_low[i], bounds_high[i])

        bounds = (bounds_low, bounds_high)

        try:
            print(f"\nFitting {name} Model with cross-validation...")

            cv_train_rmse, cv_test_rmse, cv_train_r2, cv_test_r2_scores, cv_test_mae = [], [], [], [], []
            best_popt_overall = None # Best popt across all CV folds, for final fit initialization
            best_overall_test_r2 = -np.inf # R2 score for best_popt_overall

            # First fit on full data to get better initial parameters for CV step
            try:
                full_popt_initial, _ = curve_fit(model_func, X, y,
                                       p0=p0_initial,
                                       bounds=bounds,
                                       maxfev=100000,
                                       method='trf',
                                       ftol=1e-6, xtol=1e-6, gtol=1e-6) # Added tolerances
                initial_params_for_cv = full_popt_initial
            except RuntimeError:
                print(f"  Warning: Initial fit on full data failed for {name}. Using default p0 for CV.")
                initial_params_for_cv = p0_initial
            except ValueError as ve:
                print(f"  ValueError during initial fit for {name}: {ve}. Using default p0 for CV.")
                initial_params_for_cv = p0_initial

            if not cv_splits: # Ensure cv_splits is not empty
                print(f"  Warning: cv_splits is empty for {name}. Skipping CV.")
                # Fallback: fit on full data only if no CV splits
                try:
                    final_popt, _ = curve_fit(model_func, X, y, p0=initial_params_for_cv, bounds=bounds, maxfev=100000, method='trf')
                    y_pred_full = model_func(X, *final_popt)
                    residuals_full = y - y_pred_full
                    plot_residuals(y, y_pred_full, name, filename_suffix="_full_data_no_cv", filename=f"residuals_{name}_full_data_no_cv.png")
                    plot_qq(residuals_full, name, filename_suffix="_full_data_no_cv", filename=f"qq_plot_{name}_full_data_no_cv.png")
                    plot_residuals_histogram(residuals_full, name, filename_suffix="_full_data_no_cv", filename=f"residuals_histogram_{name}_full_data_no_cv.png")

                    model_metrics[name] = {
                        'train_rmse': np.sqrt(mean_squared_error(y, y_pred_full)), 'test_rmse': np.nan,
                        'train_r2': r2_score(y, y_pred_full), 'test_r2': np.nan,
                        'test_mae': np.nan, 'parameters': final_popt, 'cv_results': {}
                    }
                    fitted_models[name] = {'function': model_func, 'parameters': final_popt}
                    print(f"{name} Model (No CV) Metrics: Train RMSE: {model_metrics[name]['train_rmse']:.2f}, Train R2: {model_metrics[name]['train_r2']:.4f}")
                    continue # Skip to next model
                except Exception as e_full_fit:
                    print(f"  Error fitting {name} on full data (no CV): {e_full_fit}")
                    continue


            # Loop through CV splits
            for i, (train_index, test_index) in enumerate(cv_splits):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                if len(X_train) < len(initial_params_for_cv) or len(y_train) < len(initial_params_for_cv):
                    print(f"  Warning: Not enough data points in split {i+1} for {name} (Train: {len(X_train)}). Skipping split.")
                    continue

                best_split_popt = None
                best_split_r2 = -np.inf

                for attempt in range(5): # Try multiple initializations
                    try:
                        p0_attempt = initial_params_for_cv
                        if attempt > 0: # Perturb for subsequent attempts
                            noise = np.random.normal(0, 0.1 * (attempt / 4.0), len(initial_params_for_cv))
                            p0_attempt = initial_params_for_cv * (1 + noise)
                            p0_attempt = np.clip(p0_attempt, bounds[0], bounds[1])

                        popt_split, _ = curve_fit(model_func, X_train, y_train,
                                          p0=p0_attempt, bounds=bounds, maxfev=50000, method='trf',
                                          ftol=1e-6, xtol=1e-6, gtol=1e-6)

                        y_test_pred_split = model_func(X_test, *popt_split)
                        current_test_r2 = r2_score(y_test, y_test_pred_split)

                        if current_test_r2 > best_split_r2:
                            best_split_r2 = current_test_r2
                            best_split_popt = popt_split

                    except RuntimeError:
                        # print(f"    Attempt {attempt+1} for split {i+1} failed to converge.")
                        continue
                    except ValueError as ve_split:
                        # print(f"    ValueError in attempt {attempt+1} for split {i+1}: {ve_split}")
                        continue # problem with bounds or p0

                if best_split_popt is None:
                    print(f"  Warning: Failed to converge on split {i+1} for {name} after all attempts. Using initial_params_for_cv as fallback for this split's metrics.")
                    best_split_popt = initial_params_for_cv

                y_train_pred = model_func(X_train, *best_split_popt)
                y_test_pred = model_func(X_test, *best_split_popt)

                cv_train_rmse.append(np.sqrt(mean_squared_error(y_train, y_train_pred)))
                cv_test_rmse.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))
                cv_train_r2.append(r2_score(y_train, y_train_pred))
                current_fold_test_r2 = r2_score(y_test, y_test_pred)
                cv_test_r2_scores.append(current_fold_test_r2)
                cv_test_mae.append(mean_absolute_error(y_test, y_test_pred))

                if current_fold_test_r2 > best_overall_test_r2: # Check if this split's popt is the best overall
                    best_overall_test_r2 = current_fold_test_r2
                    best_popt_overall = best_split_popt

                # print(f"  Split {i+1}: Test RMSE = {cv_test_rmse[-1]:.2f}, R² = {cv_test_r2_scores[-1]:.4f}")

            if best_popt_overall is None: # If all splits failed or no better popt found
                best_popt_overall = initial_params_for_cv # Fallback to initial parameters from full data fit

            # Final fit using best parameters from CV (or initial full data fit) as initialization
            final_popt, _ = curve_fit(model_func, X, y,
                                    p0=best_popt_overall, bounds=bounds, maxfev=100000, method='trf',
                                    ftol=1e-7, xtol=1e-7, gtol=1e-7) # Stricter tolerance for final fit

            y_pred_full = model_func(X, *final_popt)
            residuals_full = y - y_pred_full

            plot_residuals(y, y_pred_full, name, filename_suffix="_full_data", filename=f"residuals_{name}_full_data.png")
            plot_qq(residuals_full, name, filename_suffix="_full_data", filename=f"qq_plot_{name}_full_data.png")
            plot_residuals_histogram(residuals_full, name, filename_suffix="_full_data", filename=f"residuals_histogram_{name}_full_data.png")

            model_metrics[name] = {
                'train_rmse': np.mean(cv_train_rmse) if cv_train_rmse else np.nan,
                'test_rmse': np.mean(cv_test_rmse) if cv_test_rmse else np.nan,
                'train_r2': np.mean(cv_train_r2) if cv_train_r2 else np.nan,
                'test_r2': np.mean(cv_test_r2_scores) if cv_test_r2_scores else np.nan,
                'test_mae': np.mean(cv_test_mae) if cv_test_mae else np.nan,
                'parameters': final_popt,
                'cv_results': {
                    'train_rmse': cv_train_rmse, 'test_rmse': cv_test_rmse,
                    'train_r2': cv_train_r2, 'test_r2': cv_test_r2_scores, 'test_mae': cv_test_mae
                }
            }

            fitted_models[name] = {'function': model_func, 'parameters': final_popt}

            print(f"{name} Model Cross-Validation Metrics:")
            print(f"  Avg Train RMSE: {np.mean(cv_train_rmse):.2f} ± {np.std(cv_train_rmse):.2f}" if cv_train_rmse else "  Avg Train RMSE: N/A")
            print(f"  Avg Test RMSE: {np.mean(cv_test_rmse):.2f} ± {np.std(cv_test_rmse):.2f}" if cv_test_rmse else "  Avg Test RMSE: N/A")
            print(f"  Avg Train R²: {np.mean(cv_train_r2):.4f} ± {np.std(cv_train_r2):.4f}" if cv_train_r2 else "  Avg Train R²: N/A")
            print(f"  Avg Test R²: {np.mean(cv_test_r2_scores):.4f} ± {np.std(cv_test_r2_scores):.4f}" if cv_test_r2_scores else "  Avg Test R²: N/A")
            print(f"  Avg Test MAE: {np.mean(cv_test_mae):.2f} ± {np.std(cv_test_mae):.2f}" if cv_test_mae else "  Avg Test MAE: N/A")

        except Exception as e:
            print(f"Error fitting {name} model: {e}")
            import traceback
            traceback.print_exc()

    return fitted_models, model_metrics
