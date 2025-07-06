import pandas as pd
import numpy as np
import warnings
import os # For checking file_path

# Custom module imports
from hiv_enugu.data_processing import load_data
from hiv_enugu.modeling.data_prep import prepare_data_for_modeling
from hiv_enugu.modeling.fit import fit_growth_models
from hiv_enugu.modeling.ensemble import build_ensemble_models
from hiv_enugu.utils import generate_bootstrap_predictions

from hiv_enugu.plotting.evaluation import (
    visualize_individual_models,
    visualize_ensemble_comparison,
    visualize_metrics_comparison,
    create_validation_plot,
    forecast_future_trends
)
from hiv_enugu.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR
# plot_basic_timeseries is called within load_data, so not explicitly here.
# Diagnostic plots (residuals, qq, histogram) are called within fit_growth_models.

warnings.filterwarnings('ignore') # As in the user's provided code

# Main function
def main():
    # Define the data file path
    # This path should exist relative to where run_analysis.py is executed
    # or be an absolute path.
    file_path = PROCESSED_DATA_DIR / 'cleaned_enrollments.csv'
    # 1. Load and explore data
    print("Step 1: Loading data...")
    df_weekly = load_data(file_path)

    if df_weekly is None or df_weekly.empty:
        print("Failed to load data or data is empty. Exiting analysis.")
        return
    if 'time_idx' not in df_weekly.columns or 'cumulative' not in df_weekly.columns:
        print("Error: 'time_idx' or 'cumulative' column is missing from loaded data. Exiting.")
        return
    print("Data loading complete.")

    # 2. Prepare data for modeling with cross-validation
    print("\nStep 2: Preparing data for modeling...")
    # Ensure enough data points for n_splits, prepare_data_for_modeling handles adjustments
    # but good to be mindful here too.
    n_modeling_splits = 5
    if len(df_weekly) < n_modeling_splits * 5 and len(df_weekly) > n_modeling_splits : # Heuristic
         pass # prepare_data_for_modeling will adjust
    elif len(df_weekly) <= n_modeling_splits:
        print(f"Warning: Data length ({len(df_weekly)}) is too short for {n_modeling_splits} splits. Analysis may fail or be unreliable.")
        # Potentially reduce n_modeling_splits here or let prepare_data_for_modeling handle it.

    try:
        X, y, cv_splits, tscv_object = prepare_data_for_modeling(df_weekly, n_splits=n_modeling_splits)
    except ValueError as e:
        print(f"Error preparing data for modeling: {e}")
        return
    print("Data preparation complete.")

    # Use last split for some visualization details if needed by plot functions
    # The plotting functions themselves are responsible for using these indices if they need specific train/test sets
    final_train_idx, final_test_idx = cv_splits[-1] if cv_splits else (np.array([]), np.array([]))
    y_train = y[final_train_idx] if final_train_idx.size > 0 else y
    y_test = y[final_test_idx] if final_test_idx.size > 0 else y

    # 3. Fit individual growth models
    print("\nStep 3: Fitting individual growth models...")
    # Global variables as in user's code (though generally better to pass them around)
    # For now, adhering to the structure.
    global fitted_models, model_metrics, best_model_name
    fitted_models, model_metrics = fit_growth_models(X, y, cv_splits)
    if not fitted_models:
        print("No individual models were successfully fitted. Exiting.")
        return
    print("Individual model fitting complete.")

    # 4. Visualize individual model fits
    print("\nStep 4: Visualizing individual model fits...")
    visualize_individual_models(df_weekly, X, final_train_idx, final_train_idx, y_train, y_test, fitted_models,
                                filename='hiv_individual_models_comparison.png')
    print("Individual model fit visualization complete.")

    # 5. Build ensemble models
    print("\nStep 5: Building ensemble models...")
    # The build_ensemble_models from user's code takes X, y, fitted_models, cv_splits
    ensemble_models_dict, ensemble_metrics_dict = build_ensemble_models(X, y, fitted_models, model_metrics, cv_splits)
    if not ensemble_models_dict:
        print("No ensemble models were successfully built. Proceeding with individual models only for remaining steps.")
        # Set ensemble_metrics_dict to empty if no models, to avoid errors later
        ensemble_metrics_dict = {}
    print("Ensemble model building complete.")

    # 6. Identify best models
    print("\nStep 6: Identifying best models...")
    if model_metrics:
        best_model_name = max(model_metrics, key=lambda k: model_metrics[k].get('test_r2', -np.inf))
        print(f"Best Individual Model (by test R²): {best_model_name} (R²: {model_metrics[best_model_name].get('test_r2', -np.inf):.4f})")
    else:
        best_model_name = None
        print("Could not determine best individual model (no metrics).")

    global best_ensemble_model_name # Adhering to user's structure
    if ensemble_metrics_dict:
        best_ensemble_model_name = max(ensemble_metrics_dict, key=lambda k: ensemble_metrics_dict[k].get('test_r2', -np.inf))
        print(f"Best Ensemble Model (by test R²): {best_ensemble_model_name} (R²: {ensemble_metrics_dict[best_ensemble_model_name].get('test_r2', -np.inf):.4f})")
    else:
        best_ensemble_model_name = None
        print("Could not determine best ensemble model (no metrics).")

    # 7. Visualize ensemble comparisons
    if ensemble_models_dict and best_model_name:
        print("\nStep 7: Visualizing ensemble comparisons...")
        visualize_ensemble_comparison(df_weekly, X, y, cv_splits, fitted_models, ensemble_models_dict,
                                      best_model_name, filename='hiv_ensemble_models_comparison.png')
        print("Ensemble comparison visualization complete.")
    else:
        print("\nSkipping Step 7: Ensemble comparison visualization (missing data).")

    # 8. Compare model metrics
    print("\nStep 8: Visualizing metrics comparison...")
    visualize_metrics_comparison(model_metrics, ensemble_metrics_dict, filename='hiv_model_metrics_comparison.png')
    print("Metrics comparison visualization complete.")

    # 9. Create validation plot
    if best_model_name and best_ensemble_model_name and \
       best_model_name in fitted_models and best_ensemble_model_name in ensemble_models_dict:
        print("\nStep 9: Creating validation plot...")
        create_validation_plot(
            df_weekly, X, y,
            fitted_models[best_model_name],
            ensemble_models_dict[best_ensemble_model_name],
            best_ensemble_model_name, # ensemble_name for the plot title/label
            best_model_name,
            filename='hiv_model_validation_plot.png'
        )
        print("Validation plot creation complete.")
    else:
        print("\nSkipping Step 9: Validation plot (missing best models or their details).")

    # 10. Forecast future trends
    if best_model_name and best_ensemble_model_name and \
       best_model_name in fitted_models and best_ensemble_model_name in ensemble_models_dict:
        print("\nStep 10: Forecasting future trends...")
        forecast_future_trends(
            df_weekly, X, y,
            fitted_models[best_model_name],
            ensemble_models_dict[best_ensemble_model_name],
            best_ensemble_model_name, # ensemble_name for the plot
            fitted_models, # Pass all fitted_models
            best_model_name,
            generate_bootstrap_predictions, # Pass the actual function
            filename='hiv_future_forecast_plot.png'
        )
        print("Future trends forecasting and plot generation complete.")
    else:
        print("\nSkipping Step 10: Future trends forecast (missing best models or their details).")

    print("\n------------------------------------")
    print("Analysis complete.")
    print("All visualizations saved to 'plots/' directory.")
    print(f"Forecast data potentially saved to 'data/forecast_results_next_5_years.csv'")
    print("------------------------------------")

if __name__ == "__main__":
    main()
