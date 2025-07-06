# hiv_enugu

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https.img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project analyzes HIV case data from Enugu State, Nigeria, using various growth models (exponential, logistic, Richards, Gompertz) and ensemble techniques (Simple Average, Weighted Average, Random Forest, Gradient Boosting) to predict and forecast HIV cases. It includes a comprehensive pipeline for data processing, model training, evaluation, and visualization.

## Running the Analysis

The primary way to run the full analysis pipeline is using the `run_analysis.py` script located in the root directory. This script will:
1. Load and preprocess data from `data/cleaned_enrollments.csv`.
2. Fit individual growth models and ensemble models.
3. Evaluate models using cross-validation.
4. Generate and save various plots (e.g., model fits, metric comparisons, forecasts) to the `plots/` directory.
5. Save trained machine learning models (Random Forest, Gradient Boosting) and scalers to the `saved_models/` directory.
6. Save forecast data to `data/forecast_results_next_5_years.csv`.

You can run the analysis using Make:
```bash
make run-analysis
```
This requires that you have set up the environment and installed dependencies (e.g., via `make requirements` or `uv pip install -r requirements.txt`).

Alternatively, you can run the script directly:
```bash
python run_analysis.py
```

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         hiv_enugu and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── hiv_enugu   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes hiv_enugu a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

