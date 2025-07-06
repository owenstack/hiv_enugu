#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = hiv_enugu
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"




#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset (Data processing is also part of run-analysis)
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) hiv_enugu/data_processing.py

## Run the full analysis pipeline (training and analysis mode)
.PHONY: run-analysis
run-analysis: requirements
	$(PYTHON_INTERPRETER) hiv_enugu/analysis_pipeline.py

# The functionality of the old train.py and predict.py scripts is now consolidated
# into hiv_enugu/analysis_pipeline.py, which can be run with different modes.
# The `run-analysis` target above runs the default "train_and_analyze" mode.
#
# If you need to run only prediction, you would modify analysis_pipeline.py
# to accept command-line arguments for the mode, or call it like:
# $(PYTHON_INTERPRETER) -c "from hiv_enugu.analysis_pipeline import main_pipeline; main_pipeline(mode='predict', data_file='your_prediction_input.csv')"
#
# Example commented-out targets for specific modes (would require analysis_pipeline.py to parse sys.argv or similar):
# ## Run only the training part of the consolidated pipeline
# .PHONY: train-pipeline
# train-pipeline: requirements
#	$(PYTHON_INTERPRETER) hiv_enugu/analysis_pipeline.py --mode train_and_analyze # Assuming default or explicit mode
#
# ## Run only the prediction part of the consolidated pipeline
# .PHONY: predict-pipeline
# predict-pipeline: requirements
#	$(PYTHON_INTERPRETER) hiv_enugu/analysis_pipeline.py --mode predict --data-file name_of_file_for_prediction.csv


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
