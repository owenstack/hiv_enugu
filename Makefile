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

## Run the full analysis pipeline
.PHONY: run-analysis
run-analysis: requirements
	$(PYTHON_INTERPRETER) run_analysis.py

# The following targets point to older, potentially superseded scripts.
# Kept for reference or if specific parts of the old pipeline are needed.
## Run the original training script
#.PHONY: train
#train: requirements
#	$(PYTHON_INTERPRETER) hiv_enugu/modeling/train.py

## Run the original prediction script
#.PHONY: predict
#predict: requirements
#	$(PYTHON_INTERPRETER) hiv_enugu/modeling/predict.py


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
