# Makefile
SHELL = /bin/bash

.PHONY: help
help:
	@echo "Commands:"
	@echo "venv    : creates a virtual environment."
	@echo "style   : executes style formatting."
	@echo "clean   : cleans all unnecessary files."
	@echo "test    : execute tests on code, data and models."

# Styling
.PHONY: style
style:
	black .
	flake8
	python3 -m isort .

# Environment
.ONESHELL:
venv:
	source .venv/bin/activate && \
	python3 -m pip install --upgrade pip setuptools wheel && \
	python3 -m pip install -e ".[dev]" && \
	pre-commit install && \
	pre-commit autoupdate

# Cleaning
.PHONY: clean
clean: style
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . | grep -E ".trash" | xargs rm -rf
	rm -f .coverage

# Test
.PHONY: test
test:
	pytest -m "not training"

# Parser data
.ONESHELL:
preprocessing:
	python logparser/data.py --window-size=5 --step-size=1 --train-ratio=0.4 && \
	python logparser/logbert.py create-vocab

.PHONY: train
train:
	python logparser/logbert.py train-model

.PHONY: mlflow
mlflow:
	mlflow server -h 0.0.0.0 -p 8000 --backend-store-uri ./stores/model/
