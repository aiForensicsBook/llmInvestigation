.PHONY: help install clean test train api cli lint format docs

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make clean      - Clean generated files"
	@echo "  make test       - Run tests"
	@echo "  make train      - Train the model"
	@echo "  make api        - Start the API server"
	@echo "  make cli        - Run CLI in interactive mode"
	@echo "  make lint       - Run code linting"
	@echo "  make format     - Format code with black"
	@echo "  make docs       - Build documentation"

install:
	pip install -r requirements.txt
	pip install -e .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/
	rm -rf models/*.pkl
	rm -rf data/synthetic/*.json

test:
	pytest tests/ -v --cov=src

train:
	python -m src.train --evaluate

api:
	cd src && uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

cli:
	python -m src.cli.cli interactive

lint:
	flake8 src/ --max-line-length=100
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ --line-length=100

docs:
	mkdocs build