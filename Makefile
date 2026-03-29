.PHONY: install install-dev lint format test train grid-search app docker-build docker-run

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

test:
	pytest

train:
	snake-train --episodes 10000 --model-path artifacts/best_model.pkl

grid-search:
	python -m snake_rl.grid_search --output-dir artifacts/grid_results --episodes 50000

app:
	streamlit run app/streamlit_app.py

docker-build:
	docker build -t snake-rl .

docker-run:
	docker run -p 8501:8501 snake-rl
