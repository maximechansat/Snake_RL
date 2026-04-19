.PHONY: install install-dev lint format test train grid-search app docker-build docker-run

S3_ENDPOINT_URL ?= https://minio.lab.sspcloud.fr
S3_ARTIFACTS_URI ?= s3://mchansat/snake-rl/artifacts/grid_results
LOCAL_ARTIFACTS_DIR ?= artifacts/grid_results

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
	python -m snake_rl.grid_search --output-dir $(LOCAL_ARTIFACTS_DIR) --episodes 50000

upload-artifacts:
	python -c "from pathlib import Path; import os, s3fs; local=Path('$(LOCAL_ARTIFACTS_DIR)'); remote='$(S3_ARTIFACTS_URI)'; fs=s3fs.S3FileSystem(client_kwargs={'endpoint_url': os.environ.get('S3_ENDPOINT_URL', '$(S3_ENDPOINT_URL)')}); [fs.put_file(str(p), f'{remote}/{p.relative_to(local).as_posix()}') for p in local.rglob('*') if p.is_file()]; print('upload done')"

train-grid-upload: grid-search upload-artifacts

app:
	streamlit run app/streamlit_app.py

docker-build:
	docker build -t snake-rl .

docker-run:
	docker run -p 8501:8501 snake-rl
