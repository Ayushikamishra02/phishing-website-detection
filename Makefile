.PHONY: help setup data train train-sample test lint app api scan notebook docker clean

PY ?= python

help:          ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

setup:         ## install dependencies + package (editable)
	$(PY) -m pip install -r requirements-dev.txt
	$(PY) -m pip install -e .

data:          ## download the full dataset (25 MB) into data/raw/
	$(PY) -c "from phishguard.data import download; download()"

train:         ## all experiments + final models (~10 min)
	$(PY) -m phishguard.train

train-sample:  ## quick run on the bundled 5k sample (~1 min)
	$(PY) -m phishguard.train --sample

test:          ## run tests
	$(PY) -m pytest

lint:          ## static checks
	ruff check src api app tests

scan:          ## scan a URL from the command line: make scan URL=example.com
	$(PY) -m phishguard.predict "$(URL)"

app:           ## Streamlit dashboard on :8501
	streamlit run app/streamlit_app.py

api:           ## FastAPI on :8000 (docs at /docs)
	uvicorn api.main:app --reload --port 8000

notebook:      ## re-execute the analysis notebook
	jupyter nbconvert --to notebook --execute --inplace notebooks/phishing_analysis.ipynb

docker:        ## API + dashboard via docker compose
	docker compose up --build

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
