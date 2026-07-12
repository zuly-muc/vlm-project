IMAGE ?= vlm-project
NUSCENES_DIR ?= ./data

.PHONY: help install test lint docker-build ingest query clean

help:
	@echo "install          pip install the package with dev extras"
	@echo "test             run unit tests (no dataset/model needed)"
	@echo "test-int         run the opt-in BLIP integration test"
	@echo "lint             ruff check"
	@echo "docker-build     build the container (bakes BLIP weights)"
	@echo "ingest           run ingest in the container over NUSCENES_DIR"
	@echo "query Q=...      full-text search the produced DB"
	@echo "production-test  full real-data acceptance (build+deps+nuScenes+ingest+verify)"
	@echo "assets           regenerate the README caption montage from out/scenes.db"

install:
	pip install -e ".[dev]"

test:
	pytest -m "not integration"

test-int:
	pytest -m integration

lint:
	ruff check src tests

docker-build:
	docker build -t $(IMAGE) .

ingest:
	docker run --rm -v "$(NUSCENES_DIR):/data:ro" -v "$(PWD)/out:/out" \
		-e DATAROOT=/data -e DB=/out/scenes.db $(IMAGE) ingest -v

query:
	docker run --rm -v "$(PWD)/out:/out" -e DB=/out/scenes.db $(IMAGE) query "$(Q)"

production-test:
	bash scripts/test_vlm_app_production.sh

assets:
	python scripts/make_readme_assets.py --db out/scenes.db

clean:
	rm -rf out *.db descriptions.json .pytest_cache .ruff_cache
