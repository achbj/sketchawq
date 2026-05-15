.PHONY: test lint smoke list

test:
	pytest

lint:
	ruff check .

smoke:
	PYTHONPATH=src python -m sketchawq.cli run configs/smoke/tiny.yaml

list:
	PYTHONPATH=src python -m sketchawq.cli list-targets
