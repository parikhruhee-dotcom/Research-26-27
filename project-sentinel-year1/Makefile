.PHONY: setup data atlas aggregation md target design validate biosensor bench figures test report all clean

PYTHON ?= python3

setup:
	$(PYTHON) -m pip install -e .[dev]

data:
	$(PYTHON) -m sentinel.io.fetch_sequence
	$(PYTHON) -m sentinel.io.fetch_structures
	$(PYTHON) -m sentinel.io.fetch_known_inhibitors
	$(PYTHON) -m sentinel.io.prepare_structures

atlas:
	$(PYTHON) -m sentinel.atlas.run_atlas

aggregation:
	$(PYTHON) -m sentinel.aggregation.run_aggregation

md:
	$(PYTHON) -m sentinel.md.run_md

target:
	$(PYTHON) -m sentinel.target.build_target

design:
	$(PYTHON) -m sentinel.design.run_design_loop

validate:
	$(PYTHON) -m sentinel.validate.run_validation

biosensor:
	$(PYTHON) -m sentinel.biosensor.run_biosensor

bench:
	$(PYTHON) -m sentinel.bench.run_benchmarks

figures:
	$(PYTHON) -m sentinel.figures.make_all_figures

test:
	$(PYTHON) -m pytest tests/ -v

report:
	$(PYTHON) -m sentinel.utils.build_report_stats

all: setup data atlas aggregation md target design validate biosensor bench figures test report
	@echo "Project SENTINEL Year 1 pipeline complete."

clean:
	rm -rf results/* figures/* data/interim/* data/raw/* .pytest_cache
