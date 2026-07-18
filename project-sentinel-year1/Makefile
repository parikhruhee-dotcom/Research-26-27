.PHONY: setup data atlas aggregation md target design validate biosensor bench figures test report all clean

PYTHON ?= python3

setup:
	$(PYTHON) -m pip install -e .[dev]
	$(PYTHON) -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install fair-esm
	test -f /tmp/ProteinMPNN/protein_mpnn_run.py || \
		git clone --depth 1 https://github.com/dauparas/ProteinMPNN.git /tmp/ProteinMPNN

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
	$(PYTHON) -m sentinel.design.colab_notebooks
	$(PYTHON) -m sentinel.design.gpu_tier_status

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
