# Notebook re-execution targets, organized by execution-cost tier.
#
# Tier A: cheap (<5 min), re-executed in place (refreshes outputs).
# Tier B: moderate, smoke-runs to /tmp (preserves original outputs).
# Tier C: expensive, smoke-runs via SMOKE_TEST parameter to /tmp.
#
# Tier A is what CI runs on every PR. B and C are on-demand
# (`make smoke-tier-b`, `make smoke-tier-c`).
#
# All targets assume papermill is on PATH and the notebooks' kernel
# can import nnx (i.e., you ran scripts/setup-in-jupyter.sh in the
# target environment, or have `pip install -e ./nnx` locally).

TIER_A := \
    image_classification-mnist-ffnn-numpy/notebook.ipynb \
    image_classification-mnist-ffnn-pytorch/notebook.ipynb \
    node_classification-reddit-gnn-pyg/phase1-dataset-exploration-notebook.ipynb \
    tabular_classification-iris-mlp-pytorch/notebook.ipynb \
    model_surgery-mnist-ffnn-pytorch/notebook.ipynb \
    quantization-mnist-ffnn-pytorch/notebook.ipynb

TIER_B := \
    node_classification-reddit-gnn-pyg/phase2-model-selection-notebook1.ipynb \
    node_classification-reddit-gnn-pyg/phase2-model-selection-notebook2.ipynb \
    node_classification-reddit-gnn-pyg/phase2-model-selection-notebook3.ipynb \
    node_classification-reddit-gnn-pyg/phase2-model-selection-notebook4.ipynb

TIER_C := \
    node_classification-reddit-gnn-pyg/phase3-main-model-training-and-eval-notebook.ipynb \
    node_classification-reddit-gnn-pyg/phase3-main-model-training-and-eval-notebook2.ipynb \
    node_classification-reddit-gnn-pyg/phase3-main-model-training-and-eval-notebook3.ipynb \
    node_classification-reddit-gnn-pyg/phase3-main-model-training-and-eval-notebook4.ipynb

SMOKE_OUT := /tmp/ml-smoke

.PHONY: help run-tier-a smoke-tier-b smoke-tier-c test verify

help:
	@echo "Targets:"
	@echo "  run-tier-a    Re-execute Tier-A notebooks in place. CI runs this on every PR."
	@echo "  smoke-tier-b  Papermill Tier-B notebooks to $(SMOKE_OUT)/ (preserves source outputs)."
	@echo "  smoke-tier-c  Papermill Tier-C notebooks with SMOKE_TEST=1 to $(SMOKE_OUT)/."
	@echo "  test          Run pytest on tests/ directory."
	@echo "  verify        Run repo verifier (scripts/verify_repo.py --check all --fast)."

run-tier-a:
	@for nb in $(TIER_A); do \
		echo "==> $$nb"; \
		dir=$$(dirname "$$nb"); base=$$(basename "$$nb"); \
		(cd "$$dir" && papermill --kernel python3 "$$base" "$$base") || exit 1; \
	done

smoke-tier-b:
	@mkdir -p $(SMOKE_OUT)
	@for nb in $(TIER_B); do \
		out=$(SMOKE_OUT)/$$(basename "$$nb"); \
		echo "==> $$nb -> $$out"; \
		dir=$$(dirname "$$nb"); base=$$(basename "$$nb"); \
		(cd "$$dir" && papermill --kernel python3 "$$base" "$$out") || exit 1; \
	done

smoke-tier-c:
	@mkdir -p $(SMOKE_OUT)
	@for nb in $(TIER_C); do \
		out=$(SMOKE_OUT)/$$(basename "$$nb"); \
		echo "==> $$nb -> $$out"; \
		dir=$$(dirname "$$nb"); base=$$(basename "$$nb"); \
		(cd "$$dir" && papermill --kernel python3 -p SMOKE_TEST 1 "$$base" "$$out") || exit 1; \
	done

test:
	pytest tests/ -v

verify:
	python scripts/verify_repo.py --check all --fast
