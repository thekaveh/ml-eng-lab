# Notebook re-execution targets, organized by execution-cost tier.
#
# Tier A: cheap (<5 min), re-executed in place (refreshes outputs).
# Tier B: moderate, smoke-runs to /tmp (preserves original outputs).
# Tier C: expensive, smoke-runs via SMOKE_TEST parameter to /tmp.
#
# Tier A is what CI runs on every PR. B and C are on-demand
# (`make smoke-tier-b`, `make smoke-tier-c`).
#
# All targets assume papermill is on PATH and the notebooks' kernel can
# import nnx. nnx is shipped natively in the genai-vanilla jupyterhub image
# as of cbad341 (PR #26, 2026-06-02); locally, `pip install 'nnx-pytorch[lm]'`
# (or `pip install -e './nnx[lm]'` for the editable-dev workflow) gets the
# same. The `[lm]` extra pulls tokenizers+datasets for the two notebooks
# that call train_bpe/NNTokenizerParams (text_generation-tinyshakespeare-...
# and preference_alignment-toy-dpo-...) — issue #12. Without it those
# notebooks ImportError at the first tokenizer call.
# scripts/setup-in-jupyter.sh is only needed for the editable-install
# developer override inside the wrapper-and-bind-mount path.

TIER_A := \
    image_classification-mnist-ffnn-numpy/notebook.ipynb \
    node_classification-reddit-gnn-pyg/phase1-dataset-exploration-notebook.ipynb \
    tabular_classification-iris-mlp-pytorch/notebook.ipynb \
    model_surgery-mnist-ffnn-pytorch/notebook.ipynb \
    pruning-mnist-ffnn-pytorch/notebook.ipynb \
    knowledge_distillation-mnist-ffnn-pytorch/notebook.ipynb \
    text_generation-tinyshakespeare-transformer-pytorch/notebook.ipynb \
    peft-mnist-to-fmnist-dora-vs-lora-pytorch/notebook.ipynb \
    dim_reduction-iris-autoencoder-pytorch/notebook.ipynb \
    tabular_regression-diabetes-mlp-pytorch/notebook.ipynb \
    diffusion-mnist-ddpm-pytorch/notebook.ipynb \
    moe-fmnist-mixture-of-experts-pytorch/notebook.ipynb \
    clustering-iris-kmeans-vs-ae-pytorch/notebook.ipynb \
    link_prediction-karate-graphsage-pyg/notebook.ipynb \
    community_detection-karate-louvain-vs-gnn-pyg/notebook.ipynb \
    text_classification-agnews-spacy-mlp-pytorch/notebook.ipynb \
    sentiment_classification-vader-mlp-pytorch/notebook.ipynb \
    preference_alignment-toy-dpo-pytorch/notebook.ipynb \
    self_supervised-fmnist-jepa-pytorch/notebook.ipynb

TIER_B := \
    image_classification-mnist-ffnn-pytorch/notebook.ipynb \
    quantization-mnist-ffnn-pytorch/notebook.ipynb \
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

.PHONY: help run-tier-a smoke-tier-b smoke-tier-c test test-nnx-surface lint nlp-assets verify

help:
	@echo "Targets:"
	@echo "  run-tier-a        Re-execute Tier-A notebooks in place. CI runs this on every PR."
	@echo "  smoke-tier-b      Papermill Tier-B notebooks to $(SMOKE_OUT)/ (preserves source outputs)."
	@echo "  smoke-tier-c      Papermill Tier-C notebooks with SMOKE_TEST=1 to $(SMOKE_OUT)/."
	@echo "  test              Run pytest on tests/ directory."
	@echo "  test-nnx-surface  Run only tests/nnx_surface (matches the CI pytest-nnx-surface job)."
	@echo "  lint              Run ruff check . using the [tool.ruff] config in pyproject.toml."
	@echo "  nlp-assets        Download spaCy en_core_web_sm + NLTK vader_lexicon (needed by the 2 NLP Tier-A notebooks)."
	@echo "  verify            Run repo verifier (scripts/verify_repo.py --check all --fast)."

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

test-nnx-surface:
	pytest tests/nnx_surface -v

lint:
	ruff check .

nlp-assets:
	python -m spacy download en_core_web_sm
	python -c "import nltk; nltk.download('vader_lexicon', quiet=True)"

verify:
	python scripts/verify_repo.py --check all --fast
