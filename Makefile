# Notebook re-execution targets, organized by execution-cost tier.
#
# Tier A: cheap (<5 min), re-executed in place (refreshes outputs).
# Tier B: moderate, smoke-runs to /tmp (preserves original outputs).
# Tier C: expensive, smoke-runs via SMOKE_TEST parameter to /tmp.
#
# Tier A is what CI runs on every PR. B/C smoke targets can run locally or via
# workflow_dispatch; CI also runs both on the weekly schedule and Tier B on PRs
# labeled `tier-b-smoke`.
#
# All targets assume papermill is on PATH and the notebooks' kernel can
# import nnx. nnx is consumed from PyPI via the `thekaveh-nnx[lm]==0.2.0`
# pin in requirements.txt (as of 2026-06-14). The `[lm]` extra pulls
# tokenizers+datasets for the two notebooks that call train_bpe /
# NNTokenizerParams (text_generation-tinyshakespeare-... and
# preference_alignment-toy-dpo-...) — issue #12. Without it those
# notebooks ImportError at the first tokenizer call.

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
    node_classification-reddit-gnn-pyg/phase2-model-selection-notebook1.ipynb \
    node_classification-reddit-gnn-pyg/phase2-model-selection-notebook2.ipynb \
    node_classification-reddit-gnn-pyg/phase2-model-selection-notebook3.ipynb \
    node_classification-reddit-gnn-pyg/phase2-model-selection-notebook4.ipynb

# quantization-mnist-ffnn-pytorch/notebook.ipynb was previously the 2nd entry
# above. Removed 2026-06-16 after the weekly smoke-tier-b cron failed at the
# quantization import: `torchao>=0.17` (requirements.txt pin, smallest version
# exposing nnx.quantize_int8's `Int8WeightOnlyConfig` API) references
# `torch.int1` at module load; `torch.int1` was added in torch 2.5; ml-lab
# pins `torch==2.4.1` for genai-vanilla image-parity (see torch-core-requirements.txt
# + issue #10). No torchao version satisfies both nnx's API requirement AND
# the torch 2.4.1 import surface, so the notebook cannot execute under
# CI's pinned environment. Notebook stays in the repo as a manual-only task
# (run locally under a `torch>=2.5` env). The Tier-B move (PR #11) was made
# under the assumption the weekly cron would still exercise it — that turned
# out to be wrong; removing it here unblocks the 5 remaining Tier-B notebooks
# the cron was supposed to cover.

TIER_C := \
    node_classification-reddit-gnn-pyg/phase3-main-model-training-and-eval-notebook.ipynb \
    node_classification-reddit-gnn-pyg/phase3-main-model-training-and-eval-notebook2.ipynb \
    node_classification-reddit-gnn-pyg/phase3-main-model-training-and-eval-notebook3.ipynb \
    node_classification-reddit-gnn-pyg/phase3-main-model-training-and-eval-notebook4.ipynb

SMOKE_OUT := /tmp/ml-smoke

.PHONY: help run-tier-a check-tier-a-clean smoke-tier-b smoke-tier-c test test-nnx-surface lint nlp-assets verify install-torch-stack codespace-setup

help:
	@echo "Targets:"
	@echo "  run-tier-a        Re-execute Tier-A notebooks in place. CI runs this on every PR."
	@echo "  check-tier-a-clean Fail if Tier-A notebook execution changed tracked outputs."
	@echo "  smoke-tier-b      Papermill Tier-B notebooks with SMOKE_TEST=1 to $(SMOKE_OUT)/ (preserves source outputs)."
	@echo "  smoke-tier-c      Papermill Tier-C notebooks with SMOKE_TEST=1 to $(SMOKE_OUT)/."
	@echo "  test              Run pytest on tests/ directory."
	@echo "  test-nnx-surface  Run only tests/nnx_surface (matches the CI pytest-nnx-surface job)."
	@echo "  lint              Run ruff check . using the [tool.ruff] config in pyproject.toml."
	@echo "  nlp-assets        Download spaCy en_core_web_sm + NLTK vader_lexicon (needed by the 2 NLP Tier-A notebooks)."
	@echo "  verify            Run repo verifier (scripts/verify_repo.py --check all --fast)."
	@echo "  install-torch-stack Install pinned Torch core first, then PyG/runtime deps."
	@echo "  codespace-setup   Full dep install + NLP assets. Invoked by .devcontainer/devcontainer.json's postCreateCommand."

run-tier-a:
	@for nb in $(TIER_A); do \
		echo "==> $$nb"; \
		dir=$$(dirname "$$nb"); base=$$(basename "$$nb"); \
		(cd "$$dir" && papermill --kernel python3 "$$base" "$$base") || exit 1; \
	done

check-tier-a-clean:
	git diff --exit-code -- $(TIER_A)

smoke-tier-b:
	@mkdir -p $(SMOKE_OUT)
	@for nb in $(TIER_B); do \
		out=$(SMOKE_OUT)/$$(basename "$$nb"); \
		echo "==> $$nb -> $$out"; \
		dir=$$(dirname "$$nb"); base=$$(basename "$$nb"); \
		(cd "$$dir" && papermill --kernel python3 -p SMOKE_TEST 1 "$$base" "$$out") || exit 1; \
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

install-torch-stack:
	pip install --upgrade pip
	pip install -r torch-core-requirements.txt
	pip install --no-build-isolation -r torch-requirements.txt

# Full one-shot dep install for the GitHub Codespaces / "Reopen in Container"
# path (README §3.4). Reuses the same Torch-first install order as CI and
# Docker so PyG source-build fallback can import torch during extension builds.
# Recursively invokes nlp-assets so the spaCy + NLTK download steps stay in
# one place across the §3.2 (Docker), §3.3 (venv), and §3.4 (Codespaces) paths.
codespace-setup: install-torch-stack
	pip install -r requirements.txt
	$(MAKE) nlp-assets
