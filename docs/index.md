<p align="center">
  <img src="assets/ml-eng-lab-poster.png" alt="ML Eng Lab — notebooks, systems, and reproducibility" width="100%">
</p>

<h1 align="center">1 · ML ENG LAB</h1>

<p align="center"><strong>Local notebooks. Remote Atlas execution. Explicit infrastructure contracts.</strong></p>

<p align="center">
  <sub><strong>Core ML</strong></sub><br>
  <img alt="Python" src="assets/badges/python.svg"> <img alt="Jupyter" src="assets/badges/jupyter.svg"> <img alt="NumPy" src="assets/badges/numpy.svg"> <img alt="pandas" src="assets/badges/pandas.svg"> <img alt="PyTorch" src="assets/badges/pytorch.svg"> <img alt="PyTorch Geometric" src="assets/badges/pytorch-geometric.svg"> <img alt="scikit-learn" src="assets/badges/scikit-learn.svg">
</p>

<p align="center">
  <sub><strong>NLP and graphs</strong></sub><br>
  <img alt="spaCy" src="assets/badges/spacy.svg"> <img alt="NLTK" src="assets/badges/nltk.svg"> <img alt="NetworkX" src="assets/badges/networkx.svg">
</p>

<p align="center">
  <sub><strong>Runtime</strong></sub><br>
  <img alt="Atlas" src="assets/badges/atlas.svg"> <img alt="Docker" src="assets/badges/docker.svg"> <img alt="VS Code" src="assets/badges/vscode.svg"> <img alt="GitHub Codespaces" src="assets/badges/github-codespaces.svg">
</p>

<p align="center">
  <sub><strong>Engineering</strong></sub><br>
  <img alt="NNx" src="assets/badges/nnx.svg"> <img alt="Papermill" src="assets/badges/papermill.svg"> <img alt="pytest" src="assets/badges/pytest.svg"> <img alt="Ruff" src="assets/badges/ruff.svg"> <img alt="GitHub Actions" src="assets/badges/github-actions.svg">
</p>

<!-- project-summary:start -->
ml-eng-lab is a portfolio of machine-learning notebook experiments. Explore
classification, graphs, language models and other techniques through narrative
notebooks with committed results. The collection contains 21 active task folders
and 29 active notebooks; older experiments are kept separately in a read-only archive.

Use Atlas JupyterHub with local VS Code, or choose a supported local environment.
Each task declares its runtime requirements. Start with the small Iris
classification notebook, then use the task catalog and deeper guides below.
<!-- project-summary:end -->

## 1.1 Start with a notebook

Choose a runtime in [Environment setup](env-setup.md), including its tools,
platform and resource requirements. Use the
[Iris classification walkthrough](notebooks/tabular_classification-iris-mlp-pytorch.md)
as the first small CPU example. Open its notebook in your configured environment,
select the kernel and run cells from top to bottom. Compare the metrics,
confusion matrices and final verdict under **Evaluation & Results**.

A rerun trains models and can replace displayed results when saved. Preserve
committed outputs in a working copy if needed. Read the
[notebook output policy](conventions.md#524-notebook-output-freshness) before
updating committed artifacts. The active deep-dives appear in section 8.

## 1.2 Deeper guides

- [System & context view](architecture.md): runtime and repository architecture.
- [Atlas pin-bump and service-admission runbook](atlas-pin-bump-runbook.md): infrastructure ownership, native Ollama and future service admission.
- [NNx library](nnx-library.md): current package contract and upstream development.
- [Repository conventions](conventions.md): contributions, execution tiers, repository map and planned work.
