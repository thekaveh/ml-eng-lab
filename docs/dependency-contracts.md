# Dependency Contracts

This ledger records consumed dependency contracts that are intentionally pinned,
manual-only, or known to carry security/tooling constraints. It complements
`requirements.txt`, `torch-requirements.txt`, and the CI workflow; the manifests
remain the source of truth for installation.

## 1. Audit Snapshot

Last reviewed: 2026-07-02, on branch `codex/overnight-maintenance`.

Command:

```bash
pip-audit -r requirements.txt -r torch-requirements.txt
```

Result: 23 known vulnerabilities across three pinned packages:

| Package | Pinned Version | Finding Count | Current Disposition |
| --- | ---: | ---: | --- |
| `torch` | `2.4.1` | 21 | Accepted temporarily for genai-vanilla image parity; upgrade requires a coordinated PyTorch/PyG/torchao compatibility pass. |
| `pytorch-lightning` | `2.4.0` | 1 | Accepted temporarily because it is pinned to the current Torch stack; revisit with the Torch upgrade. |
| `nltk` | `3.9.4` | 1 | Review on the next dependency bump; VADER usage is local/offline and does not deserialize untrusted corpus files. |

The audit must be re-run after any dependency pin change. New unreviewed audit
findings are maintenance issues until either fixed or added here with rationale.

## 2. Torch Stack Pin

`torch-requirements.txt` pins:

- `torch==2.4.1`
- `torchvision==0.19.1`
- `torchaudio==2.4.1`
- `torchmetrics==1.4.2`
- PyG wheels resolved from `https://data.pyg.org/whl/torch-2.4.0+cpu.html`

Reason: these versions match the genai-vanilla JupyterHub image lineage used by
the documented runtime paths.

Upgrade criteria:

1. Select a Torch version with matching `torchvision`, `torchaudio`, and PyG CPU
   wheels.
2. Confirm `torchao>=0.17` imports under that Torch version.
3. Re-run `make test`, `make verify`, `make test-nnx-surface`, and at least the
   smoke Tier-B/Tier-C notebooks on Linux.
4. Update README, environment docs, and this ledger in the same change.

## 3. Manual-Only Quantization Notebook

`quantization-mnist-ffnn-pytorch/notebook.ipynb` depends on `torchao>=0.17`.
That torchao API references `torch.int1` at import time, which is unavailable in
the pinned `torch==2.4.1` environment. The notebook remains an active task but is
manual-only until the Torch stack is upgraded.

Expected local environment for this notebook:

- `torch>=2.5`
- `torchao>=0.17`

Do not add the quantization notebook back to `Makefile` Tier-A/B/C until the
repository-wide Torch stack supports it.

## 4. External Assets

`make nlp-assets` downloads:

- spaCy `en_core_web_sm`
- NLTK `vader_lexicon`

These assets are consumed by the text-classification and sentiment notebooks.
They are not locked by checksum today. If reproducibility becomes stricter than
the current educational-notebook standard, add a lock/verification mechanism and
update this section.
