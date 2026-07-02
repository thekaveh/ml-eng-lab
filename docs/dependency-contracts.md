# Dependency Contracts

This ledger records consumed dependency contracts that are intentionally pinned,
manual-only, or known to carry security/tooling constraints. It complements
`requirements.txt`, `torch-core-requirements.txt`, `torch-requirements.txt`,
and the CI workflow; the manifests remain the source of truth for installation.

## 1. Audit Snapshot

Last reviewed: 2026-07-02, on branch `codex/overnight-maintenance`.

Command:

```bash
pip-audit -r requirements.txt -r torch-requirements.txt
```

Result: 23 known vulnerabilities across three resolved packages:

| Package | Manifest Constraint | Audited Resolved Version | Finding Count | Current Disposition |
| --- | --- | ---: | ---: | --- |
| `torch` | `torch==2.4.1` | `2.4.1` | 21 | Accepted temporarily for genai-vanilla image parity; upgrade requires a coordinated PyTorch/PyG/torchao compatibility pass. |
| `pytorch-lightning` | `pytorch-lightning==2.4.0` | `2.4.0` | 1 | Accepted temporarily because it is pinned to the current Torch stack; revisit with the Torch upgrade. |
| `nltk` | `nltk>=3.9.3` | `3.9.4` | 1 | Review on the next dependency bump; VADER usage is local/offline and does not deserialize untrusted corpus files. |

The audit must be re-run after any dependency pin change. New unreviewed audit
findings are maintenance issues until either fixed or added here with rationale.
Because several manifest entries are intentionally ranged or floating today,
the audited resolved versions and advisory IDs below are the accepted state, not
just the package-level counts.

Accepted advisory IDs from the 2026-07-02 audit:

| Package | Advisory ID | Fix Versions |
| --- | --- | --- |
| `torch` | `PYSEC-2025-191` | `2.7.1rc1` / none listed |
| `torch` | `PYSEC-2025-41` | `2.6.0` |
| `torch` | `PYSEC-2024-259` | `2.5.0` |
| `torch` | `PYSEC-2025-205` | `2.7.1` |
| `torch` | `PYSEC-2025-206` | `2.9.0` |
| `torch` | `PYSEC-2025-207` | `2.7.1` |
| `torch` | `PYSEC-2025-204` | `2.9.0` |
| `torch` | `PYSEC-2026-139` | none listed |
| `torch` | `PYSEC-2025-209` | `2.7.1` |
| `torch` | `PYSEC-2025-208` | `2.7.1` |
| `torch` | `PYSEC-2025-198` | `2.7.0` |
| `torch` | `PYSEC-2025-203` | `2.9.0` |
| `torch` | `CVE-2025-3730` | `2.8.0` |
| `torch` | `CVE-2025-2148` | none listed |
| `torch` | `CVE-2025-2149` | none listed |
| `torch` | `CVE-2025-2998` | none listed |
| `torch` | `CVE-2025-2999` | `2.9.1` |
| `torch` | `CVE-2025-3000` | none listed |
| `torch` | `CVE-2025-3001` | `2.10.0` |
| `pytorch-lightning` | `CVE-2026-31221` | none listed |
| `nltk` | `PYSEC-2026-597` | none listed |

## 2. Torch Stack Pin

`torch-core-requirements.txt` pins the core Torch stack:

- `torch==2.4.1`
- `pytorch-lightning==2.4.0`
- `torchvision==0.19.1`
- `torchaudio==2.4.1`
- `torchmetrics==1.4.2`

`torch-requirements.txt` includes `torch-core-requirements.txt` and then pins:

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

## 5. Deferred Reproducibility Hardening

The current manifests still include floating and ranged Python dependencies, and
the Docker/devcontainer bases are tag-pinned rather than digest-pinned. A full
lockfile, CI install against that lock, `pip-audit` comparison against accepted
advisory IDs, and base-image digest pinning are intentionally deferred to a
coordinated dependency-refresh pass because they can change every notebook
runtime at once.
