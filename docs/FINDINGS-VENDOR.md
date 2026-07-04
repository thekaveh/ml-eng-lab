# Vendor findings

Issues found by the verify_repo.py loop in vendored dependencies under
`vendor/`. These are NOT fixed by this loop (per spec §1.3); they are
surfaced for upstream contribution.

## 1. genai-vanilla

### 1.1. Stale ml-lab comments in JupyterHub runtime files

Status: deferred upstream documentation cleanup.

The current `vendor/genai-vanilla` pin still has comments in
`services/jupyterhub/build/requirements.txt` and
`services/jupyterhub/build/Dockerfile` that use the old `ml-lab` repository name
and URL. The URL redirects to `ml-eng-lab`, and the actual runtime contract is
current, so this maintenance branch records the issue instead of patching
vendored source directly.

See [dependency-contracts.md §7](dependency-contracts.md#7-genai-vanilla-submodule-contract)
for the consumed-contract ledger entry.
