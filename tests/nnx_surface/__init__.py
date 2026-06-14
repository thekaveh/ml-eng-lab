"""NNx-surface contract tests.

This package contains tiny, fast tests that pin the *call shapes* and
data-contract of the small subset of nnx primitives that the task notebooks
depend on (`NNModel`, `NNParams`, `NNTabularDataset`, etc.). They exist to
catch `thekaveh-nnx` version upgrades that would silently break the notebooks.

Coverage is intentionally narrow: only tasks whose notebooks exercise a
distinct nnx call-shape get a sibling test module here. Tasks that use nnx
only via `set_seed` / `Devices.CPU` are out of scope — the Tier-A papermill
job in CI is the operative end-to-end check for those.
"""
