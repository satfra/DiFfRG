# Example CI baselines

This directory stores small, normalized text baselines for example regression
runs. The intended CI flow is:

1. Run the Wolfram generator where available.
2. Rebuild the example CMake project against the generated sources.
3. Run the example with short CI timestepping overrides.
4. Normalize text outputs and compare them with the baseline in this directory.

Baselines are created or refreshed by running `containers/ci/run-example-regressions.sh`
with `UPDATE_BASELINES=1` after the examples have been built. Only examples with
a subdirectory here are part of the baseline comparison by default.

The regression runner stores each executable's stdout as `stdout.txt` beside
any CSV/JSON/TXT files emitted by the example, so examples whose primary data is
HDF5 still get a small text readout in CI.
