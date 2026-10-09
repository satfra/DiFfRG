# CI scripts

CI (`.github/workflows/ci.yml`) builds and tests the DiFfRG library against the
same pre-built dependency bundle users install (`containers/release/`,
`install-diffrg-deps.sh`), so the multi-hour dependency superbuild stays out of
the regular test path and CI tests exactly what users get.

## The dependency bundle in CI

`.github/deps-bundle-version` pins the `deps-v<X.Y.Z>` release CI uses (its
`linux-x86_64-v3-cpu` variant). The `Dependency bundle` job

1. hashes the dependency inputs of the checkout
   (`containers/release/deps-inputs-hash.sh`: the superbuild (`superbuild.cmake`),
   `dependencies/` without PETSc, `patches/`, and the CPU variant's recipe and
   post-processing);
2. downloads the pinned bundle and compares that hash with the
   `deps_inputs_hash` its `BUNDLE_MANIFEST.json` records;
3. on a mismatch, builds a bundle from the checkout with
   `containers/release/build-release.sh` (~3 h; version `0.0.0`) and caches it
   under the input hash, so further pushes with the same inputs reuse it;
4. hands the bundle to the build jobs, which install it with
   `install-diffrg-deps.sh --file` in a plain `ubuntu:24.04` container.

So a PR that changes dependencies is tested against bundles built from its own
inputs. Before merging it, cut a release with those inputs (see
`containers/release/README.md`) and bump `.github/deps-bundle-version`, so that
later runs download it instead of rebuilding. Fork PRs cannot build a bundle
(no cache writes); their dependency changes need a matching release first.

## Files

| File | Purpose |
|------|---------|
| `build-examples.sh` | Installs DiFfRG against the bundle at `$DiFfRG_BUNDLED_DIR` into `.ci/diffrg-install` and builds the current example CMake projects, with per-example logs. |
| `run-wolfram-checks.sh` | Runs the Wolfram checks on a host with `wolframscript`, FORM, FunKit and FormTracer, then builds the examples and runs the baseline regressions. |
| `wolfram-example-checks.sh` | Records Wolfram preflight, generator, and generated-flow-drift results. |
| `run-example-regressions.sh` | Runs selected built examples with short CI overrides and compares text outputs against `Examples/ci_baselines/`. |
| `compare-example-baseline.py` | Normalizes and compares or updates small text baselines for example regression outputs. |

The example and Wolfram jobs in `ci.yml` are currently disabled (`if: false`):
the examples are too slow for hosted runners, and the Wolfram job needs a
self-hosted runner with a Wolfram installation.

## Test-count badge (README)

The README shows a `tests N/M passing` badge (green when all pass, red otherwise).
GitHub doesn't expose the test count, so `ci.yml` publishes it to a Gist that
[shields.io](https://shields.io) renders. One-time setup:

1. **Create a public Gist** (gist.github.com) with a file named `diffrg-tests.json`
   (any placeholder content). Note its ID — the long hash in the Gist URL.
2. **Create a PAT** (classic) with **only** the `gist` scope, and add it as a repo
   secret named **`GIST_SECRET`** (Settings → Secrets and variables → Actions).
3. **Fill in the two placeholders:**
   - `.github/workflows/ci.yml` → `gistID: REPLACE_WITH_GIST_ID`
   - `README.md` badge URL → `REPLACE_WITH_GIST_OWNER` (the Gist owner's username)
     and `REPLACE_WITH_GIST_ID`.

The badge updates on every **push to `main`** (the `Update test-count badge` step,
which runs with `if: always()`). It shows `N/M passing` (green if all pass, red if
any fail), or red **`build failed`** if the library didn't compile — so a broken
build never leaves the badge stuck at a stale/placeholder value. PRs intentionally
skip the update — fork PRs can't read secrets — but PR runs still build and test.
The count comes from parsing ctest's `"<F> tests failed out of <T>"` summary line,
so it needs no extra tooling.

## Verifying the CI path locally

`containers/release/test-tarball.sh -f <tarball> -d ubuntu24.04` runs the same
install, library build and quick test suite in a clean `ubuntu:24.04` image.
