#!/usr/bin/env bash
# Run the Wolfram example checks on a host with wolframscript, FORM, FunKit and
# FormTracer installed, then (RUN_EXAMPLE_REGRESSIONS=1, the default) build the
# examples against the dependency bundle at $DiFfRG_BUNDLED_DIR and compare
# their short-run outputs against Examples/ci_baselines/.
set -euo pipefail

workspace="${WORKSPACE:-$(cd -- "$(dirname "$0")/../.." >/dev/null 2>&1 && pwd -P)}"
export WORKSPACE="${workspace}"
summary_file="${workspace}/.ci/logs/wolfram-summary.md"
mkdir -p "$(dirname "${summary_file}")"

if ! command -v wolframscript >/dev/null 2>&1; then
  {
    echo "## Wolfram generation"
    echo
    echo "Skipped: \`wolframscript\` was not found on PATH."
  } > "${summary_file}"
  exit 0
fi

wolfram_status=0
bash "${workspace}/containers/ci/wolfram-example-checks.sh" || wolfram_status=$?
if [[ ${wolfram_status} -eq 75 ]]; then
  echo "Wolfram preflight failed; cannot run generators or baseline regressions."
  exit 1
fi
[[ ${wolfram_status} -eq 0 ]] || exit "${wolfram_status}"

if [[ "${RUN_EXAMPLE_REGRESSIONS:-1}" == "1" ]]; then
  bash "${workspace}/containers/ci/build-examples.sh"
  bash "${workspace}/containers/ci/run-example-regressions.sh"
fi
