#!/usr/bin/env bash
# Run selected built examples and compare their small text outputs against
# committed baselines. Run with UPDATE_BASELINES=1 to refresh baselines.
set -euo pipefail

workspace="${WORKSPACE:-$(cd -- "$(dirname "$0")/../.." >/dev/null 2>&1 && pwd -P)}"
examples_build_root="${DIFFRG_EXAMPLES_BUILD_ROOT:-${workspace}/.ci/examples}"
run_root="${DIFFRG_EXAMPLE_RUN_ROOT:-${workspace}/.ci/run-results}"
log_dir="${DIFFRG_EXAMPLE_REGRESSION_LOG_DIR:-${workspace}/.ci/logs/example-regressions}"
summary_file="${DIFFRG_EXAMPLE_REGRESSION_SUMMARY:-${workspace}/.ci/logs/example-regressions-summary.md}"
baseline_root="${DIFFRG_EXAMPLE_BASELINE_ROOT:-${workspace}/Examples/ci_baselines}"
update_baselines="${UPDATE_BASELINES:-0}"
required_examples="${REQUIRED_EXAMPLE_REGRESSIONS:-}"

# Only examples with committed baselines are checked by default. This lets the
# current legacy examples remain untouched while making the regression surface
# opt-in and reviewable.
default_examples=()
if [[ -d "${baseline_root}" ]]; then
  while IFS= read -r -d '' dir; do
    default_examples+=("$(basename "${dir}")")
  done < <(find "${baseline_root}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
fi

if [[ -n "${EXAMPLE_REGRESSIONS:-}" ]]; then
  read -r -a examples <<< "${EXAMPLE_REGRESSIONS}"
elif [[ "${update_baselines}" == "1" && ${#default_examples[@]} -eq 0 ]]; then
  examples=(ONfiniteT QuarkMesonLPAprime YangMills_SP YangMills_Full)
else
  examples=("${default_examples[@]}")
fi

mkdir -p "${run_root}" "${log_dir}" "$(dirname "${summary_file}")"

{
  echo "## Example regression outputs"
  echo
  echo "| Example | Result | Log | Note |"
  echo "| --- | --- | --- | --- |"
} > "${summary_file}"

is_required() {
  local name="$1"
  case " ${required_examples} " in
    *" ${name} "*) return 0 ;;
    *) return 1 ;;
  esac
}

run_example() {
  local name="$1"
  local exe=""
  local build_dir="${examples_build_root}/${name}"

  case "${name}" in
    ONfiniteT) exe="${build_dir}/CG" ;;
    QuarkMesonLPAprime) exe="${build_dir}/QuarkMesonLPAprime" ;;
    YangMills_SP) exe="${build_dir}/YangMills" ;;
    YangMills_Full) exe="${build_dir}/YangMills" ;;
    FourFermi) exe="${build_dir}/FourFermi" ;;
    *) return 64 ;;
  esac

  if [[ ! -x "${exe}" ]]; then
    echo "Executable not found: ${exe}" >&2
    return 127
  fi

  local out_dir="${run_root}/${name}/raw"
  rm -rf "${run_root:?}/${name}"
  mkdir -p "${out_dir}"

  pushd "$(dirname "${exe}")" >/dev/null
  local status=0
  "${exe}" \
    -ss "/output/folder=${out_dir}/" \
    -ss "/output/name=output" \
    -sd "/timestepping/final_time=${EXAMPLE_REGRESSION_FINAL_TIME:-0.02}" \
    -sd "/timestepping/output_dt=${EXAMPLE_REGRESSION_OUTPUT_DT:-0.02}" \
    2>&1 | tee "${out_dir}/stdout.txt" || status=$?
  popd >/dev/null
  return "${status}"
}

if [[ ${#examples[@]} -eq 0 ]]; then
  echo "| all | skipped | | no baselines found under \`${baseline_root#${workspace}/}\` |" >> "${summary_file}"
  exit 0
fi

failures=0

for name in "${examples[@]}"; do
  log="${log_dir}/${name}.log"
  baseline="${baseline_root}/${name}"
  work="${run_root}/${name}"
  diff="${log_dir}/${name}.diff"

  set +e
  run_example "${name}" > "${log}" 2>&1
  run_status=$?
  set -e

  if [[ ${run_status} -ne 0 ]]; then
    echo "| ${name} | run failed | \`${log#${workspace}/}\` | exit ${run_status} |" >> "${summary_file}"
    if is_required "${name}"; then
      failures=$((failures + 1))
    fi
    continue
  fi

  compare_args=(
    --actual "${work}/raw"
    --baseline "${baseline}"
    --work "${work}"
    --diff-output "${diff}"
  )
  if [[ "${update_baselines}" == "1" ]]; then
    compare_args+=(--update)
  fi

  set +e
  python3 "${workspace}/containers/ci/compare-example-baseline.py" "${compare_args[@]}" >> "${log}" 2>&1
  compare_status=$?
  set -e

  if [[ "${update_baselines}" == "1" ]]; then
    echo "| ${name} | baseline updated | \`${log#${workspace}/}\` | \`${baseline#${workspace}/}\` |" >> "${summary_file}"
  elif [[ ${compare_status} -eq 0 ]]; then
    rm -f "${diff}"
    echo "| ${name} | passed | \`${log#${workspace}/}\` | |" >> "${summary_file}"
  else
    echo "| ${name} | baseline mismatch | \`${log#${workspace}/}\` | diff: \`${diff#${workspace}/}\` |" >> "${summary_file}"
    failures=$((failures + 1))
  fi
done

if [[ ${failures} -ne 0 ]]; then
  echo "${failures} example regression(s) failed." >&2
  exit 1
fi
