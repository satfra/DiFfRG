#!/usr/bin/env bash
# Install DiFfRG against the dependency bundle at $DiFfRG_BUNDLED_DIR and build
# all current Examples against that install.
set -euo pipefail

workspace="${WORKSPACE:-$(cd -- "$(dirname "$0")/../.." >/dev/null 2>&1 && pwd -P)}"
build_jobs="${DIFFRG_BUILD_JOBS:-4}"
install_prefix="${DIFFRG_EXAMPLES_INSTALL_PREFIX:-${workspace}/.ci/diffrg-install}"
library_build_dir="${DIFFRG_EXAMPLES_LIBRARY_BUILD_DIR:-${workspace}/.ci/diffrg-examples-lib}"
examples_build_root="${DIFFRG_EXAMPLES_BUILD_ROOT:-${workspace}/.ci/examples}"
log_dir="${DIFFRG_EXAMPLES_LOG_DIR:-${workspace}/.ci/logs/examples}"
summary_file="${DIFFRG_EXAMPLES_SUMMARY:-${workspace}/.ci/logs/examples-summary.md}"
bundle_dir="${DiFfRG_BUNDLED_DIR:?set DiFfRG_BUNDLED_DIR to an installed dependency bundle}"

examples=(
  "ONfiniteT:Examples/ONfiniteT"
  "QuarkMesonLPAprime:Examples/QuarkMesonLPAprime"
  "YangMills_SP:Examples/YangMills/SP"
  "YangMills_Full:Examples/YangMills/Full"
  "FourFermi:Examples/FourFermi"
)

mkdir -p "${install_prefix}" "${library_build_dir}" "${examples_build_root}" "${log_dir}" "$(dirname "${summary_file}")"

{
  echo "## Example builds"
  echo
  echo "| Example | Result | Log | Note |"
  echo "| --- | --- | --- | --- |"
} > "${summary_file}"

library_log="${log_dir}/diffrg-install.log"
{
  echo "Configuring and installing DiFfRG for downstream example builds"
  echo "Install prefix: ${install_prefix}"
  echo "Bundle dir: ${bundle_dir}"
} > "${library_log}"

cmake -S "${workspace}/DiFfRG" -B "${library_build_dir}" \
  -DBUNDLED_DIR="${bundle_dir}" \
  -DCMAKE_INSTALL_PREFIX="${install_prefix}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DDiFfRG_TEST=OFF \
  -DDiFfRG_DOCUMENTATION=OFF \
  -DMARCH=none >> "${library_log}" 2>&1
cmake --build "${library_build_dir}" --target install -j "${build_jobs}" >> "${library_log}" 2>&1

# DiFfRGConfig.cmake records ${CMAKE_INSTALL_PREFIX}/bundled. Reuse the
# installed bundle instead of copying it into the workspace.
rm -rf "${install_prefix}/bundled"
ln -s "${bundle_dir}" "${install_prefix}/bundled"

unexpected_failures=0

for item in "${examples[@]}"; do
  name="${item%%:*}"
  relpath="${item#*:}"
  src="${workspace}/${relpath}"
  build_dir="${examples_build_root}/${name}"
  log="${log_dir}/${name}.log"

  rm -rf "${build_dir}"
  mkdir -p "${build_dir}"

  set +e
  (
    echo "Configuring ${name} from ${relpath}"
    cmake -S "${src}" -B "${build_dir}" \
      -DDiFfRG_DIR="${install_prefix}/lib/cmake/DiFfRG" \
      -DCMAKE_PREFIX_PATH="${install_prefix}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DNATIVE=OFF
    configure_status=$?
    if [[ ${configure_status} -eq 0 ]]; then
      cmake --build "${build_dir}" -j "${build_jobs}"
      build_status=$?
    else
      build_status=${configure_status}
    fi
    exit "${build_status}"
  ) > "${log}" 2>&1
  status=$?
  set -e

  log_link="${log#${workspace}/}"
  if [[ ${status} -eq 0 ]]; then
    echo "| ${name} | passed | \`${log_link}\` | |" >> "${summary_file}"
  else
    echo "Example build ${name} failed with exit ${status}."
    echo "---- ${log_link} ----"
    cat "${log}"
    echo "---- end ${log_link} ----"
    echo "| ${name} | failed | \`${log_link}\` | exit ${status} |" >> "${summary_file}"
    unexpected_failures=$((unexpected_failures + 1))
  fi
done

echo >> "${summary_file}"
echo "DiFfRG install log: \`${library_log#${workspace}/}\`" >> "${summary_file}"

if [[ ${unexpected_failures} -ne 0 ]]; then
  echo "---- ${summary_file#${workspace}/} ----"
  cat "${summary_file}"
  echo "---- end ${summary_file#${workspace}/} ----"
  echo "${unexpected_failures} unexpected example build(s) failed." >&2
  exit 1
fi
