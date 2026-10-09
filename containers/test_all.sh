#!/bin/bash
# ##############################################################################
# Build DiFfRG in every container under containers/Base/ (CPU) and containers/
# CUDA/ to check that the build system works across distributions, and run the
# test suite in each.
#
# Each image builds the local working tree via a cmake superbuild (no installer
# scripts involved). Per-image logs in containers/logs/:
#   <image>.log         build (compile) output, incl. system-vs-bundled deps
#   <image>_ctest.log   ctest results (CUDA images: only when a host GPU exists)
# Docker images are removed after each build to reclaim disk space; Singularity
# SIFs are cached under containers/logs/sif-cache for the matching test run.
# A PASS/FAIL summary (build + tests) is printed at the end.
#
# Usage: test_all.sh [-j <threads>] [-a <cuda_arch>]
# Expect roughly ~20-40 min per image (deal.II dominates).
# ##############################################################################

scriptpath="$(
  cd -- "$(dirname "$0")" >/dev/null 2>&1
  pwd -P
)"
repo="$(cd -- "${scriptpath}/.." >/dev/null 2>&1 && pwd -P)"

threads=''
cuda_arch='AMPERE80'
while getopts j:a: flag; do
  case "${flag}" in
  j) threads=${OPTARG} ;;
  a) cuda_arch=${OPTARG} ;;
  esac
done

if [[ -z ${threads} ]]; then
  ncores=''
  if command -v nproc >/dev/null 2>&1; then
    ncores=$(nproc)
  elif command -v sysctl >/dev/null 2>&1; then
    ncores=$(sysctl -n hw.ncpu 2>/dev/null)
  fi
  ncores=${ncores:-2}
  threads=$((ncores / 2))
  [[ ${threads} -lt 1 ]] && threads=1
fi

cd "${scriptpath}" || exit 1
mkdir -p logs

# Collect "<subdir>/<name>" definitions from Base/ (CPU) and CUDA/.
images=()
for sub in Base CUDA; do
  [[ -d ${sub} ]] || continue
  for f in "${sub}"/*; do
    [[ -f ${f} ]] && images+=("${f}")
  done
done

have_gpu=0
command -v nvidia-smi >/dev/null 2>&1 && have_gpu=1

echo "Building DiFfRG in ${threads}-thread containers for:"
printf '  - %s\n' "${images[@]}"
[[ ${have_gpu} -eq 0 ]] && echo "(no host GPU detected: CUDA images build only, ctest skipped)"
echo

testdir="/build/DiFfRG/src/DiFfRG-build"
summary=""
anyfail=0
start=$(date +%s)

for dockerfile in "${images[@]}"; do
  name="$(basename "${dockerfile}")"
  tag="diffrg-test-${name}"
  is_cuda=0
  [[ ${dockerfile} == CUDA/* ]] && is_cuda=1

  echo "###############################################################"
  echo "## Building ${name}  ($(date '+%H:%M:%S'))"
  echo "###############################################################"

  build_args=(--build-arg "threads=${threads}")
  [[ ${is_cuda} -eq 1 ]] && build_args+=(--build-arg "cuda_arch=${cuda_arch}")

  if docker buildx build --load \
    -t "${tag}" \
    -f "${scriptpath}/${dockerfile}" \
    "${repo}" \
    --no-cache --progress=plain \
    "${build_args[@]}" &>"logs/${name}.log"; then
    build_status="PASS"
    echo "   ${name}: build PASS"
  else
    build_status="FAIL"
    anyfail=1
    echo "   ${name}: build FAIL  (see logs/${name}.log)"
  fi

  # Run tests only if the build succeeded. ctest via bash -c so BASH_ENV (e.g.
  # Rocky's gcc-toolset-12) is sourced.
  test_status="SKIP"
  if [[ ${build_status} == "PASS" ]]; then
    if [[ ${is_cuda} -eq 1 && ${have_gpu} -eq 0 ]]; then
      echo "No host GPU; skipping ctest for ${name}." >"logs/${name}_ctest.log"
      test_status="SKIP(no GPU)"
    else
      singularity_args=(-W)
      [[ ${is_cuda} -eq 1 ]] && singularity_args+=(-g)
      if bash "${scriptpath}/singularity-run.sh" "${singularity_args[@]}" "docker-daemon://${tag}" \
        bash -lc "ctest --test-dir ${testdir} --output-on-failure" &>"logs/${name}_ctest.log"; then
        test_status="PASS"
        echo "   ${name}: tests PASS"
      else
        test_status="FAIL"
        anyfail=1
        echo "   ${name}: tests FAIL  (see logs/${name}_ctest.log)"
      fi
    fi
  fi

  # Reclaim Docker image space; we only keep the logs and Singularity cache.
  docker rmi -f "${tag}" &>/dev/null
  summary="${summary}$(printf '  %-20s build:%-5s tests:%-12s (logs/%s.log, logs/%s_ctest.log)\n' \
    "${name}" "${build_status}" "${test_status}" "${name}" "${name}")"$'\n'
done

end=$(date +%s)
runtime=$((end - start))

echo
echo "###############################################################"
echo "## DiFfRG multi-distro build + test summary"
echo "###############################################################"
printf "%b" "${summary}"
echo "  Elapsed: $((runtime / 3600))h $(((runtime / 60) % 60))m $((runtime % 60))s"
if [[ ${anyfail} -ne 0 ]]; then
  echo "  Some builds or tests FAILED."
  exit 1
fi
echo "  All builds and tests passed."
