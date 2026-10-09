#!/bin/bash
# ##############################################################################
# Interactively build a single DiFfRG build-test container and run its tests.
#
# Usage: build-container.sh [-j <threads>] [-a <cuda_arch>]
#   -j  build threads (default: half the host cores)
#   -a  Kokkos CUDA arch for CUDA images (default: AMPERE80)
#
# The image builds the local working tree (cmake superbuild) into the default
# install prefix ($HOME/.local/share/DiFfRG) and
# compiles the test suite. After a successful build the tests are run through
# Singularity and both logs are written to containers/logs/:
#   <image>.log         the build (compile) output, incl. system-vs-bundled deps
#   <image>_ctest.log   the ctest results
# CUDA images only run ctest when a host GPU (nvidia-smi) is available.
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

# Default to half the host cores.
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

# Collect images as "<subdir>/<name>" from both Base/ (CPU) and CUDA/.
images=()
for sub in Base CUDA; do
  [[ -d ${sub} ]] || continue
  for f in "${sub}"/*; do
    [[ -f ${f} ]] && images+=("${f}")
  done
done
if [[ ${#images[@]} -eq 0 ]]; then
  echo "No container definitions found under Base/ or CUDA/."
  exit 1
fi

echo "Available DiFfRG build-test images:"
i=1
for img in "${images[@]}"; do
  echo "  $i) ${img}"
  i=$((i + 1))
done
echo
read -r -p "Enter the number of the image to build: " choice
if ! [[ ${choice} =~ ^[0-9]+$ ]] || [[ ${choice} -lt 1 ]] || [[ ${choice} -gt ${#images[@]} ]]; then
  echo "Invalid choice."
  exit 1
fi
dockerfile="${images[$((choice - 1))]}"
name="$(basename "${dockerfile}")"
tag="diffrg-test-${name}"
is_cuda=0
[[ ${dockerfile} == CUDA/* ]] && is_cuda=1

echo
echo "Building ${tag} with ${threads} threads (context: ${repo})..."
echo "Build log: ${scriptpath}/logs/${name}.log"
build_args=(--build-arg "threads=${threads}")
[[ ${is_cuda} -eq 1 ]] && build_args+=(--build-arg "cuda_arch=${cuda_arch}")

docker buildx build --load \
  -t "${tag}" \
  -f "${scriptpath}/${dockerfile}" \
  "${repo}" \
  --no-cache --progress=plain \
  "${build_args[@]}" 2>&1 | tee "logs/${name}.log"

if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
  echo "   Build FAILED for ${name}. See logs/${name}.log."
  exit 1
fi
echo "   Successfully built ${tag}."

# Run the test suite. ctest is invoked through bash -c so non-interactive shells
# pick up BASH_ENV (e.g. Rocky's gcc-toolset-12 runtime).
testdir="/build/DiFfRG/src/DiFfRG-build"
ctest_cmd="ctest --test-dir ${testdir} --output-on-failure"
if [[ ${is_cuda} -eq 1 ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "Running GPU tests (Singularity --nv). Log: logs/${name}_ctest.log"
    bash "${scriptpath}/singularity-run.sh" -g -W "docker-daemon://${tag}" bash -lc "${ctest_cmd}" 2>&1 | tee "logs/${name}_ctest.log"
  else
    echo "No host GPU (nvidia-smi) found; skipping ctest for ${name}." | tee "logs/${name}_ctest.log"
  fi
else
  echo "Running tests (Singularity). Log: logs/${name}_ctest.log"
  bash "${scriptpath}/singularity-run.sh" -W "docker-daemon://${tag}" bash -lc "${ctest_cmd}" 2>&1 | tee "logs/${name}_ctest.log"
fi
