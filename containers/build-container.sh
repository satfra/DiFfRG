#!/bin/bash
# ##############################################################################
# Interactively build a single DiFfRG build-test container from containers/Base/.
#
# Usage: build-container.sh [-j <threads>]
# The image builds the local working tree (via build.sh -d -f) into /opt/DiFfRG.
# ##############################################################################

scriptpath="$(
  cd -- "$(dirname "$0")" >/dev/null 2>&1
  pwd -P
)"
repo="$(cd -- "${scriptpath}/.." >/dev/null 2>&1 && pwd -P)"

threads=''
while getopts j: flag; do
  case "${flag}" in
  j) threads=${OPTARG} ;;
  esac
done

# Default to half the host cores (matches build.sh / run_tests.sh).
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

cd "${scriptpath}"
mkdir -p logs

# List the available Base images.
images=$(ls Base/)
echo "Available DiFfRG build-test images (CPU-only):"
i=1
for image in ${images}; do
  echo "  $i) ${image}"
  i=$((i + 1))
done
echo
read -p "Enter the number of the image to build: " choice
if ! [[ ${choice} =~ ^[0-9]+$ ]] || [[ ${choice} -lt 1 ]] || [[ ${choice} -gt $(echo "${images}" | wc -l) ]]; then
  echo "Invalid choice."
  exit 1
fi
image=$(echo "${images}" | sed -n "${choice}p")

echo
echo "Building diffrg-test-${image} with ${threads} threads (context: ${repo})..."
echo "Log: ${scriptpath}/logs/${image}.log"
docker buildx build \
  -t "diffrg-test-${image}" \
  -f "${scriptpath}/Base/${image}" \
  "${repo}" \
  --no-cache --progress=plain \
  --build-arg threads="${threads}" 2>&1 | tee "logs/${image}.log"

# tee hides docker's exit code; recover it.
if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
  echo "   Successfully built diffrg-test-${image}."
else
  echo "   Build FAILED for ${image}. See logs/${image}.log."
  exit 1
fi
