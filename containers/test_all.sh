#!/bin/bash
# ##############################################################################
# Build the DiFfRG library in every CPU-only container in containers/Base/ to
# check that the build system works across distributions.
#
# Each image builds the local working tree (build.sh -d -f). Per-image logs are
# written to containers/logs/<image>.log; images are removed after each build to
# reclaim disk space. A PASS/FAIL summary is printed at the end.
#
# Usage: test_all.sh [-j <threads>]
# Expect roughly ~20 min per image (deal.II dominates), ~1.5 h for four.
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

images=$(ls Base/)
echo "Building DiFfRG in ${threads}-thread containers for:"
echo "${images}" | sed 's/^/  - /'
echo

summary=""
anyfail=0
start=$(date +%s)

for image in ${images}; do
  echo "###############################################################"
  echo "## Building ${image}  ($(date '+%H:%M:%S'))"
  echo "###############################################################"
  if docker buildx build \
    -t "diffrg-test-${image}" \
    -f "${scriptpath}/Base/${image}" \
    "${repo}" \
    --no-cache --progress=plain \
    --build-arg threads="${threads}" &>"logs/${image}.log"; then
    status="PASS"
    echo "   ${image}: PASS"
  else
    status="FAIL"
    anyfail=1
    echo "   ${image}: FAIL  (see logs/${image}.log)"
  fi
  # Reclaim space: the image is large and we only care about build success.
  docker rmi -f "diffrg-test-${image}" &>/dev/null
  summary="${summary}$(printf '  %-16s %s   (containers/logs/%s.log)\n' "${image}" "${status}" "${image}")"$'\n'
done

end=$(date +%s)
runtime=$((end - start))

echo
echo "###############################################################"
echo "## DiFfRG multi-distro build summary"
echo "###############################################################"
printf "%b" "${summary}"
echo "  Elapsed: $((runtime / 3600))h $(((runtime / 60) % 60))m $((runtime % 60))s"
if [[ ${anyfail} -ne 0 ]]; then
  echo "  Some builds FAILED."
  exit 1
fi
echo "  All builds passed."
