#!/bin/bash
# ##############################################################################
# Build the DiFfRG library in every CPU-only container in containers/Base/ to
# check that the build system works across distributions.
#
# Each image builds the local working tree (build.sh -d -f). After a successful
# build the library test suite (run_tests.sh) is run inside the container.
# Per-image logs are written to containers/logs/<image>.log (build) and
# containers/logs/<image>-tests.log (tests); images are removed after each run
# to reclaim disk space. A PASS/FAIL summary is printed at the end.
#
# Usage: test_all.sh [-j <threads>]
# Expect roughly ~25 min per image (deal.II dominates), ~1.5 h+ for four.
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
    echo "   ${image}: build PASS"

    # Build succeeded: run the library test suite inside the freshly built
    # image. The repo is copied to /DiFfRG (see Base/<image>), so run_tests.sh
    # finds the superbuild it just produced.
    echo "   ${image}: running tests  ($(date '+%H:%M:%S'))"
    if docker run --rm "diffrg-test-${image}" \
      bash /DiFfRG/run_tests.sh -j "${threads}" &>"logs/${image}-tests.log"; then
      teststatus="PASS"
      echo "   ${image}: tests PASS"
    else
      teststatus="FAIL"
      anyfail=1
      echo "   ${image}: tests FAIL  (see logs/${image}-tests.log)"
    fi
  else
    status="FAIL"
    teststatus="SKIP"
    anyfail=1
    echo "   ${image}: build FAIL  (see logs/${image}.log)"
  fi
  # Reclaim space: the image is large and we only care about build/test success.
  docker rmi -f "diffrg-test-${image}" &>/dev/null
  summary="${summary}$(printf '  %-16s build %-4s  tests %-4s   (containers/logs/%s.log, containers/logs/%s-tests.log)\n' "${image}" "${status}" "${teststatus}" "${image}" "${image}")"$'\n'
done

end=$(date +%s)
runtime=$((end - start))

echo
echo "###############################################################"
echo "## DiFfRG multi-distro build & test summary"
echo "###############################################################"
printf "%b" "${summary}"
echo "  Elapsed: $((runtime / 3600))h $(((runtime / 60) % 60))m $((runtime % 60))s"
if [[ ${anyfail} -ne 0 ]]; then
  echo "  Some builds or tests FAILED."
  exit 1
fi
echo "  All builds and tests passed."
