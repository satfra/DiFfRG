#!/bin/bash
# ##############################################################################
# Validate a pre-built DiFfRG dependency bundle tarball across distros.
#
# For each distro in the matrix: build the minimal test image (which has NO
# boost/tbb/hdf5/sundials dev packages -- self-containment is part of the test),
# then, through Singularity/Apptainer:
#   1. install the tarball with install-diffrg-deps.sh (exercises download-path
#      logic, relocation fixups, verify_install),
#   2. audit linkage of the relocated bundle,
#   3. build the DiFfRG library + tests against it,
#   4. run the quick test suite (skipped when the host CPU lacks x86-64-v3;
#      for MPI bundles this includes the multi-rank `mpi`-labelled tests),
#   5. install the tarball a SECOND time at another prefix and check no
#      build-prefix references survive (move-prefix/relocatability test).
#
# Usage: test-tarball.sh -f <tarball> [-j <threads>] [-d <distro>[,<distro>...]] [-s] [-g] [-m]
#   -f <tarball>  the diffrg-deps-*.tar.zst to validate (required)
#   -j <threads>  build threads inside each container (default: 6)
#   -d <list>     comma-separated subset of: ubuntu24.04 debian13 fedora41 rockylinux9
#                 (with -g: ubuntu24.04-cuda rockylinux9-cuda)
#   -s            skip ctest (build-only validation; also automatic on non-v3 hosts)
#   -g            CUDA bundle: use the -cuda test images, pass the host GPU
#                 through (docker --gpus), allow CUDA libs in the linkage audit;
#                 without a host GPU ctest is skipped automatically
#   -m            MPI (Open MPI) bundle: install the distro's Open MPI in the
#                 test images and allow it in the linkage audit
#
# DOCKER_NETWORK=host runs image builds and docker runs on the host network
# (for hosts whose VPN client blocks docker's bridge network).
#
# On success writes <tarball>.tested (consumed by publish-release.sh).
# Logs land in containers/release/logs/.
# ##############################################################################
set -euo pipefail

scriptpath="$(
  cd -- "$(dirname "$0")" >/dev/null 2>&1
  pwd -P
)"
repo="$(cd -- "${scriptpath}/../.." >/dev/null 2>&1 && pwd -P)"

tarball=''
threads=6
distros=''
skip_tests=0
gpu=0
mpi=none

while getopts f:j:d:sgm flag; do
  case "${flag}" in
  f) tarball=${OPTARG} ;;
  j) threads=${OPTARG} ;;
  d) distros="${OPTARG//,/ }" ;;
  s) skip_tests=1 ;;
  g) gpu=1 ;;
  m) mpi=openmpi ;;
  *)
    echo "Unknown flag." >&2
    exit 1
    ;;
  esac
done

# The variant is in the tarball's name; follow it rather than trust the flags to match.
case "$(basename "${tarball}")" in *-openmpi.tar.zst) mpi=openmpi ;; esac
case "$(basename "${tarball}")" in *-cuda*) gpu=1 ;; esac

if [[ -z ${distros} ]]; then
  # CUDA test images exist only where NVIDIA publishes devel bases.
  [[ ${gpu} -eq 1 ]] && distros="ubuntu24.04-cuda rockylinux9-cuda" \
    || distros="ubuntu24.04 debian13 fedora41 rockylinux9"
fi

[[ -n ${tarball} && -f ${tarball} ]] || {
  echo "A tarball is required: test-tarball.sh -f dist/diffrg-deps-....tar.zst" >&2
  exit 1
}
tarball="$(readlink -f "${tarball}")"
tarname="$(basename "${tarball}")"

# Running the built tests executes x86-64-v3 code from the bundle; degrade to
# build-only on a host without AVX2/FMA. CUDA tests additionally need a GPU.
run_tests=1
if [[ ${skip_tests} -eq 1 ]]; then
  run_tests=0
elif ! grep -qm1 avx2 /proc/cpuinfo || ! grep -qm1 fma /proc/cpuinfo; then
  echo "Host CPU lacks x86-64-v3 (AVX2+FMA): building only, skipping ctest."
  run_tests=0
elif [[ ${gpu} -eq 1 ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "No host GPU (nvidia-smi): building only, skipping ctest."
  run_tests=0
fi

logdir="${scriptpath}/logs"
mkdir -p "${logdir}"

# Prefer the Singularity path (cluster parity, same as CI); fall back to plain
# docker run where neither singularity nor apptainer exists.
runtime=docker
if command -v singularity >/dev/null 2>&1 || command -v apptainer >/dev/null 2>&1; then
  runtime=singularity
fi

run_in_image() { # <image> <bind>... -- <script>
  local image="$1" script="${*: -1}"
  local -a binds=("${@:2:$#-3}")
  if [[ ${runtime} == singularity ]]; then
    local -a args=()
    [[ ${gpu} -eq 1 && ${run_tests} -eq 1 ]] && args+=(-g)
    for b in "${binds[@]}"; do args+=(-b "${b}"); done
    bash "${scriptpath}/../singularity-run.sh" "${args[@]}" \
      "docker-daemon://${image}" bash -lc "${script}"
  else
    local -a args=()
    [[ ${gpu} -eq 1 && ${run_tests} -eq 1 ]] && args+=(--gpus all)
    for b in "${binds[@]}"; do args+=(-v "${b}"); done
    # Open MPI's shared-memory transport needs more than docker's 64 MB /dev/shm.
    [[ ${mpi} != none ]] && args+=(--shm-size=1g)
    docker run --rm --network "${DOCKER_NETWORK:-default}" -e "CHECK_LINKAGE_CUDA=${gpu}" "${args[@]}" "${image}" bash -lc "${script}"
  fi
}

declare -A results
overall=0

for distro in ${distros}; do
  dockerfile="${scriptpath}/test/${distro}.Dockerfile"
  [[ -f ${dockerfile} ]] || {
    echo "Unknown distro '${distro}' (no ${dockerfile})" >&2
    exit 1
  }
  tag="${distro}$([[ ${mpi} == none ]] || echo "-${mpi}")"
  image="diffrg-deps-release-test:${tag}"
  log="${logdir}/${tag}.log"
  workdir="$(mktemp -d)"

  echo "=== ${distro}: building test image..."
  if ! docker buildx build --load --network "${DOCKER_NETWORK:-default}" -t "${image}" -f "${dockerfile}" \
    --build-arg "mpi=${mpi}" "${repo}" >"${log}" 2>&1; then
    results[${distro}]="FAIL (image build)"
    overall=1
    rm -rf "${workdir}"
    continue
  fi

  ctest_step=''
  if [[ ${run_tests} -eq 1 ]]; then
    ctest_step="ctest --test-dir /work/build -LE slow -j1 --output-on-failure"
  fi

  echo "=== ${distro}: install + build + test via ${runtime} (log: ${log})..."
  if run_in_image "${image}" \
    "${repo}:/src" "$(dirname "${tarball}"):/dist" "${workdir}:/work" -- "
      set -ex
      # Docker runs as root, which Open MPI refuses without these; oversubscribe
      # (4.x and 5.x spellings) so multi-rank tests run on small CI runners.
      if [[ ${mpi} != none ]]; then
        export CHECK_LINKAGE_MPI=1 OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
        export OMPI_MCA_rmaps_base_oversubscribe=1 PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe
      fi
      # CUDA bundles reference the driver's libcuda.so.1; without a GPU mounted
      # (build-only validation) satisfy the ldd audit with the toolkit's stub.
      # Register the stub's own directory with the loader rather than guessing
      # a lib dir -- /usr/lib64 exists on Ubuntu but is not in its ldconfig
      # path, which made a symlink there succeed yet stay invisible.
      if [[ \${CHECK_LINKAGE_CUDA:-0} == 1 ]] && ! ldconfig -p | grep -q libcuda.so.1; then
        stub=\$(find /usr/local/cuda* -name libcuda.so -path '*stubs*' 2>/dev/null | head -1)
        if [[ -n \$stub ]]; then
          ln -sf \"\$stub\" \"\${stub%/*}/libcuda.so.1\"
          echo \"\${stub%/*}\" > /etc/ld.so.conf.d/zz-cuda-stubs.conf
          ldconfig
        fi
      fi
      bash /src/install-diffrg-deps.sh --file /dist/${tarname} \
          --prefix /work/diffrg --skip-cpu-check
      bash /src/containers/release/check-linkage.sh /work/diffrg/bundled
      # No -DMPI: the library must pick it up from the bundle's pin.
      cmake -S /src/DiFfRG -B /work/build \
          -DCMAKE_BUILD_TYPE=Release \
          -DBUNDLED_DIR=/work/diffrg/bundled \
          -DDiFfRG_TEST=ON -DDiFfRG_DOCUMENTATION=OFF -DMARCH=none >/work/configure.log 2>&1 \
        || { cat /work/configure.log; exit 1; }
      cat /work/configure.log
      grep -q 'MPI support has been set to $([[ ${mpi} == none ]] && echo OFF || echo ON)' /work/configure.log
      cmake --build /work/build -j ${threads}
      ${ctest_step}
      # Move-prefix test: a second install at a different prefix must be just as
      # functional and carry no build-prefix references. The manifest's
      # build_prefix field and the kept deal.II provenance logs are deliberate
      # records of the build environment and exempt, matching the installer's
      # own rewrite exclusions.
      bash /src/install-diffrg-deps.sh --file /dist/${tarname} \
          --prefix /work/diffrg2 --skip-cpu-check
      if grep -rI --exclude=BUNDLE_MANIFEST.json --exclude='*.log' -l /opt/diffrg /work/diffrg2/bundled; then
        echo 'move-prefix test FAILED: build prefix survived relocation'; exit 1
      fi
    " >>"${log}" 2>&1; then
    results[${distro}]="PASS"
  else
    results[${distro}]="FAIL"
    overall=1
  fi
  # Docker runs leave root-owned files in the workdir; chown them via a tiny
  # container run so cleanup works on failure paths too.
  if [[ ${runtime} == docker ]]; then
    docker run --rm -v "${workdir}:/work" "${image}" chmod -R a+rwX /work >/dev/null 2>&1 || true
  fi
  rm -rf "${workdir}" 2>/dev/null || true
done

echo
echo "=== Results for ${tarname}:"
for distro in ${distros}; do
  echo "  ${distro}: ${results[${distro}]:-SKIPPED}"
done

if [[ ${overall} -eq 0 ]]; then
  [[ ${run_tests} -eq 1 ]] && echo "tested-with-ctest" >"${tarball}.tested" \
    || echo "tested-build-only" >"${tarball}.tested"
  echo "All distros PASSED. Stamp written: ${tarball}.tested"
else
  rm -f "${tarball}.tested"
  echo "FAILURES -- see logs in ${logdir}/" >&2
fi
exit ${overall}
