#!/bin/bash
# ##############################################################################
# Build the relocatable DiFfRG dependency bundle tarball for a release.
#
# Runs the release Dockerfile for the requested variant and extracts the
# resulting tarball + sha256 into the output directory. The Dockerfile itself
# performs all relocation fixups and hard audits (ISA guard, symbol-version
# caps, linkage self-containment); a tarball only appears here if they passed.
#
# Usage: build-release.sh -v <bundle_version> [-V <variant>] [-j <threads>] [-o <outdir>]
#   -v <version>  bundle version, e.g. 1.0.0 (required; becomes the deps-v<version> tag)
#   -V <variant>  bundle variant: <base>[-openmpi], where <base> has a
#                 <base>.Dockerfile in this directory (default: linux-x86_64-v3-cpu;
#                 also linux-x86_64-v3-cuda12 and the -openmpi variants of both)
#   -j <threads>  build threads (default: 6 -- deal.II TUs are RAM-hungry)
#   -o <outdir>   where to place the tarball (default: containers/release/dist)
#
# DOCKER_NETWORK=host builds on the host network -- needed where a VPN client
# blocks docker's bridge network (downloads then fail with DNS errors).
#
# Test the result with containers/release/test-tarball.sh, publish it with
# containers/release/publish-release.sh.
# ##############################################################################
set -euo pipefail

scriptpath="$(
  cd -- "$(dirname "$0")" >/dev/null 2>&1
  pwd -P
)"
repo="$(cd -- "${scriptpath}/../.." >/dev/null 2>&1 && pwd -P)"

variant="linux-x86_64-v3-cpu"
version=''
threads=6
outdir="${scriptpath}/dist"

while getopts v:V:j:o: flag; do
  case "${flag}" in
  v) version=${OPTARG} ;;
  V) variant=${OPTARG} ;;
  j) threads=${OPTARG} ;;
  o) outdir=${OPTARG} ;;
  *)
    echo "Unknown flag." >&2
    exit 1
    ;;
  esac
done

# An MPI suffix selects the base variant's Dockerfile with its mpi build arg.
base="${variant}"
mpi=none
case "${variant}" in
*-openmpi) base="${variant%-openmpi}" mpi=openmpi ;;
esac
[[ -f "${scriptpath}/${base}.Dockerfile" ]] \
  || { echo "Unknown variant '${variant}' (no ${scriptpath}/${base}.Dockerfile)" >&2; exit 1; }

if [[ -z ${version} ]]; then
  echo "A bundle version is required: build-release.sh -v 1.0.0" >&2
  exit 1
fi
if ! [[ ${version} =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Bundle version must be X.Y.Z, got '${version}'" >&2
  exit 1
fi

name="diffrg-deps-${version}-${variant}"
image="diffrg-deps-release:${version}-${variant}"

echo "Building ${name}.tar.zst with ${threads} threads (context: ${repo})"
docker buildx build --load \
  --network "${DOCKER_NETWORK:-default}" \
  -t "${image}" \
  -f "${scriptpath}/${base}.Dockerfile" \
  --build-arg "mpi=${mpi}" \
  --build-arg "threads=${threads}" \
  --build-arg "bundle_version=${version}" \
  --build-arg "git_sha=$(git -C "${repo}" rev-parse HEAD 2>/dev/null || echo unknown)" \
  --build-arg "deps_inputs_hash=$(bash "${scriptpath}/deps-inputs-hash.sh" "${variant}" "${repo}")" \
  --progress=plain \
  "${repo}"

mkdir -p "${outdir}"
cid="$(docker create "${image}")"
trap 'docker rm -f "${cid}" >/dev/null' EXIT
docker cp "${cid}:/dist/${name}.tar.zst" "${outdir}/"
docker cp "${cid}:/dist/${name}.tar.zst.sha256" "${outdir}/"
zstd -t "${outdir}/${name}.tar.zst"
(cd "${outdir}" && sha256sum -c "${name}.tar.zst.sha256")

# GitHub Releases caps individual assets at 2 GiB; refuse anything close.
size=$(stat -c %s "${outdir}/${name}.tar.zst")
if ((size > 1900 * 1024 * 1024)); then
  echo "Tarball is $((size / 1024 / 1024)) MiB -- too close to the 2 GiB GitHub asset limit." >&2
  exit 1
fi

echo
echo "Built ${outdir}/${name}.tar.zst ($((size / 1024 / 1024)) MiB)"
echo "Next: containers/release/test-tarball.sh -f ${outdir}/${name}.tar.zst"
