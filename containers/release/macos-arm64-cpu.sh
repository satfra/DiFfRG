#!/bin/bash
# ##############################################################################
# Build the relocatable DiFfRG dependency bundle for Apple Silicon
# (diffrg-deps-<version>-macos-arm64-cpu.tar.zst). EXPERIMENTAL.
#
# Runs natively on an arm64 Mac -- locally or on a GitHub macos-14+ runner
# (driven by .github/workflows/release-deps-macos.yml). No containers: macOS
# builds happen straight on the host, at the canonical prefix /opt/diffrg
# (must exist and be writable: sudo mkdir -p /opt/diffrg && sudo chown $(id -u) /opt/diffrg).
#
# Requirements: Xcode command line tools, and from Homebrew: gcc (for
# gfortran), gsl, gnu-tar, zstd.
#
# Usage: macos-arm64-cpu.sh -v <bundle_version> [-j <threads>] [-o <outdir>]
# ##############################################################################
set -euo pipefail

scriptpath="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 && pwd -P)"
repo="$(cd -- "${scriptpath}/../.." >/dev/null 2>&1 && pwd -P)"

variant="macos-arm64-cpu"
version=''
threads=3
outdir="${scriptpath}/dist"
build_dir="${TMPDIR:-/tmp}/diffrg-release-build"

while getopts v:j:o: flag; do
  case "${flag}" in
  v) version=${OPTARG} ;;
  j) threads=${OPTARG} ;;
  o) outdir=${OPTARG} ;;
  *) echo "Unknown flag." >&2; exit 1 ;;
  esac
done
[[ -n ${version} ]] || { echo "A bundle version is required: -v 1.0.0" >&2; exit 1; }

[[ "$(uname -s)-$(uname -m)" == Darwin-arm64 ]] || { echo "This script must run on an arm64 Mac." >&2; exit 1; }
for tool in cmake git gfortran gtar zstd shasum; do
  command -v "${tool}" >/dev/null || { echo "'${tool}' is required (brew install gcc gnu-tar zstd)." >&2; exit 1; }
done
[[ -d /opt/diffrg && -w /opt/diffrg ]] \
  || { echo "/opt/diffrg must exist and be writable (sudo mkdir -p /opt/diffrg && sudo chown \$(id -u) /opt/diffrg)." >&2; exit 1; }

# All M-series CPUs share the arm64e-adjacent baseline Apple clang targets by
# default, so no -march/-mcpu pinning is needed for portability across M1-M4.
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}"

echo "Configuring the superbuild (deps only, prefix /opt/diffrg)..."
cmake -S "${repo}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/diffrg \
  -DGPU=OFF -DMPI=OFF -DDiFfRG_DOCUMENTATION=OFF \
  -DMARCH=none \
  -DBUILD_BOOST=ON -DBUILD_TBB=ON -DBUILD_HDF5=ON -DBUILD_SUNDIALS=ON \
  -DDEAL_II_GSL=OFF \
  -DUSE_CCACHE=OFF \
  -DBUILD_JOBS=$((2 * threads)) \
  -DDEALII_MAX_JOBS="${threads}" \
  -DPETSC_MAX_JOBS="${threads}"
cmake --build "${build_dir}" \
  --target general_dep deal.II_dep kokkos_dep autodiff_dep \
  -j "${threads}"

GIT_SHA="$(git -C "${repo}" rev-parse HEAD 2>/dev/null || echo unknown)" \
  DEPS_INPUTS_HASH="$(bash "${scriptpath}/deps-inputs-hash.sh" "${variant}" "${repo}")" \
  bash "${scriptpath}/postprocess-bundle-macos.sh" \
  /opt/diffrg/bundled "${repo}" "${version}" "${variant}" "${MACOSX_DEPLOYMENT_TARGET}"

name="diffrg-deps-${version}-${variant}"
mkdir -p "${outdir}"
staging="$(mktemp -d)"
trap 'rm -rf "${staging}"' EXIT
mkdir -p "${staging}/${name}"
cp -a /opt/diffrg/bundled "${staging}/${name}/bundled"
cp /opt/diffrg/bundled/BUNDLE_MANIFEST.json "${staging}/${name}/"
gtar -C "${staging}" --sort=name --owner=0 --group=0 --numeric-owner \
  -cf - "${name}" | zstd -19 -T0 -o "${outdir}/${name}.tar.zst"
(cd "${outdir}" && shasum -a 256 "${name}.tar.zst" > "${name}.tar.zst.sha256")

echo "Built ${outdir}/${name}.tar.zst"
