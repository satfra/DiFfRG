#!/bin/bash
# ##############################################################################
# Print the hash of everything a dependency bundle of one variant is built from.
#
# build-release.sh records it in the bundle's BUNDLE_MANIFEST.json
# (deps_inputs_hash); CI compares it against the pinned bundle to decide whether
# that bundle still matches the checkout or a fresh one has to be built.
#
# The inputs are the files that reach the bundle (tracked, or new and not
# ignored): the superbuild (superbuild.cmake), dependencies/, patches/, and the
# variant's recipe and post-processing. PETSc counts only for the -openmpi variants, which are the
# only ones that build it. Audits that cannot change a bundle's content
# (check-linkage.sh, the test images) are left out. Hashed from the working tree,
# so uncommitted changes count.
#
# Usage: deps-inputs-hash.sh <variant> [repo]
#   variant  e.g. linux-x86_64-v3-cpu, linux-x86_64-v3-cuda12-openmpi, macos-arm64-cpu
#   repo     default: the repository holding this script
# ##############################################################################
set -euo pipefail

variant="${1:?Usage: deps-inputs-hash.sh <variant> [repo]}"
repo="${2:-$(cd -- "$(dirname "$0")/../.." >/dev/null 2>&1 && pwd -P)}"
cd "${repo}"

base="${variant%-openmpi}"
inputs=(superbuild.cmake dependencies patches)
[[ ${variant} == *-openmpi ]] || inputs+=(':(exclude)dependencies/petsc')
case "${base}" in
macos-*)
  inputs+=("containers/release/${base}.sh" containers/release/postprocess-bundle-macos.sh)
  ;;
*)
  [[ -f containers/release/${base}.Dockerfile ]] || {
    echo "Unknown variant '${variant}' (no containers/release/${base}.Dockerfile)" >&2
    exit 1
  }
  inputs+=("containers/release/${base}.Dockerfile" containers/release/postprocess-bundle.sh)
  ;;
esac

# macOS has shasum, not sha256sum; both print "<hash>  <path>".
sha256() { if command -v sha256sum >/dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }

# Path and content of every input file, in a fixed order; a tracked file deleted in the working tree is left out.
git ls-files -z --cached --others --exclude-standard -- "${inputs[@]}" | LC_ALL=C sort -z | while IFS= read -r -d '' f; do
  if [[ -f ${f} ]]; then sha256 "${f}"; fi
done | sha256 | cut -d' ' -f1
