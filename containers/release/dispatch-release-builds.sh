#!/bin/bash
# ##############################################################################
# Trigger the release-deps workflows for every dependency bundle variant and
# collect the results in one draft release deps-v<version>.
#
# Variants: linux-x86_64-v3-{cpu,cuda12}[-openmpi] and macos-arm64-cpu.
#
# The draft is created here, before dispatching, so that the five finalize
# jobs all upload into the same release instead of racing to create it.
# Publishing the draft stays a manual step (Releases UI).
#
# Usage: dispatch-release-builds.sh -v <bundle_version> [-r <ref>] [-n]
#   -v <version>  bundle version X.Y.Z (becomes tag deps-vX.Y.Z)
#   -r <ref>      branch or tag the workflows build from (default: the current
#                 branch, which must be pushed and match the local HEAD)
#   -n            dry run: print the gh commands without running them
# ##############################################################################
set -euo pipefail

version=''
ref=''
dry_run=false

while getopts v:r:n flag; do
  case "${flag}" in
  v) version=${OPTARG} ;;
  r) ref=${OPTARG} ;;
  n) dry_run=true ;;
  *)
    echo "Usage: $0 -v <bundle_version> [-r <ref>] [-n]" >&2
    exit 1
    ;;
  esac
done

[[ ${version} =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || {
  echo "A bundle version X.Y.Z is required: $0 -v 1.0.0" >&2
  exit 1
}

run() {
  echo "+ $*"
  ${dry_run} || "$@"
}

gh auth status >/dev/null
git fetch --quiet origin

if [[ -z ${ref} ]]; then
  ref="$(git rev-parse --abbrev-ref HEAD)"
  [[ ${ref} != HEAD ]] || {
    echo "Detached HEAD; pass the branch or tag to build with -r." >&2
    exit 1
  }
  if [[ "$(git rev-parse HEAD)" != "$(git rev-parse "origin/${ref}" 2>/dev/null)" ]]; then
    echo "Local ${ref} differs from origin/${ref}; push it first (the workflows build the remote ref)." >&2
    exit 1
  fi
fi
sha="$(git rev-parse "origin/${ref}" 2>/dev/null || git rev-parse "refs/tags/${ref}^{commit}")"

tag="deps-v${version}"
if is_draft="$(gh release view "${tag}" --json isDraft --jq .isDraft 2>/dev/null)"; then
  [[ ${is_draft} == true ]] || {
    echo "Release ${tag} is already published; pick a new version." >&2
    exit 1
  }
  echo "Draft ${tag} exists; the workflows will add their assets to it (--clobber)."
else
  run gh release create "${tag}" --draft --target "${sha}" \
    --title "DiFfRG dependency bundle ${version}" \
    --notes "Draft -- assets uploaded by the release-deps workflows."
fi

echo "Dispatching the release-deps workflows on ${ref} (${sha:0:9}) for ${tag}:"
for mpi in none openmpi; do
  run gh workflow run release-deps-linux.yml --ref "${ref}" \
    -f bundle_version="${version}" -f mpi="${mpi}" -f draft_release=true
  run gh workflow run release-deps-linux-cuda.yml --ref "${ref}" \
    -f bundle_version="${version}" -f mpi="${mpi}" -f draft_release=true
done
run gh workflow run release-deps-macos.yml --ref "${ref}" \
  -f bundle_version="${version}" -f run_smoke=quick-tests -f draft_release=true

cat <<EOF

Follow the runs:   gh run list --workflow release-deps-linux.yml  (likewise -cuda, -macos)
Draft release:     gh release view ${tag} --web
The CUDA bundles are only build-validated in CI; run
  containers/release/test-tarball.sh -g -f <tarball>
on a GPU machine before publishing the draft.
EOF
