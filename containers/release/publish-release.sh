#!/bin/bash
# ##############################################################################
# Publish a tested DiFfRG dependency bundle to GitHub Releases.
#
# Preconditions (all checked):
#   - the tarball exists together with its .sha256 and the .tested stamp from
#     containers/release/test-tarball.sh,
#   - the working tree is clean (the tag must point at the code that built it),
#   - `gh auth status` succeeds.
#
# Creates the GitHub release deps-v<version> (and with it the tag) with the
# tarball, checksum, and manifest attached. The variants of one version share
# that release: publishing a further variant uploads its assets into it.
#
# Usage: publish-release.sh -v <bundle_version> [-V <variant>] [-o <distdir>]
#   -v <version>  bundle version, e.g. 1.0.0 (as passed to build-release.sh)
#   -V <variant>  bundle variant (default: linux-x86_64-v3-cpu)
#   -o <distdir>  where the tarball lives (default: containers/release/dist)
# ##############################################################################
set -euo pipefail

scriptpath="$(
  cd -- "$(dirname "$0")" >/dev/null 2>&1
  pwd -P
)"
repo="$(cd -- "${scriptpath}/../.." >/dev/null 2>&1 && pwd -P)"

variant="linux-x86_64-v3-cpu"
version=''
distdir="${scriptpath}/dist"

while getopts v:V:o: flag; do
  case "${flag}" in
  v) version=${OPTARG} ;;
  V) variant=${OPTARG} ;;
  o) distdir=${OPTARG} ;;
  *)
    echo "Unknown flag." >&2
    exit 1
    ;;
  esac
done

[[ -n ${version} ]] || {
  echo "A bundle version is required: publish-release.sh -v 1.0.0" >&2
  exit 1
}

name="diffrg-deps-${version}-${variant}"
tarball="${distdir}/${name}.tar.zst"
tag="deps-v${version}"

for f in "${tarball}" "${tarball}.sha256"; do
  [[ -f ${f} ]] || {
    echo "Missing ${f} -- run build-release.sh -v ${version} -V ${variant} first." >&2
    exit 1
  }
done
[[ -f "${tarball}.tested" ]] || {
  echo "Missing ${tarball}.tested -- run test-tarball.sh -f ${tarball} first." >&2
  exit 1
}
if [[ "$(cat "${tarball}.tested")" != "tested-with-ctest" ]]; then
  echo "WARNING: the tarball only passed a build-only validation (no ctest run)." >&2
  read -rp "Publish anyway? [y/N] " ans
  [[ ${ans} == y || ${ans} == Y ]] || exit 1
fi

if [[ -n "$(git -C "${repo}" status --porcelain)" ]]; then
  echo "The working tree is not clean; commit or stash before tagging a release." >&2
  exit 1
fi
gh auth status >/dev/null

# Extract a manifest copy for attachment + release-note fields.
workdir="$(mktemp -d)"
trap 'rm -rf "${workdir}"' EXIT
tar --zstd -xf "${tarball}" -C "${workdir}" "${name}/BUNDLE_MANIFEST.json"
manifest="${workdir}/${name}/BUNDLE_MANIFEST.json"
# Named per variant: the variants of a version share one release.
cp "${manifest}" "${workdir}/${name}.manifest.json"

field() { grep -oE "\"$1\": *\"[^\"]*\"" "${manifest}" | head -1 | sed -E 's/.*: *"([^"]*)"/\1/'; }
glibc_floor="$(field glibc_floor)"
dep_list="$(sed -n '/"dependency_versions"/,/}/p' "${manifest}" |
  grep -oE '"[A-Za-z0-9_.+-]+": "[0-9.]+"' | sed 's/"//g; s/^/- /')"

notes="${workdir}/notes.md"
sed -e "s|@VERSION@|${version}|g" \
  -e "s|@VARIANT@|${variant}|g" \
  -e "s|@GLIBC_FLOOR@|${glibc_floor}|g" \
  -e "s|@SHA256@|$(cut -d' ' -f1 "${tarball}.sha256")|g" \
  "${scriptpath}/RELEASE_NOTES.md.in" >"${notes}"
printf '\n### Bundled dependency versions\n\n%s\n' "${dep_list}" >>"${notes}"

assets=("${tarball}" "${tarball}.sha256" "${workdir}/${name}.manifest.json")
if gh release view "${tag}" >/dev/null 2>&1; then
  echo "Release ${tag} exists; adding ${variant}..."
  gh release upload "${tag}" "${assets[@]}"
  echo "Release notes were written for the first variant published; add a line"
  echo "for ${variant} by hand if needed: sha256 $(cut -d' ' -f1 "${tarball}.sha256")"
else
  # gh creates the tag on the remote as part of the release; doing it in one
  # step avoids a stray pushed tag when the release upload fails.
  echo "Creating release ${tag}..."
  gh release create "${tag}" "${assets[@]}" \
    --target "$(git -C "${repo}" rev-parse HEAD)" \
    --title "DiFfRG dependency bundle ${version}" \
    --notes-file "${notes}"
fi
git -C "${repo}" fetch origin "refs/tags/${tag}:refs/tags/${tag}" 2>/dev/null || true

echo "Published ${tag}."
