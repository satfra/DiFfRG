#!/usr/bin/env bash
# macOS (arm64) counterpart of postprocess-bundle.sh. EXPERIMENTAL.
#
# Differences from the Linux pipeline: dylibs keep their absolute /opt/diffrg
# install names (the installer rewrites them with install_name_tool and
# ad-hoc re-signs at install time -- on arm64 macOS any binary edit
# invalidates the code signature); there is no ISA guard (all M-series share
# the compiler's arm64 baseline) and no glibc audit (MACOSX_DEPLOYMENT_TARGET
# plays that role and is recorded in the manifest).
#
# Usage: postprocess-bundle-macos.sh <bundle-dir> <repo-src> <version> <variant> <min-macos>
set -euo pipefail

BUNDLE="$1"
SRC="$2"
BUNDLE_VERSION="$3"
VARIANT="$4"
MIN_MACOS="$5"

fail() { echo "postprocess-macos: FAIL: $*" >&2; exit 1; }
note() { echo "postprocess-macos: $*"; }

[ -d "$BUNDLE/lib" ] || fail "$BUNDLE/lib does not exist"

# ---------------------------------------------------------------- 1. verify --
mkdir -p "$BUNDLE/share/DiFfRG"
cp "$SRC/DiFfRG/cmake/verify_install.cmake" "$BUNDLE/share/DiFfRG/verify_install.cmake"

# ----------------------------------------- 2. SDK/system lib paths -> flags --
# deal.II may export absolute SDK stub paths, which are runner-specific
# (Xcode versions move); rewrite them to plain linker flags.
[ -f "$BUNDLE/lib/cmake/deal.II/deal.IITargets.cmake" ] \
  || fail "deal.IITargets.cmake not found -- bundle layout changed?"
for f in "$BUNDLE"/lib/cmake/deal.II/deal.IITargets.cmake \
         "$BUNDLE"/lib/cmake/deal.II/deal.IIConfig.cmake; do
  [ -f "$f" ] || continue
  # Three shapes of SDK path: framework references become -framework flags,
  # SDK libs become -l flags, and SDK include dirs are dropped outright (every
  # consumer compiler supplies its own SDK sysroot). Then tidy the ;-separated
  # lists the removals may have dented.
  sed -i '' -E \
    -e 's#[^;"]*/SDKs/[^;"]*/System/Library/Frameworks/([A-Za-z0-9_]+)\.framework#-framework \1#g' \
    -e 's#(/Applications/Xcode[^;"]*|/Library/Developer/CommandLineTools[^;"]*)/usr/lib/lib([A-Za-z0-9_+.-]+)\.(tbd|dylib)#-l\2#g' \
    -e 's#(/Applications/Xcode[^;"]*|/Library/Developer/CommandLineTools[^;"]*)/usr/include##g' \
    -e 's#;;+#;#g' -e 's#";#"#g' -e 's#;"#"#g' \
    "$f"
done
if grep -nE '(/Applications/Xcode|/Library/Developer/CommandLineTools)' \
    "$BUNDLE"/lib/cmake/deal.II/deal.II*.cmake; then
  fail "SDK paths survived the rewrite (above)"
fi

# ------------------------------------------------------------------ 3. strip --
find "$BUNDLE/lib" -name '*.dylib' -type f -exec strip -x {} \; 2>/dev/null || true

# ----------------------------------------------------------- 4. arch + link --
# Every dylib must be arm64, resolve only system libraries or bundle-internal
# paths, and never a Homebrew path (those differ per machine).
while IFS= read -r dylib; do
  lipo -archs "$dylib" | grep -q arm64 || fail "$dylib is not arm64"
  while IFS= read -r dep; do
    case "$dep" in
    /usr/lib/* | /System/Library/* | "$BUNDLE"/* | /opt/diffrg/* | @*) ;;
    /opt/homebrew/*) fail "$dylib links a Homebrew library ($dep) -- not portable" ;;
    *) fail "$dylib links unexpected path: $dep" ;;
    esac
  done < <(otool -L "$dylib" | awk 'NR>1{print $1}')
done < <(find "$BUNDLE/lib" -name '*.dylib' -type f)
note "arch and linkage audit passed"

# -------------------------------------------------- 5. residual path audit --
# (bash 3.2-compatible: macOS ships no mapfile)
_hits="$( { find "$BUNDLE/lib/cmake" "$BUNDLE/lib/pkgconfig" "$BUNDLE/cmake" \
      "$BUNDLE/bin" "$BUNDLE/share/DiFfRG" -type f 2>/dev/null;
    find "$BUNDLE" -maxdepth 1 -type f ! -name '*.log'; } |
  xargs grep -lE '(^|["=[:space:];:])(/Users|/tmp|/private/tmp)/' 2>/dev/null || true)"
if [ -n "$_hits" ]; then
  echo "$_hits"
  fail "configuration files reference build-time paths (above)"
fi
note "no stray build-time paths in configuration files"

# --------------------------------------------------------------- 6. manifest --
BOOST_VER="$(grep -m1 -oE 'DiFfRG_PINNED_BOOST_VERSION "[0-9.]+"' "$BUNDLE/DiFfRG_bundled_config.cmake" | grep -oE '[0-9.]+' || echo unknown)"
GIT_SHA="${GIT_SHA:-$(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo unknown)}"
DIFFRG_VERSION="$(tr -d '[:space:]' < "$SRC/VERSION" 2>/dev/null || echo unknown)"

dep_versions() {
  local first=1
  for cv in "$BUNDLE"/lib/cmake/*/*[Cc]onfig[Vv]ersion.cmake \
            "$BUNDLE"/lib/cmake/*/*config-version.cmake; do
    [ -f "$cv" ] || continue
    local name ver
    name="$(basename "$(dirname "$cv")")"
    case "$name" in boost_*) continue ;; esac
    ver="$(grep -m1 -oE 'PACKAGE_VERSION "?[0-9][0-9.]*"?' "$cv" | grep -oE '[0-9][0-9.]*' || true)"
    [ -n "$ver" ] || continue
    [ $first -eq 1 ] || printf ',\n'
    first=0
    printf '    "%s": "%s"' "$name" "$ver"
  done
  printf '\n'
}

cat > "$BUNDLE/BUNDLE_MANIFEST.json" <<EOF
{
  "name": "diffrg-deps-${BUNDLE_VERSION}-${VARIANT}",
  "bundle_version": "${BUNDLE_VERSION}",
  "os": "macos",
  "arch": "arm64",
  "isa": "arm64",
  "accel": "cpu",
  "mpi": "none",
  "diffrg_version": "${DIFFRG_VERSION}",
  "diffrg_git_sha": "${GIT_SHA}",
  "deps_inputs_hash": "${DEPS_INPUTS_HASH:-unknown}",
  "build_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "build_prefix": "/opt/diffrg",
  "min_macos": "${MIN_MACOS}",
  "compiler": "$(cc --version | head -1)",
  "builder_cxx": "$(command -v c++ || true)",
  "builder_cc": "$(command -v cc || true)",
  "builder_fc": "$(command -v gfortran || true)",
  "boost_version": "${BOOST_VER}",
  "dependency_versions": {
$(dep_versions)
  }
}
EOF
note "manifest written to $BUNDLE/BUNDLE_MANIFEST.json"
note "OK"
