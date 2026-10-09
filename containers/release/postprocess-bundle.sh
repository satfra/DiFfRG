#!/usr/bin/env bash
# Post-process a freshly built dependency bundle into a relocatable state and
# audit it. Runs inside the release builder container (see the Dockerfiles in
# this directory); every check is a hard failure -- a bundle that does not pass
# is not shipped.
#
# Usage: postprocess-bundle.sh <bundle-dir> <repo-src> <bundle-version> <variant> <march> <glibc-floor>
#   bundle-dir     the installed dependency tree (e.g. /opt/diffrg/bundled)
#   repo-src       the DiFfRG repository checkout (for verify_install.cmake, git SHA)
#   bundle-version e.g. 1.0.0
#   variant        e.g. linux-x86_64-v3-cpu, linux-x86_64-v3-cuda12-openmpi
#   march          the -march value the bundle was compiled for (e.g. x86-64-v3)
#   glibc-floor    maximum allowed GLIBC_* symbol version (e.g. 2.34)
set -euo pipefail

BUNDLE="$1"
SRC="$2"
BUNDLE_VERSION="$3"
VARIANT="$4"
MARCH="$5"
GLIBC_FLOOR="$6"

# libstdc++ ceilings of the el9 baseline (gcc 11). gcc-toolset links newer
# symbols from its static libstdc++_nonshared.a, so nothing above these may
# leak into the shared objects' undefined symbols.
GLIBCXX_CEIL="3.4.29"
CXXABI_CEIL="1.3.13"
# libgfortran ceiling: whatever the el9 runtime libgfortran.so.5 defines (MUMPS
# and ScaLAPACK are Fortran). Read off the builder's copy, which is el9's own.
GFORTRAN_CEIL="$(objdump -T /usr/lib64/libgfortran.so.5 2>/dev/null | grep -oE 'GFORTRAN_[0-9.]+' | sed 's/GFORTRAN_//' | sort -uV | tail -1 || true)"

HERE="$(cd "$(dirname "$0")" && pwd)"

# Variants are <os>-<arch>-<isa>-<accel>[-<mpi>]: accel is cpu, cuda12, ...;
# the optional MPI suffix names the MPI implementation the bundle links.
MPI_IMPL=none
case "$VARIANT" in
  *-openmpi) MPI_IMPL=openmpi ;;
esac
BASE_VARIANT="${VARIANT%-"$MPI_IMPL"}"
ACCEL="${BASE_VARIANT##*-}"

fail() { echo "postprocess: FAIL: $*" >&2; exit 1; }
note() { echo "postprocess: $*"; }

[ -d "$BUNDLE/lib" ] || fail "$BUNDLE/lib does not exist"

# MPI variants: the builder's Open MPI locations. deal.II and PETSc record them
# by absolute path; the manifest hands them to the installer, which rewrites
# them to the host's Open MPI (the same scheme as the CUDA toolkit root below).
BUILDER_MPI_LIBDIR=''
BUILDER_MPI_INCDIR=''
BUILDER_MPI_BINDIR=''
if [ "$MPI_IMPL" = openmpi ]; then
  command -v mpicc >/dev/null || fail "mpicc not on PATH for an MPI variant"
  _ompi_version="$(mpicc --showme:version 2>&1 || true)"
  case "$_ompi_version" in
    *"Open MPI"*) ;;
    *) fail "mpicc is not Open MPI: $_ompi_version" ;;
  esac
  for d in $(mpicc --showme:libdirs); do
    if [ -e "$d/libmpi.so" ]; then BUILDER_MPI_LIBDIR="$d"; break; fi
  done
  for d in $(mpicc --showme:incdirs); do
    if [ -e "$d/mpi.h" ]; then BUILDER_MPI_INCDIR="$d"; break; fi
  done
  BUILDER_MPI_BINDIR="$(dirname "$(command -v mpicc)")"
  [ -n "$BUILDER_MPI_LIBDIR" ] || fail "no libmpi.so in mpicc --showme:libdirs"
  [ -n "$BUILDER_MPI_INCDIR" ] || fail "no mpi.h in mpicc --showme:incdirs"
  note "Open MPI: lib $BUILDER_MPI_LIBDIR, include $BUILDER_MPI_INCDIR, bin $BUILDER_MPI_BINDIR"
fi

# The superbuild pins CMAKE_INSTALL_LIBDIR=lib, so lib64/ should not exist;
# every object pass still covers it defensively in case a dependency ever
# ignores the pin again.
LIBDIRS=("$BUNDLE/lib")
[ -d "$BUNDLE/lib64" ] && LIBDIRS+=("$BUNDLE/lib64")

# ---------------------------------------------------------------- 1. verify --
# Ship the standalone dependency checker inside the bundle so the installer
# can run it without a DiFfRG checkout.
mkdir -p "$BUNDLE/share/DiFfRG"
cp "$SRC/DiFfRG/cmake/verify_install.cmake" "$BUNDLE/share/DiFfRG/verify_install.cmake"

# ------------------------------------------- 2. system lib paths -> -l flags --
# deal.II exports the build machine's system libraries as absolute paths in its
# CMake config; those differ per distro (lib64 vs multiarch), so rewrite them to
# plain -l flags the consumer's linker resolves from default directories.
[ -f "$BUNDLE/lib/cmake/deal.II/deal.IITargets.cmake" ] \
  || fail "deal.IITargets.cmake not found -- bundle layout changed?"
for f in "$BUNDLE"/lib/cmake/deal.II/deal.IITargets.cmake \
         "$BUNDLE"/lib/cmake/deal.II/deal.IIConfig.cmake; do
  [ -f "$f" ] || continue
  sed -i -E 's#/usr/(lib64|lib)(/[A-Za-z0-9_.-]+-linux-gnu[A-Za-z0-9_.-]*)?/lib([A-Za-z0-9_+.-]+)\.(so|a)#-l\3#g' "$f"
done
# Open MPI's libraries stay absolute: they live outside the default linker
# path on EL/Debian, and the installer points them at the host's copy.
_mpi_lib_strip=''
[ -n "$BUILDER_MPI_LIBDIR" ] && _mpi_lib_strip="s#${BUILDER_MPI_LIBDIR}/lib[A-Za-z0-9_+.-]+\.so##g"
_leaked=0
for f in "$BUNDLE"/lib/cmake/deal.II/deal.II*.cmake; do
  if sed -E "$_mpi_lib_strip" "$f" | grep -nE '/usr/(lib64|lib)[^;"]*\.(so|a)'; then
    echo "  in $f" >&2
    _leaked=1
  fi
done
[ "$_leaked" -eq 0 ] || fail "absolute system library paths survived the -l rewrite (above)"

# CUDA variants: deal.II bakes toolkit paths (include dir, libcudart, the
# link-time driver stub) under both /usr/local/cuda and its versioned resolve.
# Canonicalize everything to the unversioned root here; the installer then
# rewrites that single form to wherever the consumer's toolkit lives (e.g.
# /opt/cuda on Arch).
BUILDER_CUDA_ROOT=''
if [ "$ACCEL" != cpu ]; then
  BUILDER_CUDA_ROOT=/usr/local/cuda
  _cuda_versioned="$(readlink -f /usr/local/cuda 2>/dev/null || true)"
  if [ -n "$_cuda_versioned" ] && [ "$_cuda_versioned" != "$BUILDER_CUDA_ROOT" ]; then
    for f in "$BUNDLE"/lib/cmake/deal.II/deal.II*.cmake; do
      [ -f "$f" ] || continue
      sed -i "s#${_cuda_versioned}#${BUILDER_CUDA_ROOT}#g" "$f"
    done
  fi
  note "CUDA toolkit paths canonicalized to ${BUILDER_CUDA_ROOT}"
fi

# MPI variants: PETSc's petscsys.h #errors when compiled against an Open MPI
# major newer than the one it was configured with. Open MPI keeps the C ABI
# from 4.0/4.1 to 5.0 (docs.open-mpi.org, "Version numbers and compatibility"),
# and MUMPS' Fortran uses default-size integers, so a 4.1-built bundle is sound
# on Open MPI 5 hosts: drop exactly that check. The one rejecting a host Open
# MPI *older* than the build's stays.
if [ "$MPI_IMPL" = openmpi ]; then
  _petscsys="$BUNDLE/include/petscsys.h"
  [ -f "$_petscsys" ] || fail "$_petscsys not found -- MPI variant without PETSc?"
  python3 - "$_petscsys" <<'PY' || fail "could not relax PETSc's Open MPI major-version check"
import sys
path = sys.argv[1]
src = open(path).read()
check = ('  #elif PETSC_PKG_OPENMPI_VERSION_LT(OMPI_MAJOR_VERSION, 0, 0)\n'
         '    #error "PETSc was configured with one Open MPI mpi.h version but now appears to be compiling using a newer major Open MPI mpi.h version"\n')
if src.count(check) != 1 or 'using an older Open MPI mpi.h version' not in src:
    sys.exit(1)
open(path, 'w').write(src.replace(check,
    '  /* DiFfRG bundle: newer Open MPI majors accepted (C ABI is stable 4.x -> 5.x) */\n'))
PY
  note "PETSc accepts Open MPI >= its build version, newer majors included"
fi

# ------------------------------------------------------------------ 3. strip --
find "${LIBDIRS[@]}" -name '*.so*' -type f -exec strip --strip-unneeded {} +
find "${LIBDIRS[@]}" -name '*.a' -type f -exec strip -g {} +

# ------------------------------------------------------------------ 4. rpath --
# Make every shared object find its siblings relative to itself, whichever of
# lib/ and lib64/ each side lives in; after this the tarball needs no patchelf
# at install time.
RPATH='$ORIGIN:$ORIGIN/../lib:$ORIGIN/../lib64'
while IFS= read -r so; do
  file -b "$so" | grep -q ELF || continue
  patchelf --set-rpath "$RPATH" "$so"
  readelf -d "$so" | grep -q 'RUNPATH.*\$ORIGIN' \
    || fail "RUNPATH not set on $so"
done < <(find "${LIBDIRS[@]}" -name '*.so*' -type f)

# ------------------------------------------------------------------ 5. prune --
rm -rf "$BUNDLE/share/doc" "$BUNDLE/share/man"
find "$BUNDLE" -name '*.log' ! -name 'detailed.log' ! -name 'summary.log' -delete

# -------------------------------------------------------------- 6. ISA guard --
"$HERE/check-isa.sh" "$BUNDLE" "$MARCH"

# ------------------------------------------------------- 7. symbol versions --
# No undefined symbol may require a glibc newer than the floor or a libstdc++
# newer than the el9 baseline -- either would break the "runs on any distro
# with glibc >= floor" promise.
ver_gt() { [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -1)" != "$2" ]; }
while IFS= read -r so; do
  file -b "$so" | grep -q ELF || continue
  while IFS= read -r ver; do
    case "$ver" in
      GLIBC_*)   ver_gt "${ver#GLIBC_}" "$GLIBC_FLOOR"   && fail "$so needs $ver (> $GLIBC_FLOOR)" ;;
      GLIBCXX_*) ver_gt "${ver#GLIBCXX_}" "$GLIBCXX_CEIL" && fail "$so needs $ver (> $GLIBCXX_CEIL)" ;;
      CXXABI_*)  ver_gt "${ver#CXXABI_}" "$CXXABI_CEIL"  && fail "$so needs $ver (> $CXXABI_CEIL)" ;;
      GFORTRAN_*)
        [ -n "$GFORTRAN_CEIL" ] || fail "$so needs $ver, but no el9 libgfortran.so.5 to cap it against"
        ver_gt "${ver#GFORTRAN_}" "$GFORTRAN_CEIL" && fail "$so needs $ver (> $GFORTRAN_CEIL)" ;;
    esac
  done < <(objdump -T "$so" 2>/dev/null | grep -oE '(GLIBC|GLIBCXX|CXXABI|GFORTRAN)_[0-9.]+' | sort -uV)
done < <(find "${LIBDIRS[@]}" -name '*.so*' -type f)
note "symbol versions within GLIBC<=$GLIBC_FLOOR GLIBCXX<=$GLIBCXX_CEIL CXXABI<=$CXXABI_CEIL GFORTRAN<=${GFORTRAN_CEIL:-none}"

# ------------------------------------------------------------ 7b. MPI ABI --
# The bundle may link Open MPI only through libraries whose soname is the same
# in 4.x and 5.x; anything else (libmpi_cxx, libopen-pal, libopen-rte, pmix,
# ...) is renamed or gone on one side and would fail to load there. Non-MPI
# variants must not link MPI at all, and an MPI variant must actually link it
# -- which proves MPI=ON reached the build.
OMPI_STABLE_SONAMES=" libmpi.so.40 libmpi_mpifh.so.40 libmpi_usempif08.so.40 libmpi_usempi_ignore_tkr.so.40 "
mpi_linked=0
while IFS= read -r so; do
  file -b "$so" | grep -q ELF || continue
  while IFS= read -r need; do
    case "$need" in
      libmpi* | libopen-pal* | libopen-rte* | libompi* | libpmix* | libprrte*)
        if [ "$MPI_IMPL" = none ]; then fail "$so links $need, but $VARIANT is a non-MPI variant"; fi
        case "$OMPI_STABLE_SONAMES" in
          *" $need "*) ;;
          *) fail "$so links $need -- not ABI-stable across Open MPI 4.x/5.x (allowed:$OMPI_STABLE_SONAMES)" ;;
        esac
        if [ "$need" = libmpi.so.40 ]; then mpi_linked=1; fi
        ;;
    esac
  done < <(readelf -d "$so" | sed -nE 's/.*\(NEEDED\).*\[(.*)\]/\1/p')
done < <(find "${LIBDIRS[@]}" -name '*.so*' -type f)
if [ "$MPI_IMPL" != none ]; then
  [ "$mpi_linked" -eq 1 ] || fail "no bundle library links libmpi.so.40 -- did MPI=ON reach the build?"
  note "Open MPI linked through ABI-stable sonames only"
fi

# -------------------------------------------------- 8. residual path audit --
# Configuration files may reference: the canonical build prefix (rewritten by
# the installer), standard /usr paths (headers, -l search dirs), and the
# toolset compiler records in deal.IIConfig.cmake (also rewritten by the
# installer). An absolute path into the build container's scratch dirs is a
# leak that would break consumers. Scanned in config files only -- source
# headers legitimately contain substrings like "Eigen/src/".
# (the kept deal.II summary/detailed.log are provenance and exempt)
mapfile -t _cfg_files < <(find "$BUNDLE/lib/cmake" "$BUNDLE/lib/pkgconfig" \
  "$BUNDLE/lib64/cmake" "$BUNDLE/lib64/pkgconfig" "$BUNDLE/cmake" "$BUNDLE/bin" \
  "$BUNDLE/share/DiFfRG" -type f 2>/dev/null; find "$BUNDLE" -maxdepth 1 -type f ! -name '*.log')
if grep -lE '(^|["=[:space:];:])(/root|/home|/build|/src)/' "${_cfg_files[@]}" 2>/dev/null; then
  fail "configuration files reference build-time paths (above)"
fi
note "no stray build-time paths in configuration files"

# --------------------------------------------------------------- 9. manifest --
# Dependency versions straight from the installed CMake package metadata.
dep_versions() {
  local first=1
  for cv in "$BUNDLE"/lib{,64}/cmake/*/*[Cc]onfig[Vv]ersion.cmake \
            "$BUNDLE"/lib{,64}/cmake/*/*config-version.cmake; do
    [ -f "$cv" ] || continue
    local name ver
    name="$(basename "$(dirname "$cv")")"
    case "$name" in boost_*) continue ;; esac # per-component Boost dirs, top-level Boost suffices
    ver="$(grep -m1 -oE 'PACKAGE_VERSION "?[0-9][0-9.]*"?' "$cv" | grep -oE '[0-9][0-9.]*' || true)"
    [ -n "$ver" ] || continue
    [ $first -eq 1 ] || printf ',\n'
    first=0
    printf '    "%s": "%s"' "$name" "$ver"
  done
  printf '\n'
}

# CUDA variants additionally rely on the host's CUDA runtime and driver, MPI
# variants on the host's Open MPI (and whatever it links in turn).
CUDA_ALLOW=''
if [ "$ACCEL" != cpu ]; then
  CUDA_ALLOW='"libcudart", "libcuda", "libnvrtc", '
fi
MPI_ALLOW=''
if [ "$MPI_IMPL" != none ]; then
  MPI_ALLOW='"libmpi", "libmpi_mpifh", "libmpi_usempif08", "libmpi_usempi_ignore_tkr", '
fi

BOOST_VER="$(grep -m1 -oE 'DiFfRG_PINNED_BOOST_VERSION "[0-9.]+"' "$BUNDLE/DiFfRG_bundled_config.cmake" | grep -oE '[0-9.]+' || echo unknown)"
# In the release container .git is dockerignored; the SHA arrives via env from
# the build driver instead.
GIT_SHA="${GIT_SHA:-$(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo unknown)}"
DIFFRG_VERSION="$(cat "$SRC/VERSION" 2>/dev/null | tr -d '[:space:]' || echo unknown)"
COMPILER_ID="$(c++ --version | head -1)"

cat > "$BUNDLE/BUNDLE_MANIFEST.json" <<EOF
{
  "name": "diffrg-deps-${BUNDLE_VERSION}-${VARIANT}",
  "bundle_version": "${BUNDLE_VERSION}",
  "os": "linux",
  "arch": "x86_64",
  "isa": "${MARCH}",
  "accel": "${ACCEL}",
  "mpi": "${MPI_IMPL}",
  "diffrg_version": "${DIFFRG_VERSION}",
  "diffrg_git_sha": "${GIT_SHA}",
  "deps_inputs_hash": "${DEPS_INPUTS_HASH:-unknown}",
  "build_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "build_prefix": "/opt/diffrg",
  "glibc_floor": "${GLIBC_FLOOR}",
  "compiler": "${COMPILER_ID}",
  "builder_cxx": "$(command -v c++ || true)",
  "builder_cc": "$(command -v cc || true)",
  "builder_fc": "$(command -v gfortran || true)",
  "builder_cuda_root": "${BUILDER_CUDA_ROOT}",
  "builder_mpi_libdir": "${BUILDER_MPI_LIBDIR}",
  "builder_mpi_incdir": "${BUILDER_MPI_INCDIR}",
  "builder_mpi_bindir": "${BUILDER_MPI_BINDIR}",
  "boost_version": "${BOOST_VER}",
  "dependency_versions": {
$(dep_versions)
  },
  "allowed_external_libs": [
    ${CUDA_ALLOW}${MPI_ALLOW}"linux-vdso", "ld-linux-x86-64", "libc", "libm", "libmvec", "libpthread", "libdl",
    "librt", "libgcc_s", "libstdc++", "libgfortran", "libquadmath", "libgomp",
    "libz", "libopenblas"
  ]
}
EOF
note "manifest written to $BUNDLE/BUNDLE_MANIFEST.json"
note "OK"
