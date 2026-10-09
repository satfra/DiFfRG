#!/bin/bash
# ##############################################################################
# Install a pre-built DiFfRG dependency bundle.
#
# Downloads a relocatable binary bundle of DiFfRG's dependencies (deal.II,
# Kokkos, Boost, TBB, SUNDIALS, HDF5, ...) from GitHub Releases and installs it
# so the multi-hour dependency superbuild is skipped entirely. The DiFfRG
# library itself is then built from source against the bundle (minutes, not
# hours) -- either by hand or with --build-library.
#
# Usage:
#   bash <(curl -sL https://raw.githubusercontent.com/satfra/DiFfRG_current/main/install-diffrg-deps.sh) [options]
#
# Options:
#   --prefix DIR      install prefix (default: $HOME/.local/share/DiFfRG; env FOLDER)
#   --version X.Y.Z   bundle version (default: latest deps-v* release)
#   --variant NAME    bundle variant (default: auto-detected, linux-x86_64-v3-cpu)
#   --mpi             the Open MPI variant of the (default or given) bundle,
#                     e.g. linux-x86_64-v3-cpu-openmpi; needs Open MPI on the host
#   --file TARBALL    install from a local tarball instead of downloading
#   --force           replace an existing <prefix>/bundled tree
#   --skip-cpu-check  do not gate on x86-64-v3 CPU support (the binaries will
#                     still SIGILL on unsupported CPUs when run)
#   --build-library   also configure+build+install the DiFfRG library (requires
#                     running inside a DiFfRG checkout)
#   -h, --help        this text
#
# For CPUs without AVX2 (pre-2013), an MPI other than Open MPI (Intel MPI,
# Cray MPICH, MVAPICH, ...), or any other configuration the binary bundles do
# not cover, use the self-build path instead: install_diffrg.sh --mode source
# (in the same repository).
# ##############################################################################
set -euo pipefail

usage() {
  cat <<'EOF'
Install a pre-built DiFfRG dependency bundle from GitHub Releases, so the
multi-hour dependency superbuild is skipped; the DiFfRG library itself is then
built from source against the bundle (minutes).

Usage:
  bash <(curl -sL https://raw.githubusercontent.com/satfra/DiFfRG_current/main/install-diffrg-deps.sh) [options]

Options:
  --prefix DIR      install prefix (default: $HOME/.local/share/DiFfRG; env FOLDER)
  --version X.Y.Z   bundle version (default: latest deps-v* release)
  --variant NAME    bundle variant (default: auto-detected, linux-x86_64-v3-cpu)
  --mpi             the Open MPI variant of the bundle (e.g. linux-x86_64-v3-cpu-openmpi);
                    needs Open MPI's development package on this machine
  --file TARBALL    install from a local tarball instead of downloading
  --force           replace an existing <prefix>/bundled tree
  --skip-cpu-check  do not gate on x86-64-v3 CPU support
  --build-library   also configure+build+install the DiFfRG library (requires
                    running inside a DiFfRG checkout)
  -h, --help        this text

For CPUs without AVX2 (pre-2013), an MPI other than Open MPI, or any other
configuration the binary bundles do not cover, use the self-build path instead:
install_diffrg.sh --mode source (in the same repository).
EOF
}

REPO="satfra/DiFfRG_current"
BUILD_PREFIX="/opt/diffrg" # canonical prefix baked into release bundles

os_name="$(uname -s)"
case "${os_name}-$(uname -m)" in
Linux-x86_64) DEFAULT_VARIANT="linux-x86_64-v3-cpu" ;;
Darwin-arm64) DEFAULT_VARIANT="macos-arm64-cpu" ;;
*) DEFAULT_VARIANT="" ;;
esac

# Portability helpers: macOS ships BSD sed (needs -i '') and shasum instead of
# sha256sum.
sed_inplace() {
  if sed --version >/dev/null 2>&1; then sed -i "$@"; else sed -i '' "$@"; fi
}
sha256_verify() { # <checksum-file> (run from its directory)
  if command -v sha256sum >/dev/null; then sha256sum -c "$1" >/dev/null
  else shasum -a 256 -c "$1" >/dev/null; fi
}

prefix="${FOLDER:-$HOME/.local/share/DiFfRG}"
version=''
variant=''
want_mpi=0
tarball=''
force=0
skip_cpu_check=0
build_library=0

while [[ $# -gt 0 ]]; do
  case "$1" in
  --prefix) prefix="$2"; shift 2 ;;
  --version) version="$2"; shift 2 ;;
  --variant) variant="$2"; shift 2 ;;
  --mpi) want_mpi=1; shift ;;
  --file) tarball="$2"; shift 2 ;;
  --force) force=1; shift ;;
  --skip-cpu-check) skip_cpu_check=1; shift ;;
  --build-library) build_library=1; shift ;;
  -h | --help) usage; exit 0 ;;
  *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

err() { echo -e "\033[1;31mERROR:\033[0m $*" >&2; exit 1; }
warn() { echo -e "\033[1;33mWARNING:\033[0m $*" >&2; }
info() { echo -e "\033[1;32m==>\033[0m $*"; }

# '#' delimits the relocation sed, '&' and '\' are special in its replacement.
case "$prefix" in
*'#'* | *'&'* | *'\'* | *$'\n'*) err "Install prefix must not contain '#', '&', '\\' or newlines: $prefix" ;;
/*) ;;
*) prefix="$(pwd)/$prefix" ;;
esac

# ------------------------------------------------------------ platform gate --
[[ -n $DEFAULT_VARIANT ]] \
  || err "Pre-built bundles exist for Linux x86_64 and macOS arm64 only ($(uname -s) $(uname -m) detected). Use install_diffrg.sh --mode source to build from source."

[[ -z $variant ]] && variant="$DEFAULT_VARIANT"
if [[ $want_mpi -eq 1 && $variant != *-openmpi ]]; then
  [[ ${os_name} == Linux ]] || err "MPI bundles exist for Linux only. Use install_diffrg.sh --mode source."
  variant="${variant}-openmpi"
fi
if [[ $skip_cpu_check -eq 0 && ${os_name} == Linux ]]; then
  # The v3 bundles need AVX2/FMA (any consumer CPU from ~2013 on). Prefer the
  # authoritative glibc probe; fall back to /proc/cpuinfo flags.
  if ld.so --help 2>/dev/null | grep -q 'x86-64-v3'; then
    ld.so --help 2>/dev/null | grep 'x86-64-v3' | grep -q supported \
      || err "This CPU does not support x86-64-v3 (AVX2+FMA), which the pre-built bundles require.
Use the self-build path instead: install_diffrg.sh --mode source"
  elif ! grep -qm1 avx2 /proc/cpuinfo || ! grep -qm1 fma /proc/cpuinfo; then
    err "This CPU does not support x86-64-v3 (AVX2+FMA), which the pre-built bundles require.
Use the self-build path instead: install_diffrg.sh --mode source"
  fi
fi

# curl/sha256sum are only needed for the download path; a local --file install
# (e.g. air-gapped) works without them.
required_tools=(tar sed grep cmake)
[[ -z $tarball ]] && required_tools+=(curl sha256sum)
for tool in "${required_tools[@]}"; do
  command -v "$tool" >/dev/null || err "'$tool' is required but not installed."
done
tar --zstd --version >/dev/null 2>&1 || command -v zstd >/dev/null \
  || err "zstd is required to unpack the bundle (install the 'zstd' package)."

# ----------------------------------------------------------------- download --
workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

if [[ -z $tarball ]]; then
  if [[ -z $version ]]; then
    info "Looking up the latest dependency release..."
    # The || true keeps set -e/pipefail from aborting before the friendly
    # error below when the API is unreachable or no release matches.
    version="$(curl -fsSL "https://api.github.com/repos/${REPO}/releases?per_page=100" 2>/dev/null |
      grep -oE '"tag_name": *"deps-v[0-9]+\.[0-9]+\.[0-9]+"' |
      grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | sort -V | tail -1 || true)"
    [[ -n $version ]] || err "No deps-v* release found on github.com/${REPO}."
  fi
  name="diffrg-deps-${version}-${variant}"
  url="https://github.com/${REPO}/releases/download/deps-v${version}/${name}.tar.zst"
  info "Downloading ${name}.tar.zst..."
  curl -fL --progress-bar -o "$workdir/${name}.tar.zst" "$url" \
    || err "Download failed: $url
Check that release deps-v${version} provides the '${variant}' variant."
  curl -fsSL -o "$workdir/${name}.tar.zst.sha256" "${url}.sha256" \
    || err "Checksum download failed: ${url}.sha256"
  (cd "$workdir" && sha256_verify "${name}.tar.zst.sha256") \
    || err "Checksum verification FAILED for ${name}.tar.zst -- corrupted download?"
  info "Checksum verified."
  tarball="$workdir/${name}.tar.zst"
else
  [[ -f $tarball ]] || err "No such file: $tarball"
  info "Installing from local tarball $tarball"
fi

# ------------------------------------------------------------------- unpack --
info "Unpacking..."
tar --zstd -xf "$tarball" -C "$workdir" \
  || zstd -dc "$tarball" | tar -xf - -C "$workdir"
srcdir="$(find "$workdir" -maxdepth 1 -type d -name 'diffrg-deps-*' | head -1)"
[[ -n $srcdir && -d $srcdir/bundled ]] || err "Tarball does not look like a DiFfRG dependency bundle."

manifest="$srcdir/bundled/BUNDLE_MANIFEST.json"
[[ -f $manifest ]] || err "Bundle has no BUNDLE_MANIFEST.json."
# The bundle names its own variant; it is what the checks below must follow
# (a --file install would otherwise be checked as the platform default).
bundle_name="$(grep -oE '"name": *"[^"]*"' "$manifest" | sed -E 's/.*: *"([^"]*)"/\1/' || true)"
bundle_variant="$(sed -E 's/^diffrg-deps-[0-9]+\.[0-9]+\.[0-9]+-//' <<<"$bundle_name")"
if [[ -n $bundle_variant && $bundle_variant != "$variant" ]]; then
  [[ -n $tarball && $want_mpi -eq 0 && $variant == "$DEFAULT_VARIANT" ]] \
    || warn "Requested variant '$variant', but the tarball is '$bundle_variant'; using the tarball's."
  variant="$bundle_variant"
fi
if [[ ${os_name} == Linux ]]; then
  glibc_floor="$(grep -oE '"glibc_floor": *"[0-9.]+"' "$manifest" | grep -oE '[0-9.]+' || echo '')"
  if [[ -n $glibc_floor ]]; then
    glibc_have="$(ldd --version | head -1 | grep -oE '[0-9]+\.[0-9]+$' || echo '')"
    if [[ -n $glibc_have && "$(printf '%s\n%s\n' "$glibc_floor" "$glibc_have" | sort -V | head -1)" != "$glibc_floor" ]]; then
      err "This system's glibc ($glibc_have) is older than the bundle requires ($glibc_floor).
Use the self-build path instead: install_diffrg.sh --mode source"
    fi
  fi
else
  min_macos="$(grep -oE '"min_macos": *"[0-9.]+"' "$manifest" | grep -oE '[0-9.]+' || echo '')"
  if [[ -n $min_macos ]]; then
    macos_have="$(sw_vers -productVersion 2>/dev/null || echo '')"
    if [[ -n $macos_have && "$(printf '%s\n%s\n' "$min_macos" "$macos_have" | sort -V | head -1)" != "$min_macos" ]]; then
      err "This macOS ($macos_have) is older than the bundle requires ($min_macos).
Use the self-build path instead: install_diffrg.sh --mode source"
    fi
  fi
fi

# ------------------------------------------------------------------ install --
if [[ -e "$prefix/bundled" ]]; then
  if [[ $force -eq 1 ]]; then
    info "Removing existing $prefix/bundled (--force)..."
    rm -rf "$prefix/bundled"
  else
    err "$prefix/bundled already exists. Re-run with --force to replace it,
or choose another prefix with --prefix."
  fi
fi
mkdir -p "$prefix"
mv "$srcdir/bundled" "$prefix/bundled"
info "Installed bundle to $prefix/bundled"

# -------------------------------------------------------------------- fixup --
# The bundle was built with prefix /opt/diffrg; rewrite every text-file
# occurrence to the actual install prefix. (Shared-library RUNPATHs are already
# $ORIGIN-relative -- no binary patching needed.)
info "Relocating bundle to $prefix..."
# Process substitution, not a pipe: grep exits 1 when nothing needs rewriting,
# which under pipefail would abort the install halfway through. (No mapfile:
# macOS ships bash 3.2.)
while IFS= read -r f; do
  [[ -n $f ]] || continue
  sed_inplace "s#${BUILD_PREFIX}/bundled#${prefix}/bundled#g; s#${BUILD_PREFIX}#${prefix}#g" "$f"
done < <(grep -rIl --exclude='*.log' "$BUILD_PREFIX" "$prefix/bundled" 2>/dev/null || true)

# On macOS the dylibs carry the build prefix in their install names and
# cross-references; rewrite those too and ad-hoc re-sign (arm64 macOS refuses
# to load a modified, unsigned binary).
if [[ ${os_name} == Darwin ]]; then
  find "$prefix/bundled/lib" -name '*.dylib' -type f | while IFS= read -r dylib; do
    install_name_tool -id "${dylib}" "${dylib}" 2>/dev/null || true
    otool -L "$dylib" | awk 'NR>1{print $1}' | grep "^${BUILD_PREFIX}" |
      while IFS= read -r dep; do
        install_name_tool -change "$dep" "${dep/#${BUILD_PREFIX}/${prefix}}" "$dylib"
      done || true
    codesign --force -s - "$dylib" 2>/dev/null \
      || warn "could not re-sign $dylib -- it may fail to load"
  done
fi

# Several tools record the build container's compilers by absolute path
# (Kokkos_CXX_COMPILER, kokkos_launch_compiler's default, deal.II's records,
# the nvcc-wrapper shim's pinned GCC, h5cc). Those paths do not exist on this
# machine and would surface later as baffling "compiler not found" failures --
# nvcc's host-compiler error being the classic. Rewrite every text occurrence
# to this machine's toolchain, as recorded by the manifest.
# CC/CXX/FC env vars select the host compilers written into the records --
# needed e.g. on distros whose default GCC is unsuitable for the CUDA variant.
sed_escape() { printf '%s' "$1" | sed 's/[][\.*^$#]/\\&/g'; }
manifest_field() {
  grep -oE "\"$1\": *\"[^\"]*\"" "$prefix/bundled/BUNDLE_MANIFEST.json" | sed -E 's/.*: *"([^"]*)"/\1/' || true
}
rewrite_bundle_path() { # <builder-path> <host-path>
  [[ -n $1 && -n $2 && $1 != "$2" ]] || return 0
  while IFS= read -r f; do
    [[ -n $f ]] || continue
    sed_inplace "s#$(sed_escape "$1")#$2#g" "$f"
  done < <(grep -rIl --exclude='*.log' --exclude=BUNDLE_MANIFEST.json -F "$1" "$prefix/bundled" 2>/dev/null || true)
}

for pair in "builder_cxx:${CXX:-c++}" "builder_cc:${CC:-cc}" "builder_fc:${FC:-gfortran}"; do
  rewrite_bundle_path "$(manifest_field "${pair%%:*}")" "$(command -v "${pair##*:}" || true)"
done

# CUDA variants record toolkit paths (deal.II's include dir, libcudart, the
# link-time driver stub) under the canonical builder root; point them at this
# machine's toolkit, located from nvcc, CUDA_HOME, or the usual roots.
builder_cuda_root="$(manifest_field builder_cuda_root)"
if [[ -n $builder_cuda_root ]]; then
  host_cuda_root=''
  if command -v nvcc >/dev/null 2>&1; then
    nvcc_path="$(command -v nvcc)"
    nvcc_path="$(readlink -f "$nvcc_path" 2>/dev/null || echo "$nvcc_path")"
    host_cuda_root="${nvcc_path%/bin/nvcc}"
  elif [[ -n ${CUDA_HOME:-} && -d ${CUDA_HOME:-} ]]; then
    host_cuda_root="$CUDA_HOME"
  else
    for c in /usr/local/cuda /opt/cuda; do [[ -d $c ]] && { host_cuda_root="$c"; break; }; done
  fi
  if [[ -n $host_cuda_root ]]; then
    rewrite_bundle_path "$builder_cuda_root" "$host_cuda_root"
    info "CUDA toolkit paths point at $host_cuda_root"
  else
    warn "No CUDA toolkit found (nvcc/CUDA_HOME); the bundle's recorded toolkit
paths were left at $builder_cuda_root and configuring DiFfRG will fail until
a CUDA 12 toolkit is installed."
  fi
fi

# MPI variants record the builder's Open MPI (EL9 layout, /usr/lib64/openmpi)
# by absolute path in deal.II's and PETSc's configs; point them at this
# machine's Open MPI, located through its mpicc wrapper. Distros differ: e.g.
# Debian's lives in /usr/lib/x86_64-linux-gnu/openmpi/{lib,include}, Arch's in
# /usr/lib and /usr/include, where the wrapper reports no libdir at all.
mpi_missing=''
builder_mpi_libdir="$(manifest_field builder_mpi_libdir)"
host_mpi_libdir=''
if [[ -n $builder_mpi_libdir ]]; then
  host_mpicc="${MPICC:-mpicc}"
  # EL/Fedora keep Open MPI off PATH until `module load mpi/openmpi-x86_64`.
  if ! command -v "$host_mpicc" >/dev/null 2>&1 && [[ -z ${MPICC:-} && -x /usr/lib64/openmpi/bin/mpicc ]]; then
    host_mpicc=/usr/lib64/openmpi/bin/mpicc
  fi
  host_mpi_version="$("$host_mpicc" --showme:version 2>&1 || true)"
  host_mpi_version="${host_mpi_version%%$'\n'*}"
  if [[ $host_mpi_version == *"Open MPI"* ]]; then
    for d in $("$host_mpicc" --showme:libdirs) /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu; do
      if [[ -e $d/libmpi.so ]]; then host_mpi_libdir="$d"; break; fi
    done
    host_mpi_incdir=''
    for d in $("$host_mpicc" --showme:incdirs) /usr/include; do
      if [[ -e $d/mpi.h ]]; then host_mpi_incdir="$d"; break; fi
    done
    if [[ -n $host_mpi_libdir && -n $host_mpi_incdir ]]; then
      rewrite_bundle_path "$builder_mpi_libdir" "$host_mpi_libdir"
      rewrite_bundle_path "$(manifest_field builder_mpi_incdir)" "$host_mpi_incdir"
      rewrite_bundle_path "$(manifest_field builder_mpi_bindir)" "$(dirname "$(command -v "$host_mpicc")")"
      info "Open MPI paths point at $host_mpi_libdir ($host_mpi_version)"
    else
      host_mpi_libdir=''
      mpi_missing="'$host_mpicc' is Open MPI, but its libmpi.so or mpi.h was not found -- is the development package installed?"
    fi
  elif command -v "$host_mpicc" >/dev/null 2>&1; then
    mpi_missing="'$host_mpicc' is not Open MPI ($host_mpi_version); this bundle needs Open MPI >= 4.1. Other MPIs need a self-build (install_diffrg.sh --mode source)."
  else
    mpi_missing="Open MPI >= 4.1 with its development files and 'mpicc' on PATH (package 'libopenmpi-dev openmpi-bin' / 'openmpi-devel' + 'module load mpi/openmpi-x86_64' on EL/Fedora / 'openmpi' on Arch)"
  fi
  if [[ -n $mpi_missing ]]; then
    warn "The bundle's Open MPI paths were left at the build machine's $builder_mpi_libdir.
Install Open MPI and re-run this installer with --force before building DiFfRG."
  fi
fi

cat > "$prefix/bundled/INSTALL_RECEIPT.json" <<EOF
{
  "installed_to": "${prefix}",
  "installed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "source": "${tarball##*/}"
}
EOF

# -------------------------------------------------------------- host checks --
info "Checking host prerequisites for building DiFfRG against the bundle..."
missing=()

cxx_probe="$workdir/probe.cpp"
echo 'int main() { return 0; }' > "$cxx_probe"
if ! command -v c++ >/dev/null || ! c++ -std=c++20 -o "$workdir/probe" "$cxx_probe" 2>/dev/null; then
  missing+=("a C++20 compiler (GCC >= 12: package 'gcc-c++'/'g++' or 'gcc-toolset-14' on EL9; Xcode CLT on macOS)")
fi
if [[ ${os_name} == Linux ]]; then
  ldconfig -p 2>/dev/null | grep -qE 'libopenblas|liblapack' \
    || missing+=("BLAS/LAPACK (package 'libopenblas-dev' / 'openblas-devel')")
  { command -v pkg-config >/dev/null && pkg-config --exists gsl; } || [[ -e /usr/include/gsl/gsl_math.h ]] \
    || missing+=("GSL headers (package 'libgsl-dev' / 'gsl-devel')")
  [[ -e /usr/include/zlib.h ]] \
    || missing+=("zlib headers (package 'zlib1g-dev' / 'zlib-devel')")
else
  # macOS: BLAS/LAPACK come from the always-present Accelerate framework and
  # zlib from the SDK; only GSL and gfortran come from Homebrew.
  { command -v pkg-config >/dev/null && pkg-config --exists gsl; } || [[ -e /opt/homebrew/include/gsl/gsl_math.h ]] \
    || missing+=("GSL (brew install gsl)")
fi
command -v gfortran >/dev/null \
  || missing+=("gfortran (package 'gfortran' / 'gcc-gfortran'; 'brew install gcc' on macOS)")

# CUDA bundles: nvcc's frontend miscompiles GCC 13's libstdc++ in C++20 mode
# (the iterator_traits<char*> bug); GCC 12 and >= 14 are fine.
if [[ $variant == *cuda* ]]; then
  cxx_bin="${CXX:-c++}"
  cxx_major="$("$cxx_bin" -dumpversion 2>/dev/null | cut -d. -f1 || echo 0)"
  if [[ ! $cxx_major =~ ^[0-9]+$ ]] || ((cxx_major == 13)) || ((cxx_major < 12)); then
    warn "CUDA bundles need GCC 12 or >= 14 as nvcc's host compiler (13's libstdc++
triggers an nvcc bug); '$cxx_bin' reports version '${cxx_major}'.
Install a suitable g++ and re-run this installer with e.g. CXX=g++-14 (the
compiler is recorded into the bundle), then configure DiFfRG with
-DCMAKE_CXX_COMPILER=g++-14."
  fi
fi

# MPI bundles link the Fortran runtime (PETSc's MUMPS) through the C++
# compiler, which only searches its own GCC version's directories: g++-14 with
# only the distro's default gfortran-13 installed finds no libgfortran.so.
if [[ -n $builder_mpi_libdir ]]; then
  cxx_bin="${CXX:-c++}"
  gfortran_lib="$("$cxx_bin" -print-file-name=libgfortran.so 2>/dev/null || true)"
  if [[ $gfortran_lib != /* ]]; then
    cxx_major="$("$cxx_bin" -dumpversion 2>/dev/null | cut -d. -f1 || true)"
    missing+=("the Fortran runtime for '$cxx_bin' (libgfortran.so; install the gfortran matching it, e.g. 'gfortran-${cxx_major:-N}')")
  fi
fi

# deal.II links Open MPI's libraries by path, including the Fortran ones some
# distros split into a separate package; each must exist on this machine.
if [[ -n $mpi_missing ]]; then
  missing+=("$mpi_missing")
elif [[ -n $host_mpi_libdir ]]; then
  while IFS= read -r lib; do
    [[ -e $lib ]] || missing+=("$lib (recorded by deal.II; part of the Open MPI development/Fortran packages)")
  done < <(grep -ohE "${host_mpi_libdir}/libmpi[A-Za-z0-9_]*\.so" "$prefix"/bundled/lib/cmake/deal.II/deal.II*.cmake 2>/dev/null | sort -u || true)
fi

if [[ ${#missing[@]} -gt 0 ]]; then
  warn "The bundle is installed, but building DiFfRG will additionally need:"
  for m in "${missing[@]}"; do echo "  - $m" >&2; done
fi

# -------------------------------------------------------------------- verify --
info "Verifying the installed bundle..."
cmake -DBUNDLED_DIR="$prefix/bundled" -P "$prefix/bundled/share/DiFfRG/verify_install.cmake" \
  || err "Bundle verification failed -- see output above."

# ------------------------------------------------------------- next steps --
if [[ $build_library -eq 1 ]]; then
  [[ -f DiFfRG/CMakeLists.txt ]] \
    || err "--build-library requires running from a DiFfRG checkout (DiFfRG/CMakeLists.txt not found)."
  threads="${THREADS:-6}"
  info "Building the DiFfRG library against the bundle (${threads} threads)..."
  cmake -S DiFfRG -B "$prefix/library-build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUNDLED_DIR="$prefix/bundled" \
    -DCMAKE_INSTALL_PREFIX="$prefix" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DDiFfRG_DOCUMENTATION=OFF
  cmake --build "$prefix/library-build" -j "$threads"
  cmake --install "$prefix/library-build"
  info "DiFfRG installed to $prefix"
else
  echo
  info "Done. Build the DiFfRG library against the bundle with:"
  echo "    cmake -S <DiFfRG-checkout>/DiFfRG -B build \\"
  echo "        -DCMAKE_BUILD_TYPE=Release \\"
  echo "        -DBUNDLED_DIR=$prefix/bundled \\"
  echo "        -DCMAKE_INSTALL_PREFIX=$prefix"
  echo "    cmake --build build -j 6 && cmake --install build"
  if [[ ${os_name} == Linux ]]; then
    echo
    echo "  The default -march=native is safe on this machine (its CPU is a superset"
    echo "  of the bundle's x86-64-v3 baseline)."
  fi
fi
if [[ -n $builder_mpi_libdir ]]; then
  echo
  echo "  MPI is enabled by the bundle (no -DMPI=ON needed). Launch applications with"
  echo "  mpirun -n <ranks>; where Open MPI lives in a module (EL, Fedora), load it in"
  echo "  every shell that builds or runs them: module load mpi/openmpi-x86_64"
fi
