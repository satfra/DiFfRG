# Binary dependency releases

This directory builds, validates, and publishes the pre-built dependency
bundles that `install-diffrg-deps.sh` (repo root) installs for users: a
relocatable tarball of the full superbuild output (`bundled/`) attached to a
GitHub Release under a `deps-v<X.Y.Z>` tag.

CI builds and tests the library against these same bundles (see
`containers/ci/README.md`). A bundle force-bundles Boost/TBB/HDF5/SUNDIALS
(`-DBUILD_*=ON`), is compiled for the portable `x86-64-v3` ISA baseline
(`-DMARCH=x86-64-v3`), is built on Rocky 9 for a glibc 2.34 floor, and is
post-processed to be relocatable to any install prefix.

## Two equivalent build paths

Everything below can run **locally** (this walkthrough) or **on demand in CI**:
the `release-deps-linux` and `release-deps-macos` workflows
(`.github/workflows/`, `workflow_dispatch`) execute these same scripts on
GitHub runners, upload the tarball + sha256 + `.tested` stamp as a workflow
artifact, and can optionally attach everything to a **draft** release
`deps-v<version>` (input `draft_release`). Publishing is always manual: either
publish the draft in the Releases UI, or download the artifact into
`containers/release/dist/` and run `publish-release.sh`. The installers only
ever see *published* releases -- the GitHub API hides drafts -- so nothing is
user-visible until that manual step.

## Release walkthrough

```bash
# 1. Build the tarball (in Docker; ~2-3h). Writes dist/diffrg-deps-<v>-linux-x86_64-v3-cpu.tar.zst
containers/release/build-release.sh -v 1.0.0

# 2. Validate on the distro matrix (Ubuntu 24.04, Debian 13, Fedora 41, Rocky 9):
#    install from the tarball, audit linkage, build the library, run the quick
#    test suite, and re-install at a second prefix (relocation test).
containers/release/test-tarball.sh -f containers/release/dist/diffrg-deps-1.0.0-linux-x86_64-v3-cpu.tar.zst

# 3. Tag deps-v1.0.0 and create the GitHub release with tarball + sha256 + manifest.
containers/release/publish-release.sh -v 1.0.0
```

Other variants take `-V <variant>` in `build-release.sh` and
`publish-release.sh` (which then adds them to the existing release), and their
flags in `test-tarball.sh` (`-g` for CUDA, `-m` for Open MPI). In CI the
`mpi` input of both Linux workflows selects the `-openmpi` variant.

## Versioning

Bundle versions are their own `deps-vX.Y.Z` counter, independent of the DiFfRG
version (one bundle serves many DiFfRG commits; the manifest inside records the
exact git SHA it was built from). Bump the patch level for a rebuild, minor for
a dependency version bump, major for a deal.II/Boost/Kokkos major change.

The manifest also records `deps_inputs_hash` (`deps-inputs-hash.sh`), the hash
of everything that variant is built from. CI uses the release pinned in
`.github/deps-bundle-version` while that hash matches the checkout, and builds
its own bundle otherwise -- so after publishing a bundle for new dependency
inputs, bump the pin.

## What makes the tarball relocatable

Built at the canonical prefix `/opt/diffrg`, then post-processed
(`postprocess-bundle.sh`):

- every shared library gets `RUNPATH=$ORIGIN:$ORIGIN/../lib:$ORIGIN/../lib64`,
  so no binary patching is needed at install time (the superbuild pins
  `CMAKE_INSTALL_LIBDIR=lib` for a flat `lib/`; the lib64 leg and the scripts'
  lib64 handling are kept defensively for unforeseen layouts);
- deal.II's exported absolute system-library paths (`/usr/lib64/libz.so`, ...)
  are rewritten to plain `-l` flags so any distro's linker resolves them;
- the pin file `DiFfRG_bundled_config.cmake` is generated
  `${CMAKE_CURRENT_LIST_DIR}`-relative by the superbuild itself when all
  dependencies are bundled;
- the installer rewrites the remaining `/opt/diffrg` text occurrences (CMake
  configs, pkg-config files) to the user's prefix, and points deal.II's recorded
  compilers at the host toolchain.

Hard audits run before a tarball is produced: an ISA guard (no AVX-512
anywhere, AVX present in the heavyweight libraries -- catches both a
`-march=native` leak and dead `-march` plumbing), symbol-version caps
(`GLIBC_ <= 2.34`, el9 `GLIBCXX`/`CXXABI` baselines), a build-path leak scan,
and a linkage audit in a bare runtime container without any of the bundled
libraries' dev packages.

## Variants

The full set is `cpu` and `cuda12`, each also as an Open MPI build
(`cpu-openmpi`, `cuda12-openmpi`), plus the experimental macOS build.

- **cpu** (`linux-x86_64-v3-cpu`): the baseline, described above.
- **cuda12** (`linux-x86_64-v3-cuda12`, `build-release.sh -V ...`): CUDA-enabled
  Kokkos at the sm_75/Turing floor -- Kokkos allows exactly one CUDA arch per
  build, and sm_75 embeds compute_75 PTX so every newer GPU runs the bundle's
  own kernels via JIT (cached after first launch); libDiFfRG and applications
  compile for the local GPU through `DiFfRG_CUDA_ARCH`. Consumers need the CUDA 12 toolkit anyway (their
  apps compile device code), so the bundle resolves the host's libcudart. The
  host compiler must be **GCC 12 or >= 14** (never 13: nvcc's frontend
  miscompiles GCC 13's libstdc++ in C++20 mode -- the `iterator_traits<char*>`
  bug; both 12 and 14 are validated). On gcc-13 distros install g++-14 and run
  the installer with `CXX=g++-14`. The deal.II nvcc-wrapper shim ships in
  `bundled/bin` and self-relocates. The CPU bundles have no such constraint:
  any C++20 compiler, GCC >= 12, works.
  GPU-executed validation needs a GPU host: `test-tarball.sh -g` (uses the
  `*-cuda` test images; CI does build-only).
- **openmpi** (`<cpu|cuda12>-openmpi`, `build-release.sh -V linux-x86_64-v3-cpu-openmpi`):
  the superbuild's `MPI=ON` configuration -- MPI-enabled deal.II and SUNDIALS
  plus PETSc with hypre and MUMPS -- linked against EL9's Open MPI 4.1. Open
  MPI itself is not bundled: consumers compile with the host's `mpicc` anyway,
  and need Open MPI's development package (`libopenmpi-dev openmpi-bin`,
  `openmpi-devel` + `module load mpi/openmpi-x86_64`, `openmpi`). Every Open
  MPI library the bundle links (`libmpi`, `libmpi_mpifh`, `libmpi_usempif08`,
  `libmpi_usempi_ignore_tkr`) keeps soname `.40` from 4.x through 5.x, so 4.1
  is the floor and newer hosts work; a hard audit rejects any other Open MPI
  library (`libmpi_cxx`, `libopen-pal`, ...), which is renamed or gone on one
  side. Open MPI keeps the C ABI from 4.x to 5.0, but PETSc's `petscsys.h`
  refuses any newer Open MPI major at compile time; postprocessing removes
  exactly that check (a host *older* than the build's 4.1.1 is still refused).
  EL9's 4.1 wrappers are also stripped of the MPI-2 C++ bindings
  (`-lmpi_cxx`, which Open MPI 5 dropped) before the superbuild sees them.
  Two deal.II patches matter here (`patches/`): `mumps-parmetis` makes
  deal.II's MUMPS interface list ParMETIS and the Fortran runtime after
  PETSc's *static* MUMPS archives (Debian/Ubuntu link `--as-needed` and fail
  otherwise), and `vector-pool-teardown` stops a PETSc run from segfaulting
  at exit after `DiFfRG::Init()`. The installer rewrites the recorded `/usr/lib64/openmpi` paths to the
  host's Open MPI, and a library build picks MPI up from the bundle's pin file
  without `-DMPI=ON`. Other MPIs (Intel MPI, Cray MPICH, MVAPICH -- i.e. most
  clusters with a fabric- and Slurm-integrated site MPI) are ABI-incompatible
  and stay a source build (the wizard's self-build path with MPI on).
  Validate with `test-tarball.sh -m` (adds `-g` for the CUDA one): the distro
  matrix covers Open MPI 4.1 (Ubuntu 24.04, Rocky 9) and 5.x (Debian 13,
  Fedora 41), and ctest includes the multi-rank `mpi`-labelled tests.
- **macOS arm64** (`macos-arm64-cpu.sh`): experimental, see above; no `-march`
  pinning needed (all M-series share Apple clang's arm64 baseline).
