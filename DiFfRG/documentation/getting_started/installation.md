(Installation)=
# Installation
To compile and run this project, there are very few requirements which you can easily install using your package manager on Linux or MacOS:

- [git](https://git-scm.com/) for external requirements and to clone this repository.
- [CMake](https://www.cmake.org/) for the build systems of DiFfRG, deal.ii and other libraries.
- [GNU Make](https://www.gnu.org/software/make/) or another generator of your choice.
- A compiler supporting at least the C++20 standard. This project is only tested using the [GCC](https://gcc.gnu.org/) compiler suite, as well as with `AppleClang`, but in principle, ICC or standard Clang should also work.
- LAPACK and BLAS in some form, e.g. [OpenBlas](https://www.openblas.net/).
- The GNU Scientific Library [GSL](https://www.gnu.org/software/gsl/). If not found by DiFfRG, it will try to install it by itself.
- [Python](https://www.python.org/) is required by the Boost build system and used for visualization.
- [Doxygen](https://www.doxygen.org/) and [graphviz](https://www.graphviz.org/download/) to build the documentation.

The following requirements are optional:
- [ParaView](https://www.paraview.org/), a program to visualize and post-process the vtk data saved by DiFfRG when treating FEM discretizations.
- A GPU backend for the momentum integration routines, which gives a large speedup for fully momentum-dependent flow equations (10 - 100x). DiFfRG uses [Kokkos](https://kokkos.org/) to abstract the parallel backend, so GPU support is enabled through Kokkos' CUDA (NVIDIA) or HIP (AMD) backends. For the CUDA backend you need a working [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) together with a host compiler compatible with your `nvcc` version (e.g. `g++` <= 13.2 for CUDA 12.5). GPU support is controlled by the `-DGPU` CMake option (see below).

All other requirements are bundled and automatically built with DiFfRG.

## Supported systems

The framework has been tested with the following systems:

### Arch Linux
```bash
$ pacman -S git cmake gcc gcc-fortran blas-openblas paraview python doxygen graphviz gsl
```
In case you want to run with CUDA, as of January 2025 you have to have very specific versions of CUDA and gcc installed. Currently, the gcc13 compiler in the Arch package repository is incompatible with CUDA. To configure a system with a compatible CUDA+gcc configuration, them install directly from the Arch package archive
```bash
$ pacman -U https://archive.archlinux.org/packages/g/gcc12/gcc12-12.3.0-6-x86_64.pkg.tar.zst \
            https://archive.archlinux.org/packages/g/gcc12-libs/gcc12-libs-12.3.0-6-x86_64.pkg.tar.zst \
            https://archive.archlinux.org/packages/c/cuda/cuda-12.3.2-1-x86_64.pkg.tar.zst
```

### Rocky Linux
```bash
$ dnf --enablerepo=devel install -y gcc-toolset-12 cmake git openblas-devel doxygen doxygen-latex python3 python3-pip gsl-devel
$ scl enable gcc-toolset-12 bash
```

The second line is necessary to switch into a shell where `g++-12` is available

### Ubuntu
```bash
$ apt-get update
$ apt-get install git cmake gfortran libopenblas-dev paraview build-essential python3 doxygen graphviz libgsl-dev
```

### MacOS
First, install xcode and homebrew, then run
```bash
$ brew install cmake gcc doxygen paraview graphviz gsl python3 bash
```

### Windows

If using Windows, instead of running the project directly, it is recommended to use [WSL](https://learn.microsoft.com/en-us/windows/wsl/setup/environment) and then go through the installation as if on Linux (e.g. Arch or Ubuntu).

### Docker and other container runtime environments

Although a native install should be unproblematic in most cases, the setup with CUDA functionality may be daunting. Especially on high-performance clusters, and also depending on the packages available for  chosen distribution, it may be much easier to work with the framework inside a container.

The specific choice of runtime environment is up to the user, however we provide a small build script to create a Docker/OCI container in which DiFfRG will be built and then tested through Singularity/Apptainer when available.
To do this, you will need `docker`, `docker-buildx`, and `singularity` or `apptainer`. For CUDA-compatible image execution you also need the host NVIDIA driver available to Singularity/Apptainer.

To build a Docker image and test it through Singularity/Apptainer, you can run the script `build-container.sh` in the `containers/` folder:
```bash
$ cd containers
$ bash build-container.sh
```

If using other environments, e.g. [ENROOT](https://github.com/NVIDIA/enroot), the preferred approach is simply to build an image on top of the [CUDA images by NVIDIA](https://hub.docker.com/r/nvidia/cuda/tags). Optimal compatibility is given using `nvidia/cuda:12.5.1-devel-rockylinux`. Proceed with the installation setup for  Rocky Linux above.

## Setup

### Interactive installer (recommended)

From the shell, run
```bash
$ bash <(curl -s -L https://github.com/satfra/DiFfRG_current/raw/refs/heads/main/install_diffrg.sh)
```

A short wizard walks through the typical choices — pre-built dependency bundle or full self-build, install prefix, build folder, features (MPI, GPU, ...), and an optional copy of the examples and tutorials — then performs the complete installation. Every question also has a command-line flag (`--help`) for scripted use.

The **pre-built dependency bundle** downloads deal.II, Kokkos, Boost, TBB, SUNDIALS, HDF5 and friends as a ~50 MB binary from [GitHub Releases](https://github.com/satfra/DiFfRG_current/releases) instead of compiling them for hours; only the DiFfRG library itself is compiled locally (minutes). Bundles come in a CPU-only and a CUDA variant (the wizard offers the latter when it finds an NVIDIA GPU), and require a CPU with AVX2 (any consumer CPU from ~2013 on) and glibc ≥ 2.34 (Rocky/Alma 9+, Ubuntu 22.04+, Debian 12+, Fedora, Arch). Each also exists with MPI (`--mpi`, or the wizard's question): it links Open MPI ≥ 4.1 from the host — install your distro's Open MPI development package (`libopenmpi-dev openmpi-bin`, `openmpi-devel`, `openmpi`) first — and adds PETSc with hypre and MUMPS. The library build then enables MPI by itself. For other MPIs (Intel MPI, Cray MPICH, MVAPICH — typical on clusters), other CPUs, or other platforms the wizard's self-build path covers the full feature set. The bundle step alone is also available non-interactively as `install-diffrg-deps.sh` (`--help` for options).

### Quick install from source

The wizard's self-build path compiles the full dependency superbuild with your choice of features:
```bash
$ bash <(curl -s -L https://github.com/satfra/DiFfRG_current/raw/refs/heads/main/install_diffrg.sh) --mode source
```

Add e.g. `--mpi --gpu --threads 6 --prefix ${HOME}/.local/share/DiFfRG --yes` for a non-interactive run; `install_diffrg.sh --help` lists all options. Experts can drive the superbuild directly with CMake, as described next.

### Manual installation

Clone the repository:
```bash
$ git clone https://github.com/satfra/DiFfRG_current.git
```

Then, create a build directory and run cmake:
```bash
$ cd DiFfRG_current
$ mkdir build
$ cd build
$ cmake ../ -DCMAKE_INSTALL_PREFIX=~/.local/share/DiFfRG/ -DCMAKE_BUILD_TYPE=Release
$ cmake --build ./ -- -j8
```

By default, the library will install itself to `$HOME/.local/share/DiFfRG`, but you can control the destination by pointing `CMAKE_INSTALL_PREFIX` to a directory of your choice.

This top-level build is a *superbuild*: it builds the bundled dependencies (deal.II, Kokkos, and any of Boost/TBB/HDF5/SUNDIALS not found on the system) into `${CMAKE_INSTALL_PREFIX}/bundled` and then builds and installs the DiFfRG library itself.

### Build options

The most important options to pass to the top-level `cmake` invocation are:

- `-DGPU=ON/OFF` — GPU support via the Kokkos CUDA/HIP backend (default `ON`).
- `-DMPI=ON/OFF` — MPI support (default `OFF`; a library build configured from `DiFfRG/` against an existing bundle defaults to what the bundle was built with). This is a single switch for the whole superbuild: deal.II, SUNDIALS, PETSc and DiFfRG must all agree about MPI, so it cannot be enabled for DiFfRG alone afterwards.
- `-DNATIVE=ON/OFF` — optimize for the build machine's CPU (`-march=native`). Disable for portable binaries (default `ON`).
- `-DDiFfRG_CUDA_ARCH=<list>` — the GPU compute capabilities to compile for, e.g. `90` or `80;90` (also accepted in dotted form, `9.0`). Defaults to the GPUs found on the build machine via `nvidia-smi`; `bundled` keeps whatever the dependency bundle was built for. See [Choosing the GPU architecture](#choosing-the-gpu-architecture).
- `-DDiFfRG_TEST=ON` — build the test suite (default `OFF`).
- `-DDiFfRG_DOCUMENTATION=ON` — build this documentation (default `ON`).
- `-DBUILD_OpenBLAS=ON` — additionally build OpenBLAS (default `OFF`).

  The BLAS/LAPACK in use must have 32-bit integer indices (LP64): deal.II's `types::blas_int` is
  a plain `int` and PETSc is built `--with-64-bit-indices=0`. Configure probes the library it
  found and stops with an explanation if it turns out to be ILP64 — on a cluster that usually
  means an MKL or OpenBLAS module was loaded in its `ilp64` variant. Load the `lp64` one, or
  pass `-DBUILD_OpenBLAS=ON` and let DiFfRG build its own.

#### PETSc and distributed linear algebra

- `-DBUILD_PETSC=ON/OFF` — build PETSc (**default: follows `-DMPI`**). PETSc provides the
  distributed vectors, matrices and solvers, and via deal.II the only MPI-capable
  `SUNDIALS::IDA` instantiation that does not require Trilinos. Without MPI it buys nothing
  over the existing UMFPACK path, which is why it tracks `MPI`.
- `-DPETSC_GMRES=ON/OFF` — build PETSc with hypre, enabling `PreconditionBoomerAMG` for
  Krylov solves (default `ON`). Needs no Fortran and adds only a few minutes. With it off you
  still get a working distributed Krylov solve via `PreconditionBlockJacobi`, just a weaker
  preconditioner.
- `-DPETSC_MUMPS=ON/OFF` — build PETSc with MUMPS, enabling `SparseDirectMUMPS` distributed
  direct solves (default `OFF`). Requires a Fortran compiler and pulls in ScaLAPACK,
  METIS and ParMETIS; this is by far the longest part of the build.

deal.II detects what PETSc was actually built with and exports `DEAL_II_PETSC_WITH_HYPRE` /
`DEAL_II_PETSC_WITH_MUMPS`, which is what DiFfRG's solver wrappers gate on.

#### Choosing the GPU architecture

CUDA code is compiled for a specific *compute capability*. If the compiled architecture does not match the GPU, one of two things happens: an architecture **above** the device aborts at startup (Kokkos refuses to run), and one **below** it still runs, but only because the driver JIT-compiles the embedded PTX every time a module is loaded. DiFfRG's generated flow kernels are far too large for that second case to be acceptable — they overflow the driver's 256 MB JIT cache (`~/.nv/ComputeCache`), so the compilation is repeated on every run.

`-DDiFfRG_CUDA_ARCH` therefore decides what the library *and every application built against it* target, independently of the dependency bundle:

```bash
cmake .. -DDiFfRG_CUDA_ARCH=90        # H100
cmake .. -DDiFfRG_CUDA_ARCH="80;90"   # one binary for both A100 and H100
```

Left unset it detects the GPUs of the build machine, which is the right answer whenever you build where you run. Set it by hand on a driverless build node, or when the compute nodes have different GPUs than the login node. An application inherits the value the library was built with, so the two always agree unless the application overrides it.

The pre-built CUDA bundle is deliberately built for the *oldest* supported architecture (`sm_75`, Turing), which is what lets one download serve everything from Turing to Blackwell. Only the bundle's own Kokkos and deal.II kernels are affected by that floor — a few hundred small plumbing kernels, JIT-compiled once and cached. Nothing you compile is: the setting above retargets it all natively.

The one direction that does not work is asking for a GPU *older* than the bundle: DiFfRG's kernels would be right, but Kokkos' own would be above the device and abort. Configure warns about that; the fix is a bundle with a low enough floor, or a source install with `-DKokkos_ARCH_LIST=` (the installer's `--cuda-arch` sets both).

Two consequences of that split are worth knowing:

- Kokkos would print `running kernels compiled for compute capability 7.5 on device with compute capability 9.0` at startup. It detects this with a probe kernel that lives inside the bundled `libkokkoscore`, so the warning is about the bundle and says nothing about DiFfRG's kernels. `DiFfRG::Init()` suppresses that one line and prints an accurate one instead; every other Kokkos warning is untouched.
- `KOKKOS_IMPL_ARCH_NVIDIA_GPU` is baked into the bundle's installed headers and stays at the bundle's floor. It gates only half-precision conversions and 16-byte lock-free atomics, neither of which DiFfRG's double-precision kernels use. A source install (`--mode source`, which builds Kokkos for the architecture you choose) removes even that.

#### Offline CUDA builds on clusters (driverless build nodes)

Building CUDA code needs no GPU — but a build node without the NVIDIA *driver* has no real `libcuda.so.1`, so the linker must resolve the CUDA driver API against the toolkit's **stub** (`$CUDA_HOME/lib64/stubs/libcuda.so`). That is harmless in itself: dynamic linking records only the name `libcuda.so.1`, and on the compute node the loader finds the driver's real library under that name.

The trap is letting the stubs *directory* into a runtime search path — then the loader picks the stub over the driver and every CUDA call fails (`CUDA_ERROR_STUB_LIBRARY`). Expose the stubs to the linker only in ways that are never recorded:

```bash
export LIBRARY_PATH=$CUDA_HOME/lib64/stubs     # link-time only, or:
#   -Wl,-rpath-link,$CUDA_HOME/lib64/stubs
```

Never `-Wl,-rpath,...stubs`, and never leave `.../stubs` in `LD_LIBRARY_PATH` in the job environment (cluster CUDA modules often export it — check!). Verify a build with:

```bash
readelf -d ./my_app | grep -E 'RPATH|RUNPATH'          # must not mention "stubs"
echo $LD_LIBRARY_PATH | tr : '\n' | grep stubs          # in the job env: must be empty
```

An already-poisoned binary is repaired without rebuilding: `patchelf --set-rpath '<rpath without the stubs entry>' ./my_app`. DiFfRG's build system additionally refuses to configure against bundled libraries whose RUNPATH contains a stubs entry.

#### Bounding build parallelism

The dependency builds each run their own nested `make`, which the outer `make -jN` never
reaches. Two caps control them:

- `-DDEALII_MAX_JOBS=<n>` (default 16) — deal.II's own build.
- `-DPETSC_MAX_JOBS=<n>` (default 8) — PETSc's build *and*, via `--with-make-np`, the builds
  of the packages PETSc downloads.

Both are additionally clamped by available RAM (roughly one job per 2 GB). `install_diffrg.sh`
forwards its thread count to both. Raise them only if you have the memory: linking deal.II
is what usually triggers an OOM.

Boost, TBB, HDF5 and SUNDIALS are taken from the system when a viable version is found, and otherwise built from the bundled, pinned sources. For each library `<LIB>` ∈ {`BOOST`, `TBB`, `HDF5`, `SUNDIALS`} you can override this:

- `-D<LIB>_DIR=<prefix>` — use the install at this prefix (a fatal error is raised if it is not usable).
- `-DBUILD_<LIB>=ON` — always build the bundled, pinned copy, ignoring any system install.

The minimum supported versions are Boost ≥ 1.81, oneTBB ≥ 2021, HDF5 ≥ 1.12 and SUNDIALS ≥ 5.4.

The bundled Boost is built from the source archive tracked at `dependencies/boost/boost_1_81_0-diffrg-slim.tar.xz` (the upstream 1.81.0 release with documentation, tests and examples removed), so no network access is needed for it. It is unpacked into the build tree, which costs ~19 500 file writes per build tree. On a slow filesystem, do that once and reuse the result:

```bash
# once per machine
cmake -S . -B build0 -DBUILD_BOOST=ON -DCMAKE_INSTALL_PREFIX=$HOME/opt/diffrg-deps
cmake --build build0 --target boost_dep
# every build thereafter — no unpacking, no Boost rebuild
cmake -S . -B build -DBOOST_DIR=$HOME/opt/diffrg-deps/bundled
```

### Rebuilding the library only

Once the dependencies have been installed by a full build, you usually do not want to rebuild them when working on DiFfRG itself. You can then configure directly from the `DiFfRG/` subfolder, pointing CMake at the already-installed bundled dependencies via `BUNDLED_DIR`:

```bash
$ cd DiFfRG
$ mkdir build && cd build
$ cmake .. -DBUNDLED_DIR=$HOME/.local/share/DiFfRG/bundled/ -DCMAKE_BUILD_TYPE=Release
$ cmake --build ./ -- -j8
```

Note the `/bundled/` suffix: `BUNDLED_DIR` points at the directory the superbuild installed the dependencies into, which defaults to `${CMAKE_INSTALL_PREFIX}/bundled`.

### Verifying your installation

After installation, you can verify that all dependencies are correctly found:
```bash
$ cmake -DBUNDLED_DIR=~/.local/share/DiFfRG/bundled -P ~/.local/share/DiFfRG/cmake/verify_install.cmake
```

This prints a pass/fail table for each dependency.
