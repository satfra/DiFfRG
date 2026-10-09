[![Tests](https://img.shields.io/endpoint?style=for-the-badge&url=https://gist.githubusercontent.com/satfra/336ebc0aaa7dc9a0e71ca01dd4361a12/raw/diffrg-tests.json)](https://github.com/satfra/DiFfRG/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2412.13043-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2412.13043)
[![Documentation](https://img.shields.io/badge/documentation-2C4AA8?style=for-the-badge&logo=readthedocs&logoColor=white)](https://satfra.github.io/DiFfRG/)

# DiFfRG - A Discretisation Framework for functional Renormalisation Group flows

DiFfRG is a C++20 framework for discretising and solving functional Renormalisation Group (fRG) flows, including very large systems that combine derivative and vertex expansions.

- **Field-space PDEs** (derivative expansion): continuous, discontinuous and local discontinuous Galerkin finite elements, and Kurganov–Tadmor finite volumes, built on [deal.II](https://www.dealii.org/).
- **Momentum-dependent vertices** (vertex expansion): interpolation on momentum grids and a loop-integration framework, including finite-temperature Matsubara sums.
- **RG-time integration**: implicit (SUNDIALS IDA), explicit (Boost) and mixed implicit/explicit steppers. Algebraic equations can be solved alongside the flow, e.g. for flowing fields.
- **Code generation**: a Mathematica package derives the flow equations and emits the C++ kernels.

Everything runs in parallel across nodes with MPI, and within a node on CPU threads or GPUs.

DiFfRG has been developed within the [fQCD Collaboration](https://fQCD-Collaboration.github.io).

## Quick start

1. Install DiFfRG with the interactive installer:
   ```bash
   bash <(curl -s -L https://github.com/satfra/DiFfRG/raw/refs/heads/main/install_diffrg.sh)
   ```
   It asks for a pre-built dependency bundle or a build from source, the install location, features (MPI, GPU, ...), and whether to copy the examples, tutorials and Mathematica package. Make sure the [requirements](#requirements) are installed first.
2. Build and run the first tutorial (from the examples folder the installer created):
   ```bash
   cd Tutorials/tut1
   ./build.sh -j4
   cd build && ./tut1
   ```
3. Continue with the [tutorials](https://satfra.github.io/DiFfRG/tutorials/index.html).

## What a simulation looks like

A simulation is a short `main` that chains a few template types. The physics lives in the user-written `Model` class (initial condition, fluxes, sources, ...):

```cpp
#include <DiFfRG/DiFfRG.hh>
#include "model.hh"
using namespace DiFfRG;

int main(int argc, char *argv[])
{
  const auto json = Init(argc, argv).get_configuration_helper().get_json();

  using Discretization = CG::Discretization<MyModel, RectangularMesh<MyModel::dim>>;
  using Assembler = CG::Assembler<Discretization>;
  using TimeStepper = TimeStepperSUNDIALS_IDA<Assembler>;

  MyModel model(json);
  RectangularMesh<MyModel::dim> mesh{Config::ConfigurationMesh<MyModel::dim>(json)};
  OutputSession<Assembler> output(json);
  const auto log = output.report_port();
  Discretization discretization(mesh, json, log);
  Assembler assembler(discretization, model, json);
  TimeStepper time_stepper(json, assembler, output);

  FE::FlowingVariables initial_condition(discretization);
  initial_condition.interpolate(model);
  time_stepper.run(initial_condition, 0., json.get_double("/timestepping/final_time"));
}
```

Switching to another discretisation or time stepper means changing one `using` line.

## Documentation

- The ***[documentation](https://satfra.github.io/DiFfRG/)***: tutorials, guides, and the C++, Python and Mathematica API references.
- The [accompanying paper](https://arxiv.org/abs/2412.13043) gives an overview of the methods.
- `Tutorials/` builds up the features step by step. `Examples/` has complete physics applications (O(N) model at finite T, quark-meson model, Yang-Mills) to start your own simulations from.

A local copy of the documentation is built by configuring with `-DDIFFRG_DOCS=ON` and running `make documentation`. This requires the Python toolchain in `DiFfRG/documentation/requirements.txt`. The rendered site ends up in the install directory.

### Mathematica package

Momentum-dependent flows are usually not written by hand. The DiFfRG Mathematica package (`DiFfRG/Mathematica/`) interfaces with [FunKit](https://github.com/satfra/FunKit), a package to derive flow equations from a given field theory and truncation.
Then, it generates the C++ kernels and CMake files that a simulation links against. The installer offers to install the Mathematica package into your Wolfram applications directory; FunKit is installed on first load if missing. Each example in `Examples/` contains the derivation file it was generated from.

### Python package

The Python package (`DiFfRG/python/`) reads simulation output (CSV, HDF5, VTK), provides plotting helpers, and runs parameter scans and phase diagrams. Install it from the DiFfRG install directory with
```bash
pip install $HOME/.local/share/DiFfRG/python
```

## Installation

### Requirements

You need:

- [git](https://git-scm.com/), [CMake](https://www.cmake.org/), and [GNU Make](https://www.gnu.org/software/make/) or another CMake generator.
- A C++20 compiler and a Fortran compiler (e.g. `gfortran`). Tested with [GCC](https://gcc.gnu.org/), Clang and AppleClang.
- LAPACK and BLAS, e.g. [OpenBLAS](https://www.openblas.net/) (or pass `-DBUILD_OpenBLAS=ON` to build it). With the macOS bundle, the system's Accelerate framework is used.
- The GNU Scientific Library [GSL](https://www.gnu.org/software/gsl/).

Optional:

- [CUDA](https://developer.nvidia.com/cuda-toolkit) for GPU integration, which speeds up fully momentum-dependent flows by 10–100×. Your host compiler must be supported by your `nvcc` version.
- Open MPI development packages, if you want the MPI variant of the pre-built bundle.

All other dependencies (deal.II, Kokkos, Boost, TBB, SUNDIALS, HDF5, ...) are either part of the pre-built bundle or built automatically.

<details>
<summary><h3>Package lists per system</h3> (click to expand)</summary>

#### Arch Linux
```bash
pacman -S git cmake gcc gcc-fortran blas-openblas python gsl
pacman -S cuda    # optional, for GPU support
```

#### Rocky Linux
```bash
dnf --enablerepo=devel install -y gcc-toolset-12 cmake git openblas-devel python3 python3-pip gsl-devel patch
scl enable gcc-toolset-12 bash    # opens a shell where g++-12 is available
```

#### Ubuntu
```bash
apt-get update
apt-get install git cmake gfortran libopenblas-dev build-essential python3 libgsl-dev
apt-get install cuda    # optional, for GPU support
```

#### macOS
Install Xcode and Homebrew, then
```bash
brew install cmake gcc gsl python3 bash
```
The newer `bash` is needed because macOS ships version 3.2, which is too old for the PETSc build.

#### Windows
Use [WSL](https://learn.microsoft.com/en-us/windows/wsl/setup/environment) and follow the Linux instructions (e.g. Ubuntu).

</details>

### Interactive installer (recommended)

```bash
bash <(curl -s -L https://github.com/satfra/DiFfRG/raw/refs/heads/main/install_diffrg.sh)
```
Every question of the wizard also has a command-line flag for scripted use (`--help` lists them).

**Pre-built dependency bundle.** Instead of compiling the dependencies for hours, the installer downloads them as a ~50 MB binary from [GitHub Releases](https://github.com/satfra/DiFfRG/releases). Only DiFfRG itself is compiled locally, which takes minutes. Available bundles:

- Linux x86_64: CPU-only or CUDA, each with or without MPI (Open MPI with PETSc and MUMPS). Needs a CPU with AVX2 (any consumer CPU from ~2013 on) and glibc ≥ 2.34.
- macOS arm64 (Apple Silicon): CPU-only.

**Build from source.** For other platforms, other MPI implementations (Intel MPI, Cray MPICH, ..., typical on clusters), or full control over features, build the whole dependency stack:
```bash
bash <(curl -s -L https://github.com/satfra/DiFfRG/raw/refs/heads/main/install_diffrg.sh) --mode source
```
Add e.g. `--mpi --gpu --threads 6 --prefix ${HOME}/.local/share/DiFfRG --yes` for a non-interactive run.

**GPU architecture.** On a GPU machine the installer asks which NVIDIA architectures to compile for, defaulting to the GPUs it finds (`--cuda-arch "8.0;9.0"` for a scripted run). Get this right: code compiled for the wrong architecture either does not start or is recompiled by the driver at every launch, which is very slow for DiFfRG's large kernels. See [Choosing the GPU architecture](https://satfra.github.io/DiFfRG/getting_started/installation.html).

### Installing from a CMake project

To have your own project install DiFfRG on demand, add to its `CMakeLists.txt`:
```CMake
file(DOWNLOAD
  https://github.com/satfra/DiFfRG/raw/refs/heads/main/DiFfRG/cmake/InstallDiFfRG.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/InstallDiFfRG.cmake)
include(${CMAKE_CURRENT_BINARY_DIR}/cmake/InstallDiFfRG.cmake)
```
This downloads and installs DiFfRG with all dependencies to `$HOME/.local/share/DiFfRG`. Optional settings, shown with their defaults:
```CMake
set(DiFfRG_INSTALL_DIR $ENV{HOME}/.local/share/DiFfRG/)
set(DiFfRG_BUILD_DIR $ENV{HOME}/.local/share/DiFfRG/build/)
set(DiFfRG_SOURCE_DIR $ENV{HOME}/.local/share/DiFfRG/src/)
set(TRY_DiFfRG_VERSION main)
set(PARALLEL_JOBS 8)
```

### Manual installation

```bash
git clone https://github.com/satfra/DiFfRG.git
cd DiFfRG
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$HOME/.local/share/DiFfRG/ -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
```
This builds all dependencies and DiFfRG, and installs to `CMAKE_INSTALL_PREFIX` (default `$HOME/.local/share/DiFfRG`).

### Verifying the installation

```bash
cmake -DBUNDLED_DIR=$HOME/.local/share/DiFfRG/bundled -P $HOME/.local/share/DiFfRG/cmake/verify_install.cmake
```
prints a pass/fail table for each dependency.

## Tips and FAQ

### The build fails

Check the logs first. With the CMake install, the main log is `~/.local/share/DiFfRG/build/DiFfRG.log`. Otherwise, capture the build output:
```bash
cmake --build build -j8 2>&1 | tee DiFfRG.log
```
If DiFfRG does not work on your machine, please [open an issue](https://github.com/satfra/DiFfRG/issues).

### Parameter files and command-line options

A simulation reads `parameter.json` or `parameter.toml` from the working directory, or the file given with `-p`. Generate one with default values with
```bash
./my_simulation --generate-parameter-file
```
and then add the parameters your own model defines. Any parameter can be overridden from the command line by its path, with `-sd` (double), `-si` (integer), `-sb` (bool) or `-ss` (string):
```bash
./my_simulation -sd /physical/Lambda=1.0
```
`./my_simulation --help` lists all options.

### Progress output

Set `/output/verbosity` (e.g. `-si /output/verbosity=1`). Levels 1–4 print aggregated progress at most once per second, adding solver diagnostics (2) and timings (3). Level 5 prints every event and can slow down the run noticeably.

### Restarting from snapshots

A run can save its complete state at chosen RG scales, and a later run can continue from such a snapshot instead of starting again at Λ:
```bash
./my_simulation --snapshots-k 2.0,1.0                   # write snapshots at k = 2.0 and 1.0
./my_simulation --restart output/output_snapshot_000.h5  # continue a run from its first snapshot
```
A restart takes its configuration from the snapshot, and `-sd`/`-si`/`-sb`/`-ss` can change it. This continues runs killed by a wall-time limit, and saves the UV part of the flow in parameter scans: one seed run above k ~ T, μ can be continued for every point of a phase diagram. Changed parameters must not matter above the snapshot scale for the result to be exact. See [Flow snapshots and restarts](https://satfra.github.io/DiFfRG/getting_started/snapshots.html).

### Which time stepper?

- **`TimeStepperSUNDIALS_IDA`**: the default for field-space PDEs, in particular flows with convexity restoration. It can also solve additional algebraic equations.
- **`TimeStepperBoost_RK45`, `_RK78`, `_ABM`**: for systems of momentum-dependent couplings only (no field-space discretisation). `Boost_ABM` suits very large systems without fast dynamics; for a fully momentum-dependent Yang-Mills flow it can be more than 10× faster than the RK steppers at the same accuracy.
- **`TimeStepperSUNDIALS_IDA_Boost_RK45`, `_RK78`, `_ABM`**: for systems with both. The field-space part is integrated implicitly, the vertex expansion explicitly.

## Citation

If you use DiFfRG in your scientific work, please cite:
```
@article{Sattler:2024ozv,
    author = "Sattler, Franz R. and Pawlowski, Jan M.",
    title = "{DiFfRG: A discretisation framework for functional renormalisation group flows}",
    eprint = "2412.13043",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    doi = "10.1016/j.cpc.2026.110262",
    journal = "Comput. Phys. Commun.",
    volume = "327",
    pages = "110262",
    year = "2026"
}
```

## Contributing

Bug reports, feature requests and contributions are welcome. Open an [issue](https://github.com/satfra/DiFfRG/issues), or fork the repository and open a [pull request](https://github.com/satfra/DiFfRG/pulls).

## License and acknowledgements

DiFfRG is distributed under the GNU General Public License v3 or later (see [LICENSE](LICENSE)), with an additional permission for linking against NVIDIA CUDA (see [LICENSE.exceptions](LICENSE.exceptions)). Changes between versions are listed in [CHANGELOG.md](CHANGELOG.md).

DiFfRG builds on [deal.II](https://www.dealii.org/) (finite elements), [Kokkos](https://github.com/kokkos/kokkos) (CPU/GPU parallelism), [SUNDIALS](https://computing.llnl.gov/projects/sundials) (implicit time stepping), [Boost](https://www.boost.org/) (explicit time stepping, math), [autodiff](https://github.com/autodiff/autodiff) (Jacobians), [Eigen](https://eigen.tuxfamily.org/), [HDF5](https://www.hdfgroup.org/solutions/hdf5/), [spdlog](https://github.com/gabime/spdlog) and [Catch2](https://github.com/catchorg/Catch2).
