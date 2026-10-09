[![Tests](https://img.shields.io/endpoint?style=for-the-badge&url=https://gist.githubusercontent.com/satfra/336ebc0aaa7dc9a0e71ca01dd4361a12/raw/diffrg-tests.json)](https://github.com/satfra/DiFfRG_current/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2412.13043-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2412.13043)
[![Documentation](https://img.shields.io/badge/documentation-2C4AA8?style=for-the-badge&logo=readthedocs&logoColor=white)](https://satfra.github.io/DiFfRG_current/)

<h3>[This is the development repository for DiFfRG. For the current stable version, please visit [the main repository](https://github.com/satfra/DiFfRG)</h3>

# DiFfRG - A Discretization Framework for functional Renormalization Group flows

DiFfRG is a set of tools for the discretization of flow equations arising in the functional Renormalization Group (fRG).
It supports the setup and calculation of large systems of flow equations allowing for complex combinations of vertex and derivative expansions.

For spatial discretizations, i.e. discretizations of field space mostly used for derivative expansions, DiFfRG makes different finite element (FE) methods available. These include:
- Continuous Galerkin FE
- Discontinuos Galerkin FE
- Direct discontinuous Galerkin FE
- Local discontinuous Galerkin FE (including derived finite volume (FV) schemes)

The FEM methods included in DiFfRG are built upon the [deal.ii](https://www.dealii.org/) finite element library, which is highly parallelized and allows for great performance and flexibility.
PDEs consisting of RG-time dependent equations, as well as stationary equations can be solved together during the flow, allowing for techniques like flowing fields in a very accessible way.

Both explicit and implicit time-stepping methods are available and allow thus for efficient RG-time integration in the symmetric and symmetry-broken regime.

We also include a set of tools for the evaluation of integrals and discretization of momentum dependencies.

For an overview, please see the [accompanying paper](https://arxiv.org/abs/2412.13043), the ***[tutorials](https://satfra.github.io/DiFfRG/tutorials/index.html)*** in the [documentation](https://satfra.github.io/DiFfRG/) and the examples in `Examples/`. 

This library has been developed within the [fQCD Collaboration](https://fQCD-Collaboration.github.io).

## Citation

If you use DiFfRG in your scientific work, please cite the corresponding paper:
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

## TL;DR: How to install DiFfRG

From a shell, run
```bash
bash <(curl -s -L https://github.com/satfra/DiFfRG_current/raw/refs/heads/main/install_diffrg.sh)
```
A short install wizard walks through the typical choices — pre-built dependency bundle or from source, where to install and build, toggle features (MPI, GPU, ...), copy the examples and tutorials — then performs the installation.

## Requirements

To compile and run this project, there are very few requirements which you can easily install using your package manager on Linux or MacOS:

- [git](https://git-scm.com/) for external requirements and to clone this repository.
- [CMake](https://www.cmake.org/) for the build systems of DiFfRG, deal.ii and other libraries and [GNU Make](https://www.gnu.org/software/make/) or another generator of your choice.
- A compiler supporting at least the C++20 standard, including a Fortran compiler (e.g. `gfortran`). This project has been tested with [GCC](https://gcc.gnu.org/), `AppleClang`, and `Clang`.
- LAPACK and BLAS in some form, e.g. [OpenBLAS](https://www.openblas.net/). Alternatively, pass `-DBUILD_OpenBLAS=ON` to have DiFfRG build OpenBLAS.
- The GNU Scientific Library [GSL](https://www.gnu.org/software/gsl/).

The following are optional:

- [CUDA](https://developer.nvidia.com/cuda-toolkit) for integration routines on the GPU, which gives a huge speedup for the calculation of fully momentum dependent flow equations (10 - 100x). In case you wish to use CUDA, make sure you have a compiler available on your system compatible with your version of `nvcc`, e.g. `g++`<=13.2 for CUDA 12.5
- [Doxygen](https://www.doxygen.org/) and [graphviz](https://www.graphviz.org/download/) to build the documentation.

All other requirements are bundled and automatically built with DiFfRG.
The framework has been tested with the following systems:

#### Arch Linux
```bash
$ pacman -S git cmake gcc gcc-fortran blas-openblas paraview python doxygen graphviz gsl
```
For a CUDA-enabled build, additionally 
```bash
$ pacman -S cuda
```


#### Rocky Linux
```bash
$ dnf --enablerepo=devel install -y gcc-toolset-12 cmake git openblas-devel doxygen doxygen-latex python3 python3-pip gsl-devel patch
$ scl enable gcc-toolset-12 bash
```

The second line is necessary to switch into a shell where `g++-12` is available

#### Ubuntu
```bash
$ apt-get update
$ apt-get install git cmake gfortran libopenblas-dev paraview build-essential python3 doxygen graphviz libgsl-dev
```
For a CUDA-enabled build, additionally 
```bash
$ apt-get install cuda
```

#### macOS
First, install xcode and homebrew, then run
```bash
$ brew install cmake gcc doxygen paraview graphviz gsl python3 bash
```

*Note: you have to install a newer GNU version of bash, since by default, only version 3.2 is installed, and for installation, PETSc requires a newer version*

#### Windows

If using Windows, instead of running the project directly, it is recommended to use [WSL](https://learn.microsoft.com/en-us/windows/wsl/setup/environment) and then go through the installation as if on Linux (e.g. Arch or Ubuntu).

## Installation

### Interactive installer (recommended)

From the shell, run
```bash
bash <(curl -s -L https://github.com/satfra/DiFfRG_current/raw/refs/heads/main/install_diffrg.sh)
```
A short wizard walks through the typical choices — pre-built dependency bundle or full self-build, install prefix, build folder, features (MPI, GPU, ...), and an optional copy of the examples and tutorials — then performs the complete installation. Every question also has a command-line flag (`--help`) for scripted use.

The **pre-built dependency bundle** downloads deal.II, Kokkos, Boost, TBB, SUNDIALS, HDF5 and friends as a ~50 MB binary from [GitHub Releases](https://github.com/satfra/DiFfRG_current/releases) instead of compiling them for hours; only the DiFfRG library itself is compiled locally (minutes). Bundles come in a CPU-only and a CUDA variant (the wizard offers the latter when it finds an NVIDIA GPU), each with or without MPI (Open MPI, with PETSc and MUMPS; needs the distro's Open MPI development package), and need Linux x86_64 with AVX2 (any consumer CPU from ~2013 on) and glibc ≥ 2.34. For other MPIs (Intel MPI, Cray MPICH, ... — typical on clusters), other CPUs, or other platforms the wizard's self-build path covers the full feature set.

### From source

The wizard's self-build path compiles the full dependency superbuild with your choice of features:
```bash
bash <(curl -s -L https://github.com/satfra/DiFfRG_current/raw/refs/heads/main/install_diffrg.sh) --mode source
```
Add e.g. `--mpi --gpu --threads 6 --prefix ${HOME}/.local/share/DiFfRG --yes` for a non-interactive run (`--help` lists all options). Experts can drive the superbuild directly with CMake — see *Manual installation* below.

On a GPU machine the wizard also asks which NVIDIA architectures to compile for, defaulting to the GPUs it finds (`--cuda-arch "8.0;9.0"` for a scripted run). This matters: CUDA code compiled for the wrong architecture either refuses to start or is JIT-compiled by the driver on every launch, which DiFfRG's large flow kernels cannot afford. See [Choosing the GPU architecture](https://satfra.github.io/DiFfRG_current/getting_started/installation.html) for the details.

### CMake

You can download a script to install DiFfRG locally directly from a CMake file by putting into your `CMakeLists.txt` the lines
```CMake
file(DOWNLOAD
  https://github.com/satfra/DiFfRG_current/raw/refs/heads/main/DiFfRG/cmake/InstallDiFfRG.cmake
  ${CURRENT_BINARY_DIR}/cmake/InstallDiFfRG.cmake)
include(${CURRENT_BINARY_DIR}/cmake/InstallDiFfRG.cmake)
```
This will fetch a script, which will automatically download and install DiFfRG and all of its dependencies to `$HOME/.local/share/DiFfRG`.
If you wish to change this directory, or some other default values, you can set the following optional variables:
```CMake
set(DiFfRG_INSTALL_DIR $ENV{HOME}/.local/share/DiFfRG/)
set(DiFfRG_BUILD_DIR $ENV{HOME}/.local/share/DiFfRG/build/)
set(DiFfRG_SOURCE_DIR $ENV{HOME}/.local/share/DiFfRG/src/)
set(TRY_DiFfRG_VERSION main)
set(PARALLEL_JOBS 8)
```

### Manual installation

You can also manually clone DiFfRG to a directory of your choice
```bash
$ git clone https://github.com/satfra/DiFfRG_current.git
```
Then, create a build directory and run cmake
```bash
$ cd DiFfRG
$ mkdir build
$ cd build
$ cmake ../ -DCMAKE_INSTALL_PREFIX=$HOME/.local/share/DiFfRG/ -DCMAKE_BUILD_TYPE=Release
$ cmake --build ./ -- -j8
```
By default, the library will install itself to `$HOME/.local/share/DiFfRG`, but you can control the destination by pointing `CMAKE_INSTALL_PREFIX` to a directory of your choice.

### Verifying your installation

After installation, you can verify that all dependencies are correctly found:
```bash
$ cmake -DBUNDLED_DIR=$HOME/.local/share/DiFfRG/bundled -P $HOME/.local/share/DiFfRG/cmake/verify_install.cmake
```
This prints a pass/fail table for each dependency, helping diagnose any issues.

## Getting started with simulating fRG flows

For an overview, please see the ***[tutorials](https://satfra.github.io/DiFfRG/tutorials/index.html)*** in the [documentation](https://satfra.github.io/DiFfRG/). A local copy of the documentation is also built automatically when running the setup script, but can also be built manually by running
```bash
$ make documentation
```
inside the `DiFfRG_build` directory (this requires the Python documentation toolchain, see `DiFfRG/documentation/requirements.txt`). You can then find the rendered site in the install directory.

All backend code is contained in the DiFfRG directory.

Several simulations are defined in the Applications directory, which can be used as a starting point for your own simulations.

# Tips and FAQ

## Logfiles and install issues

If DiFfRG fails to build on your machine, first check the appropriate logs. You find the main log at `~/.local/share/DiFfRG/build/DiFfRG.log` if you are using the CMake install. Otherwise, you can redirect the output of the build, e.g.
```bash
cmake --build ./ -- -j8 | tee DiFfRG.log
```
and analyze the result.

If DiFfRG proves to be incompatible with your machine, please open an Issue on GitHub [here](https://github.com/satfra/DiFfRG/issues), or, alternatively, send an email to the author (see the [publication](https://arxiv.org/abs/2412.13043)).


## Contributing

DiFfRG is a work in progress. If you find some feature missing, a bug, or some other kind of improvement, you can get involved in the further development of DiFfRG. 

Thanks to the collaborative nature of GitHub, you can simply fork the project and work on a private copy on your own GitHub account. Feel also encouraged to open an [issue](https://github.com/satfra/DiFfRG/issues), or if you already have a (partially) ready contribution, open a [pull request](https://github.com/satfra/DiFfRG/pulls).


## Configuration files

A DiFfRG simulation requires you to provide a valid `parameters.json` or `parameters.toml` file in the execution path, or alternatively provide another JSON/TOML-file using the `-p` flag (see below).

To generate a "stock" `parameters.json` in the current folder, you can call any DiFfRG application as
```bash
$ ./my_simulation --generate-parameter-file
```
Before usage, don't forget to put in the parameters you defined in your own simulation!


## Progress output

To monitor the progress of the simulation, one can set the `verbosity` parameter either in the parameter file,
```json
{
  "output": {
    "verbosity": 1
  },
}
```
or from the CLI,
```bash
$ ./my_simulation -si /output/verbosity=1
```
Levels 1--4 use compact, aggregated progress output: the console is updated at most once per second for each topic,
and the run log receives the same aggregate every second. Level 2 adds Jacobian, linear-solver, and
solver-specific diagnostics; level 3 adds factorization and output timings. Level 4 remains rate-limited. Level 5 is
the explicit debugging mode: every progress event is printed and may substantially slow a run. Progress records are
kept to 100 columns, with detailed diagnostics on at most one continuation line.


## Modifying parameters from the CLI

Any DiFfRG simulation using the `DiFfRG::ConfigurationHelper` class can be asked to give some syntax pertaining to the configuration:
```bash
$ ./my_simulation --help
This is a DiFfRG simulation. You can pass the following optional parameters:
  --help                      shows this text
  --generate-parameter-file   generates a parameter file with some default values
  -p                          specifiy a parameter file other than the standard parameter.json
  -sd                         overwrite a double parameter. This should be in the format '-s physical/T=0.1'
  -si                         overwrite an integer parameter. This should be in the format '-s physical/Nc=1'
  -sb                         overwrite a boolean parameter. This should be in the format '-s physical/use_sth=true'
  -ss                         overwrite a string parameter. This should be in the format '-s physical/a=hello'
```
e.g.
```bash
$ ./my_simulation -sd /physical/Lambda=1.0
```

## Timestepper choice

In general, the `IDA` timestepper from the `SUNDIALS`-suite has proven to be the optimal choice for any fRG-flow with convexity restoration. Additionally, this solver allows for out-of-the-box solving of additional algebraic systems, which is handy for more complicated fRG setups.

If solving purely variable-dependent systems, one of the `Boost` time steppers, `Boost_RK45`, `Boost_RK78` or `Boost_ABM`. The latter is especially excellent for extremely large systems which have no extremely fast dynamics, but lacks adaptive timestepping. In practice, choosing `Boost_ABM` over one of the RK steppers may speed up a Yang-Mills simulation with full momentum dependences by more than a factor of 10 while maintaining accuracy perfectly.

For systems with both spatial discretisations and variables, consider one of the implicit-explicit mixtures, `SUNDIALS_IDA_Boost_RK45`,  `SUNDIALS_IDA_Boost_RK78` or `SUNDIALS_IDA_Boost_ABM`.

# Other Libraries used

The following third-party libraries are utilised by DiFfRG. They are automatically built and installed DiFfRG during the build process.

- The main backend for field-space discretization is [deal.II](https://www.dealii.org/), which provides the entire FE/FV-machinery as well as many other utility components.
- For performant and convenient calculation of Jacobian matrices we use the [autodiff](https://github.com/autodiff/autodiff) library, which implements automatic forward and backwards differentiation in C++ and also in CUDA.
- [Kokkos](https://github.com/kokkos/kokkos), a performance portability framework for shared-memory parallelization on GPU and CPU. We use it for the integration routines for flow equations.
- Time integration relies heavily on the [SUNDIALS](https://computing.llnl.gov/projects/sundials) suite, specifically on the IDAs solver.
- [Boost](https://www.boost.org/) provides explicit time-stepping and various math algorithms.
- [Catch2](https://github.com/catchorg/Catch2) for unit testing.
- [spdlog](https://github.com/gabime/spdlog) for logging.
- [Boost](https://www.boost.org/) provides explicit time-stepping and various math algorithms.
- [Eigen](https://eigen.tuxfamily.org/) for linear-algebra related tasks.
- [HDF5](https://www.hdfgroup.org/solutions/hdf5/) for hierarchical data format storage and I/O operations.