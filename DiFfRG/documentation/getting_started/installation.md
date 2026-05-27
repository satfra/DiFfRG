# Installation {#Installation}

To compile and run this project, there are very few requirements which you can easily install using your package manager on Linux or MacOS:

- [git](https://git-scm.com/) for external requirements and to clone this repository.
- [CMake](https://www.cmake.org/) for the build systems of DiFfRG, deal.ii and other libraries.
- [GNU Make](https://www.gnu.org/software/make/) or another generator of your choice.
- A compiler supporting at least the C++20 standard. This project is only tested using the [GCC](https://gcc.gnu.org/) compiler suite, as well as with `AppleClang`, but in principle, ICC or standard Clang should also work.
- LAPACK and BLAS in some form, e.g. [OpenBlas](https://www.openblas.net/).
- The GNU Scientific Library [GSL](https://www.gnu.org/software/gsl/). If not found by DiFfRG, it will try to install it by itself.
- [Doxygen](https://www.doxygen.org/) and [graphviz](https://www.graphviz.org/download/) to build the documentation.

The following requirements are optional:
- A Fortran compiler (e.g. `gfortran`). It is *recommended* but not required: it makes deal.II's LAPACK detection robust.
- [Boost](https://www.boost.org/) (>= 1.80). A compatible system Boost is used automatically if found; otherwise it is built from source (see [Setup](#setup) for how to control this). If the system boost is not compatible, DiFfRG will build the bundled one automatically.
- [oneTBB](https://github.com/uxlfoundation/oneTBB) (>= 2021) and [SUNDIALS](https://computing.llnl.gov/projects/sundials) (>= 5.4.0). Like Boost, a compatible system copy is used automatically if found, otherwise it is built from source (see [Setup](#setup)).
- [Python](https://www.python.org/) is used in the library for visualization purposes. Furthermore, adaptive phase diagram calculation is implemented as a python routine.
- [ParaView](https://www.paraview.org/), a program to visualize and post-process the vtk data saved by DiFfRG when treating FEM discretizations.
- [CUDA](https://developer.nvidia.com/cuda-toolkit) for integration routines on the GPU, which gives a huge speedup for the calculation of fully momentum dependent flow equations (10 - 100x). In case you wish to use CUDA, make sure you have a compiler available on your system compatible with your version of `nvcc`, e.g. `g++`<=13.2 for CUDA 12.5

All other requirements are bundled and automatically built with DiFfRG.
The framework has been tested with the following systems:

#### Arch Linux
```bash
$ pacman -S git cmake gcc blas-openblas blas64-openblas paraview python doxygen graphviz gsl
```
For a CUDA-enabled build, additionally install
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
$ apt-get install git cmake libopenblas-dev paraview build-essential python3 doxygen libeigen3-dev graphviz libgsl-dev
```
For a CUDA-enabled build, additionally
```bash
$ apt-get install cuda
```

#### MacOS
First, install xcode and homebrew, then run
```bash
$ brew install cmake doxygen paraview eigen graphviz gsl
```

#### Windows

If using Windows, instead of running the project directly, it is recommended to use [WSL](https://learn.microsoft.com/en-us/windows/wsl/setup/environment) and then go through the installation as if on Linux (e.g. Arch or Ubuntu).

#### Docker and other container runtime environments

Although a native install should be unproblematic in most cases, the setup with CUDA functionality may be daunting. Especially on high-performance clusters, and also depending on the packages available for  chosen distribution, it may be much easier to work with the framework inside a container.

The specific choice of runtime environment is up to the user, however we provide a small build script to create docker container in which DiFfRG will be built.
To do this, you will need `docker`, `docker-buildx` and the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#) in case you wish to create a CUDA-compatible image.

For a CUDA-enabled build, run
```build
$ bash setup_docker.sh -c 12.5.1 -j8
```
in the above, you may want to replace the version `12.5.1` with another version you can find on [docker hub at nvidia/cuda ](https://hub.docker.com/r/nvidia/cuda/tags).
Alternatively, for a CUDA-less build, run simply
```build
$ bash setup_docker.sh -j8
```

If using other environments, e.g. [ENROOT](https://github.com/NVIDIA/enroot), the preferred approach is simply to build an image on top of the [CUDA images by NVIDIA](https://hub.docker.com/r/nvidia/cuda/tags). Optimal compatibility is given using `nvidia/cuda:12.5.1-devel-rockylinux`. Proceed with the installation setup for  Rocky Linux above.

For example, with ENROOT a DiFfRG image can be built by following these steps:
```bash
$ enroot import docker://nvidia/cuda:12.5.1-devel-rockylinux9
$ enroot create --name DiFfRG nvidia+cuda+12.5.1-devel-rockylinux9.sqsh
$ enroot start --root --rw -m ./:/DiFfRG_source DiFfRG bash
```
Afterwards, one proceeds with the above Rocky Linux setup.

## Setup

If all requirements are met, you can clone the git to a directory of your choice,
```bash
$ git clone https://github.com/satfra/DiFfRG.git
```
and start the build after switching to the git directory.
```bash
$ cd DiFfRG
$ bash -i  build.sh -j8 -i ~/.local/share/DiFfRG
```
The `build.sh` bash script will build and setup the DiFfRG project and all its requirements. This can take up to half an hour as the deal.ii library is quite large.
This script has the following options:
-  `-f`              Perform a full build and install of everything without confirmations.
-  `-c`              Use CUDA when building the DiFfRG library.
-  `-i <directory>`  Set the installation directory for the library.
-  `-j <threads>`    Set the number of threads passed to the build.
-  `-b <directory>`  Use the Boost installation at this prefix instead of building one.
-  `-t <directory>`  Use the TBB installation at this prefix instead of building one.
-  `-s <directory>`  Use the SUNDIALS installation at this prefix instead of building one.
-  `--help`          Display this information.

Depending on your amount of CPU cores, you should adjust the `-j` parameter which indicates the number of threads used in the build process. Note that choosing this too large may lead to extreme RAM usage, so tread carefully - DiFfRG will try to auto-detect an appropriate value if `-j` is not set.

As soon as the build has finished, you can find a full install of the library in the directory passed to `-i`; the default is `~/.local/share/DiFfRG`.

If you have changes to the library code, you can update the library by running
```bash
$ bash -i update_DiFfRG.sh -c -j8 -i ~/.local/share/DiFfRG
```
where once again the `-j` parameter should be adjusted to your amount of CPU cores.
The `update_DiFfRG.sh` script takes the following optional arguments:
- `-c`               Use CUDA when building the DiFfRG library.
- `-i <directory>`   Set the installation directory for the library.
- `-j <threads>`     Set the number of threads passed to the build.
- `-m`               Install the Mathematica package locally.
- `--help`           Display this information.

### Choosing the compiler

The compiler is selected through the standard `CC`/`CXX` (and, optionally, `FC`) environment variables, which are propagated to DiFfRG and every bundled dependency:
```bash
$ CXX=clang++ CC=clang bash -i build.sh -j8 -i ~/.local/share/DiFfRG
```
You can also set these in the `config` file (see the commented `export CC=…`/`export CXX=…` lines there). Because CMake caches the compiler on the first configure, switching compilers afterwards requires a clean build tree (run `clear_all.sh`).

### Bundled dependencies: Boost, TBB and SUNDIALS

By default a compatible system copy of each is used if found, otherwise it is built from source:
- **Boost** ≥ 1.80 (used by deal.II and DiFfRG; only adopted if deal.II can consume it — see note below),
- **TBB** ≥ 2021 (used by deal.II and DiFfRG),
- **SUNDIALS** ≥ 5.4.0 (used by deal.II only).

Each can be controlled independently, e.g.:
```bash
# Use a specific prefix for one or more dependencies:
$ bash -i build.sh -j8 -b /usr -t /usr -s /opt/sundials -i ~/.local/share/DiFfRG
#   (equivalently: cmake ... -DBOOST_DIR=/usr -DTBB_DIR=/usr -DSUNDIALS_DIR=/opt/sundials)
# Force building a bundled, pinned dependency (ignore any system copy):
$ BUILD_BOOST=1 BUILD_TBB=1 BUILD_SUNDIALS=1 bash -i build.sh -j8 -i ~/.local/share/DiFfRG
#   (equivalently: cmake ... -DBUILD_BOOST=ON -DBUILD_TBB=ON -DBUILD_SUNDIALS=ON)
```
Note on Boost: deal.II uses the legacy module-mode `FindBoost`, which needs *compiled* Boost component
libraries including `boost_system`. Very recent Boost releases ship `Boost.System` header-only and
cannot be consumed by deal.II here, so the bundled Boost is built automatically (and an incompatible
`-b` prefix is a hard error). TBB and SUNDIALS have no such restriction. oneTBB and SUNDIALS each build
in only ~1–2 minutes, so the savings from a system copy are modest — the main benefit is reusing
existing/HPC-module installs.
