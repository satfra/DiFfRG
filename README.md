[![arXiv](https://img.shields.io/badge/arXiv-2412.13043-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2412.13043)
[![Doxygen](https://img.shields.io/badge/doxygen-2C4AA8?style=for-the-badge&logo=c%2B%2B&logoColor=white)](https://satfra.github.io/DiFfRG/cpp/index.html)
[![Wolfram](https://img.shields.io/badge/wolfram_doc-cf1c10?style=for-the-badge&logo=wolfram)](https://satfra.github.io/DiFfRG/wolfram/html/guide/DiFfRG.html)
[![Python](https://img.shields.io/badge/python_doc-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://satfra.github.io/DiFfRG/python/index.html)

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

Both explicit and implicit timestepping methods are available and allow thus for efficient RG-time integration in the symmetric and symmetry-broken regime.

We also include a set of tools for the evaluation of integrals and discretization of momentum dependencies.

For an overview, please see the [accompanying paper](https://arxiv.org/abs/2412.13043), the ***[tutorial page](https://satfra.github.io/DiFfRG/cpp/TutorialTOC.html)*** in the [documentation](https://satfra.github.io/DiFfRG/cpp/index.html) and the examples in `Examples/`. 

## Citation

If you use DiFfRG in your scientific work, please cite the corresponding paper:
```
@article{Sattler:2024ozv,
    author = "Sattler, Franz R. and Pawlowski, Jan M.",
    title = "{DiFfRG: A Discretisation Framework for functional Renormalisation Group flows}",
    eprint = "2412.13043",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "12",
    year = "2024"
}
```


## Requirements

To compile and run this project, there are very few requirements which you can easily install using your package manager on Linux or MacOS:

- [git](https://git-scm.com/) for external requirements and to clone this repository.
- [CMake](https://www.cmake.org/) for the build systems of DiFfRG, deal.ii and other libraries.
- [GNU Make](https://www.gnu.org/software/make/) or another generator of your choice.
- A compiler supporting at least the C++20 standard. This project is only tested using the [GCC](https://gcc.gnu.org/) compiler suite, as well as with `AppleClang`, but in principle, ICC or standard Clang should also work.
- LAPACK and BLAS in some form, e.g. [OpenBlas](https://www.openblas.net/).
- The GNU Scientific Library [GSL](https://www.gnu.org/software/gsl/). If not found by DiFfRG, it will try to install it by itself.
- [Doxygen](https://www.doxygen.org/) and [graphviz](https://www.graphviz.org/download/) to build the documentation.

The following requirements are optional:
- [Python](https://www.python.org/) is used in the library for visualization purposes. Furthermore, adaptive phase diagram calculation is implemented as a python routine.
- [ParaView](https://www.paraview.org/), a program to visualize and post-process the vtk data saved by DiFfRG when treating FEM discretizations.
- [CUDA](https://developer.nvidia.com/cuda-toolkit) for integration routines on the GPU, which gives a huge speedup for the calculation of fully momentum dependent flow equations (10 - 100x). In case you wish to use CUDA, make sure you have a compiler available on your system compatible with your version of `nvcc`, e.g. `g++`<=13.2 for CUDA 12.5

All other requirements are bundled and automatically built with DiFfRG.
The framework has been tested with the following systems:

#### Arch Linux
```bash
$ pacman -S git cmake gcc blas-openblas blas64-openblas paraview python doxygen cuda graphviz gsl
```
A further installation of the `gcc12` package from the AUR is necessary for setups with CUDA; with your AUR helper of choice (here `yay`) this can be done with
```bash
$ yay -S gcc12
```

#### Rocky Linux
```bash
$ dnf --enablerepo=devel install -y gcc-toolset-12 cmake git openblas-devel doxygen doxygen-latex python3 python3-pip gsl-devel
$ scl enable gcc-toolkit-12 bash
```

The second line is necessary to switch into a shell where `g++-12` is available

#### Ubuntu
```bash
$ apt-get update
$ apt-get install git cmake libopenblas-dev paraview build-essential python3 doxygen libeigen3-dev cuda graphviz libgsl-dev
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
$ bash -i  build.sh -j8 -cf -i /opt/DiFfRG
```
The `build_DiFfRG.sh` bash script will build and setup the DiFfRG project and all its requirements. This can take up to half an hour as the deal.ii library is quite large.
This script has the following options:
-  `-f`              Perform a full build and install of everything without confirmations.
-  `-c`              Use CUDA when building the DiFfRG library.
-  `-i <directory>`  Set the installation directory for the library.
-  `-j <threads>`    Set the number of threads passed to make and git fetch.
-  `--help`          Display this information.

Depending on your amount of CPU cores, you should adjust the `-j` parameter which indicates the number of threads used in the build process. Note that choosing this too large may lead to extreme RAM usage, so tread carefully.

As soon as the build has finished, you can find a full install of the library in the `DiFfRG_install` subdirectory.

If you have changes to the library code, you can update the library by running
```bash
$ bash -i update_DiFfRG.sh -clm -j8 -i /opt/DiFfRG
```
where once again the `-j` parameter should be adjusted to your amount of CPU cores.
The `update_DiFfRG.sh` script takes the following optional arguments:
- `-c`               Use CUDA when building the DiFfRG library.
- `-l`               Build the DiFfRG library.
- `-i <directory>`   Set the installation directory for the library.
- `-j <threads>`     Set the number of threads passed to make and git fetch.
- `-m`               Install the Mathematica package locally.
- `--help`           Display this information.

## Getting started with simulating fRG flows

For an overview, please see the ***[tutorial page](https://satfra.github.io/DiFfRG/cpp/TutorialTOC.html)*** in the [documentation](https://satfra.github.io/DiFfRG/cpp/index.html). A local documentation is also always built automatically when running the setup script, but can also be built manually by running
```bash
$ make documentation
```
inside the `DiFfRG_build` directory. You can find then a code reference in the top directory.

All backend code is contained in the DiFfRG directory.

Several simulations are defined in the Applications directory, which can be used as a starting point for your own simulations.

# Tips

## Progress output

To see how fast the simulation progresses, one can set the `verbosity` parameter either in the parameter file,
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

## Modifying parameters from the CLI

Any DiFfRG simulation using the `DiFfRG::ConfigurationHelper` class can be asked to give some syntax pertaining to the configuration:
```
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

## Timestepper choice

In general, the `IDA` timestepper from the `SUNDIALS`-suite has proven to be the optimal choice for any fRG-flow with convexity restoration. Additionally, this solver allows for out-of-the-box solving of additional algebraic systems, which is handy for more complicated fRG setups.

However, a set of alternative steppers is also provided - `ARKode` provides adaptive explicit, implicit and ImEx steppers, and furthermore explicit and implicit Euler, as well as TRBDF2 are seperately implemented. The use of the latter three is however discouraged, as the `SUNDIALS`-timesteppers always give better performace.

If solving purely variable-dependent systems, one of the `Boost` time steppers, `Boost_RK45`, `Boost_RK78` or `Boost_ABM`. The latter is especially excellent for extremely large systems which have no extremely fast dynamics, but lacks adaptive timestepping. In practice, choosing `Boost_ABM` over one of the RK steppers may speed up a Yang-Mills simulation with full momentum dependences by more than a factor of 10.

# Other Libraries used

- The main backend for field-space discretization is [deal.II](https://www.dealii.org/), which provides the entire FEM-machinery as well as many other utility components.
- For performant and convenient calculation of Jacobian matrices we use the [autodiff](https://github.com/autodiff/autodiff) library, which implements automatic forward and backwards differentiation in C++ and also in CUDA.
- Time integration relies heavily on the [SUNDIALS](https://computing.llnl.gov/projects/sundials) suite, specifically on the IDAs and ARKODE solvers.
- [Rapidcsv](https://github.com/d99kris/rapidcsv) for quick processing of .csv files.
- [Catch2](https://github.com/catchorg/Catch2) for unit testing.
- [RMM](https://github.com/rapidsai/rmm), a memory manager for CUDA, which is used for GPU-accelerated loop integrations.
- [QMC](https://github.com/mppmu/qmc) for adaptive Quasi-Monte-Carlo integration.
- [spdlog](https://github.com/gabime/spdlog) for logging.
