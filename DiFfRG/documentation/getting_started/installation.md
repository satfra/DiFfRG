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

## Setup

If all requirements are met, you can clone the git to a directory of your choice,
```bash
$ git clone https://lin0.thphys.uni-heidelberg.de:4443/frg-codes/DiFfRG.git
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

As soon as the build has finished, you can find the build folder in the `DiFfRG_build` subfolder and a full install of the library in `/opt/DiFfRG`.

If you have changes to the library code, you can always update the library by running
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