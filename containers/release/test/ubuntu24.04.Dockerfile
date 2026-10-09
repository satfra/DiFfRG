# Environment for validating a pre-built DiFfRG dependency bundle on Ubuntu 24.04.
# Deliberately WITHOUT boost/tbb/hdf5/sundials dev packages: the bundle must be
# self-contained. Driven by containers/release/test-tarball.sh.
FROM ubuntu:24.04
LABEL type=diffrg-deps-release-test

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt-get -y install --no-install-recommends \
        git cmake build-essential gfortran \
        libopenblas-dev libgsl-dev zlib1g-dev \
        python3 patch ca-certificates curl zstd file pkg-config \
    && rm -rf /var/lib/apt/lists/*

# MPI-variant validation (test-tarball.sh -m): the host Open MPI consumers build with.
ARG mpi=none
RUN if [ "${mpi}" = openmpi ]; then \
      apt-get -y update && apt-get -y install --no-install-recommends libopenmpi-dev openmpi-bin \
      && rm -rf /var/lib/apt/lists/*; \
    fi
