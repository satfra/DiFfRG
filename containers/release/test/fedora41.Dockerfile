# Environment for validating a pre-built DiFfRG dependency bundle on Fedora 41.
# Deliberately WITHOUT boost/tbb/hdf5/sundials dev packages: the bundle must be
# self-contained. Driven by containers/release/test-tarball.sh.
FROM fedora:41
LABEL type=diffrg-deps-release-test

RUN dnf -y install \
        git cmake gcc-c++ gcc-gfortran \
        openblas-devel gsl-devel zlib-devel \
        python3 patch curl zstd file pkgconf-pkg-config \
    && dnf clean all

# MPI-variant validation (test-tarball.sh -m): the host Open MPI consumers build
# with. The EL/Fedora package keeps it off the default paths (normally `module
# load mpi/openmpi-x86_64`); export them for the test driver's login shell.
ARG mpi=none
RUN if [ "${mpi}" = openmpi ]; then \
      dnf -y install openmpi-devel && dnf clean all \
      && echo 'export PATH=/usr/lib64/openmpi/bin:$PATH LD_LIBRARY_PATH=/usr/lib64/openmpi/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}' \
           > /etc/profile.d/zz-openmpi.sh; \
    fi
