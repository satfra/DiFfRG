# Environment for validating a pre-built DiFfRG dependency bundle on Rocky 9 --
# the oldest supported host (glibc 2.34, the bundle's floor). Deliberately
# WITHOUT boost/tbb/hdf5/sundials dev packages: the bundle must be
# self-contained. Driven by containers/release/test-tarball.sh.
FROM rockylinux:9
LABEL type=diffrg-deps-release-test

# Rocky 9's default GCC (11) lacks full C++20; gcc-toolset-14 matches the
# compiler generation the bundle was built with. openblas-devel lives in the
# (disabled-by-default) devel repo.
RUN dnf -y install epel-release \
    && dnf -y --enablerepo=devel install \
        gcc-toolset-14 gcc-toolset-14-gcc-gfortran \
        git cmake \
        openblas-devel gsl-devel zlib-devel \
        python3 patch which zstd file pkgconf-pkg-config \
    && dnf clean all

# Activate the toolset for non-interactive (BASH_ENV) and login (profile.d)
# shells alike -- the test driver enters via `bash -lc`.
RUN echo "source /opt/rh/gcc-toolset-14/enable" > /.bashenv \
    && cp /.bashenv /etc/profile.d/gcc-toolset-14.sh
ENV BASH_ENV=/.bashenv

# MPI-variant validation (test-tarball.sh -m): the host Open MPI consumers build
# with. The EL/Fedora package keeps it off the default paths (normally `module
# load mpi/openmpi-x86_64`); export them for login and non-login shells alike.
ARG mpi=none
RUN if [ "${mpi}" = openmpi ]; then \
      dnf -y install openmpi-devel && dnf clean all \
      && echo 'export PATH=/usr/lib64/openmpi/bin:$PATH LD_LIBRARY_PATH=/usr/lib64/openmpi/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}' \
           | tee -a /.bashenv > /etc/profile.d/zz-openmpi.sh; \
    fi
