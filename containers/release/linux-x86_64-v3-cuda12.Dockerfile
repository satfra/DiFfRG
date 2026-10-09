# Relocatable DiFfRG dependency bundle: linux x86_64, -march=x86-64-v3,
# CUDA 12 (sm_75/Turing floor). --build-arg mpi=openmpi builds the MPI variant
# linux-x86_64-v3-cuda12-openmpi, exactly as in linux-x86_64-v3-cpu.Dockerfile.
#
# Mirrors linux-x86_64-v3-cpu.Dockerfile with a CUDA-enabled Kokkos. Kokkos
# takes one CUDA arch per build, so the bundle is compiled for the oldest
# architecture DiFfRG supports and runs on everything from Turing to Blackwell
# via its embedded PTX. Only the bundle's own kernels are affected by that:
# libDiFfRG and every application retarget themselves to the local GPU through
# DiFfRG_CUDA_ARCH, so no user code is ever JIT-compiled.
#
# Consumers need the CUDA toolkit anyway (their applications compile device
# code), so the bundle does not ship CUDA: its libraries resolve the host's
# libcudart.so.12, and the toolkit minor should be >= the build's (12.8).
#
# Build from the repository root as context:
#   docker buildx build --build-arg bundle_version=1.0.0 \
#       -f containers/release/linux-x86_64-v3-cuda12.Dockerfile .

# --------------------------------------------------------------------------- #
# Stage 1: superbuild of the dependency targets + relocation post-processing.
# --------------------------------------------------------------------------- #
FROM nvidia/cuda:12.8.1-devel-rockylinux9 AS builder

# none | openmpi. Declared before the package layer, which depends on it.
ARG mpi=none

RUN case "${mpi}" in none | openmpi) ;; *) echo "mpi must be none or openmpi, got '${mpi}'" >&2; exit 1 ;; esac \
    && dnf -y install epel-release \
    && dnf -y --enablerepo=devel install \
        gcc-toolset-14 gcc-toolset-14-gcc-gfortran \
        cmake git patch python3 which \
        openblas-devel gsl-devel zlib-devel \
        patchelf zstd xz file binutils \
        $([ "${mpi}" = openmpi ] && echo openmpi-devel) \
    && dnf clean all

# EL9 keeps Open MPI off the default paths (normally `module load mpi/openmpi-x86_64`).
# Its wrappers call plain `gcc`/`gfortran`, which then resolve to the toolset.
#
# EL9's Open MPI 4.1 has the MPI-2 C++ bindings compiled in, so its mpicxx links
# libmpi_cxx -- which Open MPI 5 no longer has -- and mpi.h pulls the bindings
# in unless OMPI_SKIP_MPICXX is defined. Every FindMPI in the superbuild and its
# sub-builds copies the wrapper's flags verbatim, so fix the wrappers: drop the
# library, define the macro. Nothing here uses those bindings, and
# postprocess-bundle.sh rejects any bundle that still links libmpi_cxx.
RUN echo "source /opt/rh/gcc-toolset-14/enable" > /.bashenv \
    && if [ "${mpi}" = openmpi ]; then \
         echo 'export PATH=/usr/lib64/openmpi/bin:$PATH LD_LIBRARY_PATH=/usr/lib64/openmpi/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}' >> /.bashenv \
         && cd /usr/lib64/openmpi/share/openmpi \
         && sed -i -E '/^libs=/s/ ?-lmpi_cxx//' *-wrapper-data.txt \
         && sed -i -E '/^preprocessor_flags=/s/$/ -DOMPI_SKIP_MPICXX/' \
              mpicxx-wrapper-data.txt mpic++-wrapper-data.txt mpiCC-wrapper-data.txt \
         && ! /usr/lib64/openmpi/bin/mpicxx --showme:link | grep mpi_cxx \
         && /usr/lib64/openmpi/bin/mpicxx --showme:compile | grep -q OMPI_SKIP_MPICXX; \
       fi
ENV BASH_ENV=/.bashenv
SHELL ["/bin/bash", "-c"]

WORKDIR /src
COPY . /src

ARG threads=6
ARG cuda_arch=TURING75

# GPU=ON with a pinned Kokkos arch: no GPU is present at build time, so the
# arch cannot be auto-detected and MUST be given explicitly.
#
# The floor is deliberately the oldest architecture DiFfRG supports. Kokkos
# aborts at startup when its compiled architecture is *above* the device, so
# this is what makes one bundle usable from Turing to Blackwell; the cost is
# only that its own (small) kernels are JIT-compiled on newer GPUs. Nothing the
# user compiles is affected -- DiFfRG_CUDA_ARCH retargets the library and every
# application to the GPU actually in use.
RUN cmake -S /src -B /build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/diffrg \
        -DGPU=ON "-DKokkos_ARCH_LIST=${cuda_arch}" \
        -DMPI=$([ "${mpi}" = openmpi ] && echo ON || echo OFF) -DDiFfRG_DOCUMENTATION=OFF \
        -DMARCH=x86-64-v3 \
        -DBUILD_BOOST=ON -DBUILD_TBB=ON -DBUILD_HDF5=ON -DBUILD_SUNDIALS=ON \
        -DDEAL_II_GSL=OFF \
        -DUSE_CCACHE=OFF \
        -DBUILD_JOBS=$((2 * threads)) \
        -DDEALII_MAX_JOBS=${threads} \
        -DPETSC_MAX_JOBS=${threads} \
    && cmake --build /build \
        --target general_dep deal.II_dep kokkos_dep autodiff_dep \
        -j ${threads} \
    && chmod -R a+rX /opt/diffrg

ARG bundle_version=0.0.0
ARG git_sha=unknown
ARG deps_inputs_hash=unknown
RUN GIT_SHA="${git_sha}" DEPS_INPUTS_HASH="${deps_inputs_hash}" bash /src/containers/release/postprocess-bundle.sh \
        /opt/diffrg/bundled /src "${bundle_version}" \
        "linux-x86_64-v3-cuda12$([ "${mpi}" = none ] || echo "-${mpi}")" x86-64-v3 2.34

# --------------------------------------------------------------------------- #
# Stage 2: runtime image -- proves self-containment, then emits the tarball.
# --------------------------------------------------------------------------- #
# The runtime CUDA image provides libcudart but not the driver's libcuda.so.1
# (that comes from the host driver at run time); link the toolkit's stub so
# the ldd audit can resolve it.
FROM nvidia/cuda:12.8.1-runtime-rockylinux9
LABEL type=diffrg-deps-release
LABEL org.opencontainers.image.source=https://github.com/satfra/DiFfRG_current

ARG bundle_version=0.0.0
ARG mpi=none

RUN dnf -y install openblas-serial zlib tar zstd file \
        $([ "${mpi}" = openmpi ] && echo openmpi) \
    && dnf clean all

# The runtime image ships libcudart but not the driver's libcuda.so.1 (host
# driver territory), and not even its stub -- take the stub from the builder's
# devel toolkit so the ldd audit can resolve it.
COPY --from=builder /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib64/libcuda.so.1
RUN ldconfig

COPY --from=builder /opt/diffrg/bundled /opt/diffrg/bundled
COPY containers/release/check-linkage.sh /usr/local/bin/check-linkage.sh

RUN if [ "${mpi}" = openmpi ]; then export CHECK_LINKAGE_MPI=1 LD_LIBRARY_PATH=/usr/lib64/openmpi/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; fi \
    && CHECK_LINKAGE_CUDA=1 bash /usr/local/bin/check-linkage.sh /opt/diffrg/bundled

RUN name="diffrg-deps-${bundle_version}-linux-x86_64-v3-cuda12$([ "${mpi}" = none ] || echo "-${mpi}")" \
    && mkdir -p "/dist/${name}" \
    && cp -a /opt/diffrg/bundled "/dist/${name}/bundled" \
    && cp /opt/diffrg/bundled/BUNDLE_MANIFEST.json "/dist/${name}/" \
    && tar -C /dist --sort=name --owner=0 --group=0 --numeric-owner \
           -cf - "${name}" | zstd -19 -T0 -o "/dist/${name}.tar.zst" \
    && rm -rf "/dist/${name}" \
    && cd /dist && sha256sum "${name}.tar.zst" > "${name}.tar.zst.sha256"
