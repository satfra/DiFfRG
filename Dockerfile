ARG base="rockylinux:9"

FROM $base

ARG threads="1"
ARG cuda=""

# set a directory for the app
WORKDIR /DiFfRG/

# copy the git to the container
COPY ./.git ./.git

# install dependencies
RUN dnf --enablerepo=devel install -y gcc-toolset-12 cmake git openblas-devel doxygen doxygen-latex python3 python3-pip gsl-devel
RUN git reset --hard
SHELL [ "/usr/bin/scl", "enable", "gcc-toolset-12"]
RUN bash -i build.sh -j $threads -d -f $cuda -i DiFfRG_install &>build.log
RUN ln -s /DiFfRG/DiFfRG_install /opt/DiFfRG

# run the command
CMD ["scl", "enable", "gcc-toolset-12", "bash"]
