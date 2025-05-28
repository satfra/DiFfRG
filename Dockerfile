#ARG base="rockylinux:9"
ARG base="ubuntu"

FROM $base

ARG threads="8"
ARG cuda=""

# install dependencies
# RUN dnf --enablerepo=devel install -y gcc-toolset-12 cmake git openblas-devel doxygen doxygen-latex python3 python3-pip gsl-devel
# RUN apt install -y gcc
RUN apt-get update
RUN apt-get install -y gcc g++ git cmake libopenblas-dev

# set a directory for the app
WORKDIR /DiFfRG/

# copy code to the image
COPY ./build.sh build.sh
COPY ./update_DiFfRG.sh update_DiFfRG.sh
COPY ./build_scripts build_scripts
COPY ./.gitmodules .gitmodules
COPY ./.git ./.git
COPY ./config config
COPY ./external ./external
COPY ./DiFfRG ./DiFfRG

#SHELL [ "/usr/bin/scl", "enable", "gcc-toolset-12"]
# RUN gcc --version
RUN ./build.sh -j $threads -f $cuda -i DiFfRG_install

# install dependencies
#RUN git reset --hard
#SHELL [ "/usr/bin/scl", "enable", "gcc-toolset-12"]
#RUN bash -i build.sh -j $threads -d -f $cuda -i DiFfRG_install
#RUN ln -s /DiFfRG/DiFfRG_install /opt/DiFfRG
#
## run the command
#CMD ["scl", "enable", "gcc-toolset-12", "bash"]
