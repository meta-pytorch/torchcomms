# Build stage with Spack pre-installed and ready to be used
FROM spack/ubuntu-focal:latest as builder
# FROM ecpe4s/ubuntu18.04-spack-x86_64:0.15.3 as builder

ARG VERSION=develop
ARG PYTHON_VERSION=3.7
ARG COMPILER_SPEC=gcc

# What we want to install and how we want to install it
# is specified in a manifest file (spack.yaml)
RUN mkdir /opt/environment \
    &&  (echo "spack:" \
    &&   echo "  specs:" \
    &&   echo "  - papi" \
    &&   echo "  - mpich" \
    &&   echo "  - gotcha" \
    &&   echo "  - dyninst" \
    &&   echo "  - caliper~papi" \
    &&   echo "  - gperftools" \
    &&   echo "  - python@${PYTHON_VERSION}" \
    &&   echo "  - py-numpy" \
    &&   echo "  - py-cython" \
    &&   echo "  - py-pillow" \
    &&   echo "  - py-matplotlib" \
    &&   echo "  - py-mpi4py" \
    &&   echo "  - py-pip" \
    &&   echo "  - timemory@${VERSION}%${COMPILER_SPEC} +caliper+dyninst+gotcha+kokkos_tools+kokkos_build_config+mpi+mpip_library+ompt+ompt_library+papi+python+tools" \
    &&   echo "  concretizer:" \
    &&   echo "    unify: when_possible" \
    &&   echo "  config:" \
    &&   echo "    install_tree: /opt/software" \
    &&   echo "  view: /opt/view") > /opt/environment/spack.yaml

# Install the software, remove unecessary deps
RUN cd /tmp && \
    apt-get update && \
    apt-get dist-upgrade -y && \
    apt-get install -y wget gpg software-properties-common && \
    wget -O kitware-archive.sh https://apt.kitware.com/kitware-archive.sh && \
    chmod 755 ./kitware-archive.sh && \
    DISTRIB_CODENAME=$(cat /etc/lsb-release | grep DISTRIB_CODENAME | awk -F '=' '{print $NF}') && \
    ./kitware-archive.sh --release ${DISTRIB_CODENAME} && \
    rm -f ./kitware-archive.sh && \
    apt-get update && \
    apt-get install -y cmake build-essential ninja-build && \
    apt-get -y autoclean

# Install the software, remove unecessary deps
RUN cd /opt/environment && \
    wget https://oaciss.uoregon.edu/e4s/e4s.pub && \
    spack gpg trust e4s.pub && \
    spack mirror add E4S https://cache.e4s.io/e4s && \
    rm e4s.pub && \
    spack compiler find && \
    spack external find && \
    spack --env . install --reuse && \
    spack gc -y

# Modifications to the environment that are necessary to run
RUN cd /opt/environment && \
    spack env activate --sh -d . >> /etc/profile.d/z10_spack_environment.sh

# Bare OS image to run the installed executables
FROM spack/ubuntu-focal:latest

COPY --from=builder /opt/environment /opt/environment
COPY --from=builder /opt/software /opt/software
COPY --from=builder /opt/view /opt/view
COPY --from=builder /etc/profile.d/z10_spack_environment.sh /etc/profile.d/z10_spack_environment.sh

RUN echo 'export PS1="\[$(tput bold)\]\[$(tput setaf 1)\][timemory]\[$(tput setaf 2)\] \u\[$(tput sgr0)\]:\w $ \[$(tput sgr0)\]"' >> ~/.bashrc

ENV HOME /root
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US
ENV LC_ALL C
ENV SHELL /bin/bash
ENV BASH_ENV /etc/bash.bashrc
ENV DEBIAN_FRONTEND noninteractive

COPY ./etc/profile.d/*.sh /etc/profile.d/
COPY ./etc/bash.bashrc /etc/
COPY ./etc/compute-dir-size.py /etc/

LABEL "app"="timemory"
LABEL "mpi"="mpich"

USER root
WORKDIR /home
SHELL [ "/bin/bash", "--login", "-c" ]

# ENTRYPOINT [ "/runtime-entrypoint.sh" ]
ENTRYPOINT ["/bin/bash", "--rcfile", "/etc/profile", "-l"]
