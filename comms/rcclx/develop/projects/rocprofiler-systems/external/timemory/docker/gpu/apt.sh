#!/bin/bash -e

# function for running command verbosely
run-verbose()
{
    echo -e "\n\t##### Running : '$@'... #####\n"
    eval $@
}

# enable manpages to be installed
# sed -i 's/path-exclude/# path-exclude/g' /etc/dpkg/dpkg.cfg.d/excludes
DISTRIB_CODENAME=$(cat /etc/lsb-release | grep DISTRIB_CODENAME | awk -F '=' '{print $NF}')

#-----------------------------------------------------------------------------#
#
#   apt configuration
#
#-----------------------------------------------------------------------------#

run-verbose apt-get update
run-verbose apt-get install -y software-properties-common wget curl gpg
# test
run-verbose add-apt-repository -u -y ppa:ubuntu-toolchain-r/test
# cmake
wget -O kitware-archive.sh https://apt.kitware.com/kitware-archive.sh
bash ./kitware-archive.sh --release ${DISTRIB_CODENAME}
rm -f ./kitware-archive.sh
# llvm
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
cat << EOF > /etc/apt/sources.list.d/llvm-toolchain.list
# 9
deb http://apt.llvm.org/${DISTRIB_CODENAME}/ llvm-toolchain-${DISTRIB_CODENAME}-9 main
deb-src http://apt.llvm.org/${DISTRIB_CODENAME}/ llvm-toolchain-${DISTRIB_CODENAME}-9 main
# 10
deb http://apt.llvm.org/${DISTRIB_CODENAME}/ llvm-toolchain-${DISTRIB_CODENAME}-10 main
deb-src http://apt.llvm.org/${DISTRIB_CODENAME}/ llvm-toolchain-${DISTRIB_CODENAME}-10 main
# 11
deb http://apt.llvm.org/${DISTRIB_CODENAME}/ llvm-toolchain-${DISTRIB_CODENAME}-11 main
deb-src http://apt.llvm.org/${DISTRIB_CODENAME}/ llvm-toolchain-${DISTRIB_CODENAME}-11 main
# 12
deb http://apt.llvm.org/${DISTRIB_CODENAME}/ llvm-toolchain-${DISTRIB_CODENAME}-12 main
deb-src http://apt.llvm.org/${DISTRIB_CODENAME}/ llvm-toolchain-${DISTRIB_CODENAME}-12 main
# 13
deb http://apt.llvm.org/${DISTRIB_CODENAME}/ llvm-toolchain-${DISTRIB_CODENAME}-13 main
deb-src http://apt.llvm.org/${DISTRIB_CODENAME}/ llvm-toolchain-${DISTRIB_CODENAME}-13 main
# dev
deb http://apt.llvm.org/${DISTRIB_CODENAME}/ llvm-toolchain-${DISTRIB_CODENAME} main
EOF
# upgrade
run-verbose apt-get update
run-verbose apt-get dist-upgrade -y

#-----------------------------------------------------------------------------#
#
#   Base packages
#
#-----------------------------------------------------------------------------#

run-verbose apt-get install -y build-essential git-core ssed bash-completion gdb cmake ninja-build rdma-core

#-----------------------------------------------------------------------------#
#
#   Compiler specific installation
#
#-----------------------------------------------------------------------------#

# install compilers
run-verbose apt-get -y install {gcc,g++,gfortran}-{7,8,9,${GCC_VERSION}} gcc-{7,8,9,${GCC_VERSION}}-multilib
run-verbose apt-get -y install clang-{,tidy-,tools-,format-}{7,8,9,10,11,12,13,${CLANG_VERSION}} libc++-dev libc++abi-dev

DISPLAY_PACKAGES="xserver-xorg freeglut3-dev libx11-dev libx11-xcb-dev libxpm-dev libxft-dev libxmu-dev libxv-dev libxrandr-dev \
    libglew-dev libftgl-dev libxkbcommon-x11-dev libxrender-dev libxxf86vm-dev libxinerama-dev qt5-default \
    emacs-nox vim-nox firefox"
CUDA_VER=$(dpkg --get-selections | grep cuda-cudart- | awk '{print $1}' | tail -n 1 | sed 's/cuda-cudart-//g' | sed 's/dev-//g')

#-----------------------------------------------------------------------------#
#
#   CUDA nsight tools
#
#-----------------------------------------------------------------------------#

if [ -n "${CUDA_VER}" ]; then
    run-verbose apt-get install -y cuda-nsight-{compute,systems}-${CUDA_VER}
fi

#-----------------------------------------------------------------------------#
#
#   Install supplemental packages
#
#-----------------------------------------------------------------------------#

if [ "${ENABLE_DISPLAY}" -gt 0 ]; then
    run-verbose apt-get install -y ${DISPLAY_PACKAGES}
fi

#-----------------------------------------------------------------------------#
#   UPDATE ALTERNATIVES -- GCC
#-----------------------------------------------------------------------------#
priority=10
for i in 5 6 7 8 9 10 11 12 ${GCC_VERSION}
do
    if [ -n "$(which gcc-${i})" ]; then
        run-verbose update-alternatives --install $(which gcc) gcc $(which gcc-${i}) ${priority} \
            --slave $(which g++) g++ $(which g++-${i})
        priority=$(( ${priority}+10 ))
    fi
done

#-----------------------------------------------------------------------------#
#   UPDATE ALTERNATIVES -- CLANG
#-----------------------------------------------------------------------------#
priority=10
for i in 5.0 6.0 7.0 7 8 9 10 11 12 13 14 ${CLANG_VERSION}
do
    if [ -n "$(which clang-${i})" ]; then
        run-verbose update-alternatives --install /usr/bin/clang clang $(which clang-${i}) ${priority} \
            --slave /usr/bin/clang++ clang++ $(which clang++-${i})
        priority=$(( ${priority}+10 ))
    fi
done

#-----------------------------------------------------------------------------#
#   UPDATE ALTERNATIVES -- CLANG TOOLS
#-----------------------------------------------------------------------------#
priority=10
for i in 7.0 7 8 9 10 11 12 13 14 ${CLANG_VERSION} 9
do
    for j in tools tidy format
    do
        if [ -n "$(which clang-${j}-${i})" ]; then
            run-verbose update-alternatives --install /usr/bin/clang-${j} clang-${j} $(which clang-${j}-${i}) ${priority}
        fi
    done
    priority=$(( ${priority}+10 ))
done

#-----------------------------------------------------------------------------#
#   UPDATE ALTERNATIVES -- cc/c++
#-----------------------------------------------------------------------------#
priority=10
if [ -n "$(which clang)" ]; then
    run-verbose update-alternatives --install $(which cc)  cc  $(which clang)   ${priority}
    run-verbose update-alternatives --install $(which c++) c++ $(which clang++) ${priority}
    priority=$(( ${priority}+10 ))
fi

if [ -n "$(which gcc)" ]; then
    run-verbose update-alternatives --install $(which cc)  cc  $(which gcc)     ${priority}
    run-verbose update-alternatives --install $(which c++) c++ $(which g++)     ${priority}
    run-verbose priority=$(( ${priority}+10 ))
fi

if [ -d /opt/intel ]; then
    export PATH=/opt/intel/bin:${PATH}
    if [ -n "$(which icc)" ]; then
        run-verbose update-alternatives --install $(which cc)  cc  $(which icc)     ${priority}
        run-verbose update-alternatives --install $(which c++) c++ $(which icpc)     ${priority}
        run-verbose priority=$(( ${priority}+10 ))
    fi
fi

#-----------------------------------------------------------------------------#
#   UPDATE ALTERNATIVES -- CUDA compilers
#-----------------------------------------------------------------------------#
priority=10
for i in clang++-{7.0,7,8,9,10,11,12,13,14,${CLANG_VERSION}} nvcc
do
    if [ -n "$(which ${i})" ]; then
        run-verbose update-alternatives --install /usr/bin/cu cu $(which ${i}) ${priority}
    fi
done

#-----------------------------------------------------------------------------#
#   CLEANUP
#-----------------------------------------------------------------------------#
run-verbose apt-get -y autoclean
run-verbose rm -rf /var/lib/apt/lists/*

#-----------------------------------------------------------------------------#
#   CONDA
#-----------------------------------------------------------------------------#
# wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
# bash miniconda.sh -b -p /opt/conda
# export PATH="/opt/conda/bin:${PATH}"
# conda config --set always_yes yes --set changeps1 yes
# conda update -c defaults -n base conda
# conda install -n base -c defaults -c conda-forge python=3.6 pyctest cmake scikit-build numpy matplotlib pillow ipykernel jupyter
# source activate
# python -m ipykernel install --name base --display-name base
# conda clean -a -y
# conda config --set always_yes no
