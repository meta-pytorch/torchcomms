#!/bin/bash -e

# function for running command verbosely
run-verbose()
{
    echo -e "\n\t##### Running : '$@'... #####\n"
    eval "${@}"
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
run-verbose apt-get install -y software-properties-common wget curl
# test
# run-verbose add-apt-repository -u -y ppa:ubuntu-toolchain-r/test
# cmake
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add -
apt-add-repository -y "deb https://apt.kitware.com/ubuntu/ ${DISTRIB_CODENAME} main"
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
# run-verbose apt-get dist-upgrade -y

#-----------------------------------------------------------------------------#
#
#   Base packages
#
#-----------------------------------------------------------------------------#

run-verbose apt-get install -y build-essential git-core ssed cmake ninja-build

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
#   UPDATE ALTERNATIVES -- GCC
#-----------------------------------------------------------------------------#
priority=10
for i in 5 6 7 8 9 10 11 12 13 14 ${GCC_VERSION}
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
for i in 5.0 6.0 7.0 7 8 9 10 11 12 13 14 15 ${CLANG_VERSION}
do
    if [ -n "$(which clang-${i})" ]; then
        run-verbose update-alternatives --install /usr/bin/clang clang $(which clang-${i}) ${priority}
        run-verbose update-alternatives --install /usr/bin/clang++ clang++ $(which clang++-${i}) ${priority}
        priority=$(( ${priority}+10 ))
    fi
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
    priority=$(( ${priority}+10 ))
fi

if [ -d /opt/intel ]; then
    export PATH=/opt/intel/bin:${PATH}
    if [ -n "$(which icc)" ]; then
        run-verbose update-alternatives --install $(which cc)  cc  $(which icc)     ${priority}
        run-verbose update-alternatives --install $(which c++) c++ $(which icpc)    ${priority}
        priority=$(( ${priority}+10 ))
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
