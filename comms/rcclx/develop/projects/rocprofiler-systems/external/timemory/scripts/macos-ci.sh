#!/bin/bash -e

CONDA_PREFIX=${1}
TIMEMORY_SOURCE_DIR=${2}
: ${PYTHON_VERSION:=3.7}
: ${NPROC:=4}
: ${CMAKE_ARGS:="-DTIMEMORY_BUILD_PYTHON_HATCHET=OFF"}
if [ -n "$(which mpicc)" ]; then
    : ${BUILD_ARGS:="--minimal --build-libs shared --python --mpi --cxx-standard=17"}
else
    : ${BUILD_ARGS:="--minimal --build-libs shared --python --cxx-standard=17"}
fi

if [ -z "${CONDA_PREFIX}" ]; then
    echo "Provide existing conda root directory or a scratch directory for installation"
    exit 1
else
    shift
fi

if [ -z "${TIMEMORY_SOURCE_DIR}" ]; then
    echo "Provide existing timemory source directory or scratch directory for cloning"
    exit 1
else
    shift
fi

: ${TIMEMORY_INSTALL_DIR:="$(dirname ${TIMEMORY_SOURCE_DIR})/timemory-install"}

export CC=$(which clang)
export CXX=$(which clang++)

echo -e "Using:"
echo -e "    CONDA_PREFIX         = ${CONDA_PREFIX}"
echo -e "    PYTHON_VERSION       = ${PYTHON_VERSION}"
echo -e "    TIMEMORY_SOURCE_DIR  = ${TIMEMORY_SOURCE_DIR}"
echo -e "    TIMEMORY_INSTALL_DIR = ${TIMEMORY_INSTALL_DIR}"
echo -e "    NPROC                = ${NPROC}"
echo -e "    BUILD_ARGS           = ${BUILD_ARGS}"
echo -e "    CMAKE_ARGS           = ${CMAKE_ARGS}"
echo -en "\nContinue? [y/n]:"
read user_input

if [ "${user_input}" != "y" ]; then
    echo -e "Exiting"
    exit 0
fi

if [ ! -d ${CONDA_PREFIX}/bin ]; then
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ${TMPDIR}/miniconda.sh
    bash ${TMPDIR}/miniconda.sh -b -p ${CONDA_PREFIX}
fi

export PATH=${CONDA_PREFIX}/bin:${PATH}

conda create -y -c conda-forge -c defaults -n timemory-pyctest python=${PYTHON_VERSION} pyctest scikit-build cmake pip
source activate
conda activate timemory-pyctest

if [ ! -d ${TIMEMORY_SOURCE_DIR} ]; then
    mkdir -p $(dirname ${TIMEMORY_SOURCE_DIR})
    git clone https://github.com/NERSC/timemory.git ${TIMEMORY_SOURCE_DIR}
fi

cd ${TIMEMORY_SOURCE_DIR}
python -m pip install -r requirements.txt

if [ -n "$(which mpicc)" ]; then
    python -m pip install mpi4py
fi

if [ -z "${BUILD_TYPE}" ]; then
    BUILD_TYPE=RelWithDebInfo
fi

BASE_ARGS="-SF --pyctest-model=Continuous --pyctest-site=$(whoami)-${HOSTNAME} -j ${NPROC}"
CTEST_ARGS="-V --output-on-failure ${CTEST_ARGS}"
CMAKE_ARGS="-DCMAKE_INSTALL_PREFIX=${TIMEMORY_INSTALL_DIR} ${CMAKE_ARGS}"
PYCTEST_ARGS="${BASE_ARGS} -cc ${CC} -cxx ${CXX} --pyctest-build-type=${BUILD_TYPE} ${BUILD_ARGS}"

# main command
python ./pyctest-runner.py ${PYCTEST_ARGS} $@ -- ${CTEST_ARGS} -- ${CMAKE_ARGS}

# cd into build directory
cd build-timemory/Darwin

# make install and check that cmake configures from installation and at least one of them builds
make install -j

# for finding installation
export CMAKE_PREFIX_PATH=${TIMEMORY_INSTALL_DIR}:${CMAKE_PREFIX_PATH}

# if python install test file exists, run it
if [ -f "tests/test-python-install-import.cmake" ]; then
    cmake -P tests/test-python-install-import.cmake
    cd
    export PYTHON_PATH=${HOME}/timemory-install/lib/python${TRAVIS_PYTHON_VERSION}/site-packages:${PYTHONPATH}
    python -c "import timemory; print(f'Loaded timemory python module at {timemory.__file__}')"
fi

# build from installed version
cd ${TIMEMORY_SOURCE_DIR}/examples && mkdir build-examples && cd build-examples

# configure and build
cmake -B ${TIMEMORY_SOURCE_DIR}/examples/build-examples -DTIMEMORY_BUILD_C_EXAMPLES=ON ${TIMEMORY_SOURCE_DIR}/examples
cmake --build ${TIMEMORY_SOURCE_DIR}/examples/build-examples --target all --parallel ${NPROC}
