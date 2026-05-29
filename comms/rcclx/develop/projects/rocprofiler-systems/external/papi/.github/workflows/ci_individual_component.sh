#!/bin/bash -e

COMPONENT=$1
DEBUG=$2
SHLIB=$3
COMPILER=$4

[ -z "$COMPILER" ] && COMPILER=gcc@11

source /etc/profile
set +x
set -e
trap 'echo "# $BASH_COMMAND"' DEBUG
shopt -s expand_aliases

module load $COMPILER

cd src

# lmsensors environment variables
if [ "$COMPONENT" = "lmsensors" ]; then
  wget https://github.com/groeck/lm-sensors/archive/V3-4-0.tar.gz
  tar -zxf V3-4-0.tar.gz 
  cd lm-sensors-3-4-0
  make install PREFIX=../lm ETCDIR=../lm/etc
  cd ..
  export PAPI_LMSENSORS_ROOT=lm
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PAPI_LMSENSORS_ROOT/lib
fi

# rocm and rocm_smi environment variables
if [ "$COMPONENT" = "rocm" ] || [ "$COMPONENT" = "rocm_smi" ]; then
  export PAPI_ROCM_ROOT=/apps/rocm/rocm-5.5.3
  export PAPI_ROCMSMI_ROOT=$PAPI_ROCM_ROOT/rocm_smi
fi

# set necessary environemnt variables for cuda and nvml
if [ "$COMPONENT" = "cuda" ] || [ "$COMPONENT" = "nvml" ]; then
  module load cuda
  export PAPI_CUDA_ROOT=$ICL_CUDA_ROOT
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PAPI_CUDA_ROOT/extras/CUPTI/lib64
fi

# test linking with or without --with-shlib-tools 
if [ "$SHLIB" = "without" ]; then
    ./configure --with-debug=$DEBUG --enable-warnings --with-components="$COMPONENT" 
else
    ./configure --with-debug=$DEBUG --enable-warnings --with-components="$COMPONENT" --with-shlib-tools
fi

make -j4

# run PAPI utilities
utils/papi_component_avail

# active component check
EXPECTED_ACTIVE_COMPONENTS=$(echo "perf_event perf_event_uncore sysdetect" | sed "s/perf_event_uncore/& $COMPONENT/") 
CURRENT_ACTIVE_COMPONENTS=$(utils/papi_component_avail | grep -A1000 'Active components' | grep "Name:   " | awk '{printf "%s%s", sep, $2; sep=" "} END{print ""}')
[ "$EXPECTED_ACTIVE_COMPONENTS" = "$CURRENT_ACTIVE_COMPONENTS" ]

# without '--with-shlib-tools' in ./configure
if [ "$SHLIB" = "without" ]; then
   echo "Running full test suite for active components"
   ./run_tests.sh TESTS_QUIET --disable-cuda-events=yes
# with '--with-shlib-tools' in ./configure
else
   echo "Running single component test for active components"
   ./run_tests_shlib.sh TESTS_QUIET
fi
