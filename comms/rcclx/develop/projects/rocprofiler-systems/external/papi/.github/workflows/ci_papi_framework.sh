#!/bin/bash -e

COMPONENTS=$1
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

# set necessary environment variables for lmsensors
case "$COMPONENTS" in
  *"lmsensors"*)
    wget https://github.com/groeck/lm-sensors/archive/V3-4-0.tar.gz
    tar -zxf V3-4-0.tar.gz 
    cd lm-sensors-3-4-0
    make install PREFIX=../lm ETCDIR=../lm/etc
    cd ..
    export PAPI_LMSENSORS_ROOT=lm
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PAPI_LMSENSORS_ROOT/lib  
    ;;  
esac

# set necessary environment variables for rocm and rocm_smi
case "$COMPONENTS" in
  *"rocm rocm_smi"*)
    export PAPI_ROCM_ROOT=/apps/rocm/rocm-5.5.3
    export PAPI_ROCMSMI_ROOT=$PAPI_ROCM_ROOT/rocm_smi
    ;;
esac

# set necessary environment variables for cuda and nvml
case "$COMPONENTS" in
  *"cuda nvml"*)
    module load cuda
    export PAPI_CUDA_ROOT=$ICL_CUDA_ROOT
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PAPI_CUDA_ROOT/extras/CUPTI/lib64
    ;;
esac

# test linking with or without --with-shlib-tools 
if [ "$SHLIB" = "without" ]; then
    ./configure --with-debug=$DEBUG --enable-warnings --with-components="$COMPONENTS"
    ./configure --with-debug=$DEBUG --enable-warnings --with-components="$COMPONENTS" --with-shlib-tools
else
    ./configure --with-debug=$DEBUG --enable-warnings --with-components="$COMPONENTS" --with-shlib-tools
fi

make -j4

# run PAPI utilities
utils/papi_component_avail

# active component check
CURRENT_ACTIVE_COMPONENTS=$(utils/papi_component_avail | grep -A1000 'Active components' | grep "Name:   " | awk '{printf "%s%s", sep, $2; sep=" "} END{print ""}')
if [ "$COMPONENTS" = "cuda nvml rocm rocm_smi powercap powercap_ppc rapl sensors_ppc infiniband net appio io lustre stealtime coretemp lmsensors mx sde" ]; then 
    [ "$CURRENT_ACTIVE_COMPONENTS" = "perf_event perf_event_uncore cuda nvml powercap net appio io stealtime coretemp lmsensors sde sysdetect" ]
elif [ "$COMPONENTS" = "rocm rocm_smi" ]; then
    [ "$CURRENT_ACTIVE_COMPONENTS" = "perf_event perf_event_uncore rocm rocm_smi sysdetect" ]
elif [ "$COMPONENTS" = "infiniband" ]; then
    [ "$CURRENT_ACTIVE_COMPONENTS" = "perf_event perf_event_uncore infiniband sysdetect" ]
else
    # if the component from the .yml is not accounted for in the above
    # elif's
    exit 1
fi

# without '--with-shlib-tools' in ./configure
if [ "$SHLIB" = "without" ]; then
   echo "Running full test suite for active components"
   ./run_tests.sh TESTS_QUIET --disable-cuda-events=yes
# with '--with-shlib-tools' in ./configure
else
   echo "Running single component test for active components"
   ./run_tests_shlib.sh TESTS_QUIET
fi
