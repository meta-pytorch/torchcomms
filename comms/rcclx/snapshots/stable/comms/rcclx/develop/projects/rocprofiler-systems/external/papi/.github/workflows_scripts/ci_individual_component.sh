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

# --- Set Necessary Environment Variables ---
## Set the lmsensors component environment variables
if [ "$COMPONENT" = "lmsensors" ]; then
    wget https://github.com/groeck/lm-sensors/archive/V3-4-0.tar.gz
    tar -zxf V3-4-0.tar.gz
    cd lm-sensors-3-4-0
    make install PREFIX=../lm ETCDIR=../lm/etc
    cd ..
    export PAPI_LMSENSORS_ROOT=lm
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PAPI_LMSENSORS_ROOT/lib
fi

## Set the rocm component environment variable
if [ "$COMPONENT" = "rocm" ]; then
    module load rocm/6.2
    export PAPI_ROCM_ROOT=$ROCM_PATH
fi

## Set the rocm_smi component environment variable
if [ "$COMPONENT" = "rocm_smi" ]; then
    module load rocm/6.3.2
    export PAPI_ROCMSMI_ROOT=$ROCM_PATH
fi

## Set the rocp_sdk component environment variable
if [ "$COMPONENT" = "rocp_sdk" ]; then
    module load rocm/7.0.1
    export PAPI_ROCP_SDK_ROOT=$ROCM_PATH
fi

## Set the amd_smi component environment variable
if [ "$COMPONENT" = "amd_smi" ]; then
    module load rocm/7.0.1
    export PAPI_AMDSMI_ROOT=$ROCM_PATH
fi

## Set the cuda component or the nvml component environment variable
if [ "$COMPONENT" = "cuda" ] || [ "$COMPONENT" = "nvml" ]; then
    module unload glibc
    export MODULEPATH=$MODULEPATH:/apps/spacks/cuda/share/spack/modules/linux-rocky9-skylake_avx512/
    module load  cuda/12.8.0
    export PAPI_CUDA_ROOT=$ICL_CUDA_ROOT
    export LD_LIBRARY_PATH=$PAPI_CUDA_ROOT/lib64:$PAPI_CUDA_ROOT/extras/CUPTI/lib64:$LD_LIBRARY_PATH
fi

## Set the intel_gpu component environment variables
if [ "$COMPONENT" = "intel_gpu" ]; then
    export PAPI_INTEL_GPU_ROOT=/usr
    export ZET_ENABLE_METRICS=1
fi

# --- Configure and Build PAPI ---
## Configure without --with-shlib-tools
if [ "$SHLIB" = "without" ]; then
    ./configure --with-debug=$DEBUG --enable-warnings --with-components="$COMPONENT" 
## Configure with --with-shlib-tools
else
    ./configure --with-debug=$DEBUG --enable-warnings --with-components="$COMPONENT" --with-shlib-tools
fi
make -j4

# --- Verify Components we Expect to Be Active are Active ---
## For the PAPI build get the active components
utils/papi_component_avail
current_active_components=$(utils/papi_component_avail | grep -A1000 'Active components' | grep "Name:   " | awk '{printf "%s%s", sep, $2; sep=" "} END{print ""}')

## Defining the components we expect to be active for the PAPI build
declare -a expected_active_components
expected_active_components=("perf_event" "perf_event_uncore" "sysdetect" "$COMPONENT")

## Verify the expected active components are active for the PAPI build
for cmp in "${expected_active_components[@]}"; do
    grep -q $cmp <<< $current_active_components || \
    { echo "The component $cmp is not active and should be active!"; exit 1; }
done

# --- Run Tests: Ctests, Ftests, Component Tests, etc. ---
## Without '--with-shlib-tools' in ./configure
if [ "$SHLIB" = "without" ]; then
    echo "Running full test suite for active components"
    ./run_tests.sh TESTS_QUIET --disable-cuda-events=yes
## With '--with-shlib-tools' in ./configure
else
    echo "Running single component test for active components"
    ./run_tests_shlib.sh TESTS_QUIET
fi
