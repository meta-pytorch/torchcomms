#!/bin/bash -e

COMPONENTS=$1
DEBUG=$2
SHLIB=$3
HARDWARE=$4
COMPILER=$5

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

## Set the rocp_sdk and amd_smi component environment variables
case "$COMPONENTS" in
  *"rocp_sdk"* | *"amd_smi"*)
    module load rocm/7.0.1
    export PAPI_ROCP_SDK_ROOT=$ROCM_PATH
    export PAPI_AMDSMI_ROOT=$ROCM_PATH
    ;;
esac

## Set the cuda component and the nvml component environment variables
case "$COMPONENTS" in
  *"cuda"* | *"nvml"*)
    module unload glibc
    export MODULEPATH=$MODULEPATH:/apps/spacks/cuda/share/spack/modules/linux-rocky9-skylake_avx512/
    module load cuda/12.8.0
    export PAPI_CUDA_ROOT=$ICL_CUDA_ROOT
    export LD_LIBRARY_PATH=$PAPI_CUDA_ROOT/lib64:$PAPI_CUDA_ROOT/extras/CUPTI/lib64:$LD_LIBRARY_PATH
    ;;
esac

# --- Configure and Build PAPI ---
## Configure without --with-shlib-tools
if [ "$SHLIB" = "without" ]; then
    ./configure --with-debug=$DEBUG --enable-warnings --with-components="$COMPONENTS"
## Configure with --with-shlib-tools
else
    ./configure --with-debug=$DEBUG --enable-warnings --with-components="$COMPONENTS" --with-shlib-tools
fi
make -j4

# --- Verify Components we Expect to Be Active are Active ---
## For the PAPI build get the active components
utils/papi_component_avail
current_active_components=$(utils/papi_component_avail | grep -A1000 'Active components' | grep "Name:   " | awk '{printf "%s%s", sep, $2; sep=" "} END{print ""}')

## Defining the components we expect to be active based on hardware for a PAPI build
declare -a per_hardware_expected_active_components
if [ "$HARDWARE" = "gpu_nvidia_w_cpu_intel" ]; then
    per_hardware_expected_active_components=("perf_event" "perf_event_uncore" "cuda" "nvml" "powercap" "net" "appio" "io" "stealtime" "coretemp" "lmsensors" "sde" "sysdetect")
elif [ "$HARDWARE" = "gpu_amd" ]; then
    per_hardware_expected_active_components=("perf_event" "perf_event_uncore" "rocp_sdk" "amd_smi" "sysdetect")
elif [ "$HARDWARE" = "infiniband" ]; then
    per_hardware_expected_active_components=("perf_event" "perf_event_uncore" "infiniband" "sysdetect")
fi

## Verify expected active components are active
for cmp in "${per_hardware_expected_active_components[@]}"; do
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
