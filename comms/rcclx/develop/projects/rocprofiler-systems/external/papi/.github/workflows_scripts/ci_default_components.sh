#!/bin/bash -e

DEBUG=$1
SHLIB=$2
COMPILER=$3

[ -z "$COMPILER" ] && COMPILER=gcc@11

source /etc/profile
set +x
set -e
trap 'echo "# $BASH_COMMAND"' DEBUG
shopt -s expand_aliases

module load $COMPILER

cd src

# --- Configure and Build PAPI ---
## Configure without --with-shlib-tools
if [ "$SHLIB" = "without" ]; then
    ./configure --with-debug=$DEBUG --enable-warnings
## Configure with --with-shlib-tools 
else
    ./configure --with-debug=$DEBUG --enable-warnings --with-shlib-tools
fi
make -j4

# --- Verify Components we Expect to Be Active are Active ---
## For the PAPI build get the active components
utils/papi_component_avail
current_active_components=$(utils/papi_component_avail | grep -A1000 'Active components' | grep "Name:   " | awk '{printf "%s%s", sep, $2; sep=" "} END{print ""}')

## defining the components we expect to be active for the PAPI build
declare -a expected_active_components
expected_active_components=("perf_event" "perf_event_uncore" "sysdetect")

## Verify the expected active components are active for the PAPI build
for cmp in "${expected_active_components[@]}"; do
    grep -q $cmp <<< $current_active_components || \
    { echo "The component $cmp is not active and should be active!"; exit 1; }
done

# --- Run Tests: Ctests, Ftests, Component Tests, etc. ---
## Without '--with-shlib-tools' in ./configure
if [ "$SHLIB" = "without" ]; then
   echo "Running full test suite for active components"
   ./run_tests.sh TESTS_QUIET
## With '--with-shlib-tools' in ./configure
else
   echo "Running single component test for active components"
   ./run_tests_shlib.sh TESTS_QUIET
fi
