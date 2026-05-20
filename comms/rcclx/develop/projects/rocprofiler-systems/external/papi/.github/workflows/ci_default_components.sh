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

# test linking with or without --with-shlib-tools 
if [ "$SHLIB" = "without" ]; then
    ./configure --with-debug=$DEBUG --enable-warnings 
else
    ./configure --with-debug=$DEBUG --enable-warnings --with-shlib-tools
fi

make -j4

# run PAPI utilities
utils/papi_component_avail

# active component check
EXPECTED_ACTIVE_COMPONENTS="perf_event perf_event_uncore sysdetect" 
CURRENT_ACTIVE_COMPONENTS=$(utils/papi_component_avail | grep -A1000 'Active components' | grep "Name:   " | awk '{printf "%s%s", sep, $2; sep=" "} END{print ""}')
[ "$EXPECTED_ACTIVE_COMPONENTS" = "$CURRENT_ACTIVE_COMPONENTS" ]

# without '--with-shlib-tools' in ./configure
if [ "$SHLIB" = "without" ]; then
   echo "Running full test suite for active components"
   ./run_tests.sh TESTS_QUIET
# with '--with-shlib-tools' in ./configure
else
   echo "Running single component test for active components"
   ./run_tests_shlib.sh TESTS_QUIET
fi
