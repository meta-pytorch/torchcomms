#!/bin/bash

# Set path to components/sde/tests/lib as
# this is needed for nearly all the tests
# to run successfully.
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH

# Add -verbose to show output, i.e. sh run_sde_tests.sh -verbose.
if [ "$1" = "-verbose" ]; then
    run_verbose="$1"
fi

make_sde_test_targets=(
    "Minimal_Test"
    "Minimal_Test++"
    "Simple_Test"
    "Simple2_Test"
    "Simple2_NoPAPI_Test"
    "Simple2_Test++"
    "Recorder_Test"
    "Recorder_Test++"
    "Created_Counter_Test"
    "Overflow_Test"
    "Overflow_Static_Test"
    "Created_Counter_Test++"
    "Counting_Set_MemLeak_Test++"
    "Counting_Set_Simple_Test++"
    "Counting_Set_Simple_Test"
    "Counting_Set_MemLeak_Test"
    "sde_test_f08"
)

for sde_test in ${make_sde_test_targets[@]}; do
    echo "make $sde_test:"
    make $sde_test

    printf "\n"

    echo "Running $sde_test:"
    ./$sde_test "$run_verbose"

    echo "-------------------------------------"
done
