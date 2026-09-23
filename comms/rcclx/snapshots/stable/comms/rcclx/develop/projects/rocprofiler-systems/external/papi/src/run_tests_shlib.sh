#!/bin/sh

# File:    run_tests_shlib.sh
# Author:  Treece Burgess tburgess@icl.utk.edu
# This script is designed specifically for the PAPI GitHub CI when
# --with-shlib-tools is used during the ./configure stage.
# A single component test for each active component will be ran.

# if component tests are not built, then build them
if [ "x$BUILD" != "x" ]; then
    for comp in `ls components/*/tests` ; do \
        cd components/$$comp/tests ; make; cd ../../.. ;
    done
fi

# determine whether to suppress test output
TESTS_QUIET=""
if [ $# != 0 ]; then
  if [ $1 = "TESTS_QUIET" ]; then
     TESTS_QUIET=$1
  fi
fi

# determine if VALGRIND is set
if [ "x$VALGRIND" != "x" ]; then
  VALGRIND="valgrind --leak-check=full";
fi

# collect the current active components 
ACTIVE_COMPONENTS_PATTERN=$(utils/papi_component_avail | awk '/Active components:/{flag=1; next} flag' | grep "Name:" | sed 's/Name: //' | awk '{print $1}' | paste -sd'|' -)

# collecting inactive component tests to be filtered
INACTIVE_COMPONENTS=$(find components/*/tests -perm -u+x -type f ! \( -name "*.[c|h]" -o -name "*.cu" -o -name "*.so" \) | grep -vE "components/($ACTIVE_COMPONENTS_PATTERN)/")

# set of tests we want to ignore
EXCLUDE_TXT=`grep -v -e '^#\|^$' run_tests_exclude.txt`
EXCLUDE_TESTS="$EXCLUDE_TXT $INACTIVE_COMPONENTS"

# for each active component, collect a single component test
ACTIVE_COMPONENTS_TESTS=""
for cmp in $(echo $ACTIVE_COMPONENTS_PATTERN | sed 's/|/ /g');
do
    # query a test for the active component and make sure it is not an excluded test
    QUERY_CMP_TEST=$(find components/$cmp/tests -perm -u+x -type f ! \( -name "*.[c|h]" -o -name "*.cu" -o -name "*.so" \) | grep -E -m 1 "components/($cmp)/")
    case $EXCLUDE_TESTS in
      *"$QUERY_CMP_TEST"*)
         continue 
         ;;
    esac
    # update the excluded tests
    UPDATE_EXCLUDE_TESTS=$(find components/$cmp/tests -perm -u+x -type f ! \( -name "*.[c|h]" -o -name "*.cu" -o -name "*.so" \) | grep -vw "$QUERY_CMP_TEST")
    EXCLUDE_TESTS="$EXCLUDE_TESTS $UPDATE_EXCLUDE_TESTS"
    # update active component tests
    ACTIVE_COMPONENTS_TESTS="$ACTIVE_COMPONENTS_TESTS $QUERY_CMP_TEST"
done

# print system information
echo "Platform:"
uname -a
# print date information
echo "Date:"
date

# print cpu information
echo ""
if [ -r /proc/cpuinfo ]; then
   echo "Cpuinfo:"
   # only print info on first processor on x86
   sed '/^$/q' /proc/cpuinfo
fi

echo ""
if [ "x$VALGRIND" != "x" ]; then
  echo "The following test cases will be run using valgrind:"
else
  echo "The following test cases will be run:"
fi

# list each test for each active component, note that if more
# than one test is output for each active component then
# this script is not behaving properly
echo $ACTIVE_COMPONENTS_TESTS

echo ""
echo "The following test cases will NOT be run:"
echo $EXCLUDE_TESTS;

# set LD_LIBRARY_PATH
if [ "$LD_LIBRARY_PATH" = "" ]; then
  LD_LIBRARY_PATH=.:./libpfm4/lib
else
  LD_LIBRARY_PATH=.:./libpfm4/lib:"$LD_LIBRARY_PATH"
fi
export LD_LIBRARY_PATH

echo ""
echo "Running a Single Component Test for --with-shlib-tools"
echo ""

for cmp_test in $ACTIVE_COMPONENTS_TESTS;
do
  if [ -x $cmp_test ]; then
    printf "Running $cmp_test:\n";
    printf "%-59s" ""
    cmp=`echo $cmp_test | sed 's:components/::' | sed 's:/.*$::'`;
    if [ x$cmp == xsde ]; then
      LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PWD}/components/sde/sde_lib:${PWD}/components/sde/tests/lib $VALGRIND ./$cmp_test $TESTS_QUIET
    else
      $VALGRIND ./$cmp_test $TESTS_QUIET
    fi 
  fi 
done
