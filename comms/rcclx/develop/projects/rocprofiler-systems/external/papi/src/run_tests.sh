#!/bin/sh

# File:    papi.c
# Author:  Philip Mucci
#          mucci@cs.utk.edu
# Mods:    Kevin London
#          london@cs.utk.edu
#          Philip Mucci
#          mucci@cs.utk.edu
#          Treece Burgess
#          tburgess@icl.utk.edu

# Make sure that the tests are built, if not build them
if [ "x$BUILD" != "x" ]; then
    cd testlib; make; cd ..
    cd validation_tests; make; cd ..
    cd ctests; make; cd ..
    cd ftests; make; cd ..
    for comp in `ls components/*/tests` ; do \
	cd components/$$comp/tests ; make; cd ../../.. ;
    done
fi

AIXTHREAD_SCOPE=S
export AIXTHREAD_SCOPE
if [ "X$1" = "X-v" ]; then
  TESTS_QUIET=""
else
# This should never have been an argument, but an environment variable!
  TESTS_QUIET="TESTS_QUIET"
  export TESTS_QUIET
fi

# Determine if cuda events are enabled or disabled
DISABLE_CUDA_EVENTS=""
if [ "$2" ]; then
    # Can either be --disable-cuda-events=<yes,no>
    DISABLE_CUDA_EVENTS=$2
fi

# Disable high-level output
if [ "x$TESTS_QUIET" != "xTESTS_QUIET" ] ; then
  export PAPI_REPORT=1
fi

if [ "x$VALGRIND" != "x" ]; then
  VALGRIND="valgrind --leak-check=full";
fi

# Check for active 'perf_event' component
PERF_EVENT_ACTIVE=$(utils/papi_component_avail | awk '/Active components:/{flag=1; next} flag' | grep -q "perf_event" && echo "true" || echo "false")

if [ "$PERF_EVENT_ACTIVE" = "true" ]; then
    VTESTS=`find validation_tests/* -prune -perm -u+x -type f ! -name "*.[c|h]"`;
    CTESTS=`find ctests/* -prune -perm -u+x -type f ! -name "*.[c|h]"`;
    #CTESTS=`find ctests -maxdepth 1 -perm -u+x -type f`;
    FTESTS=`find ftests -perm -u+x -type f ! -name "*.[c|h|F]"`;
else
  EXCLUDE="$EXCLUDE $VTESTS $CTESTS $FTESTS";
fi

# List of active components
ACTIVE_COMPONENTS_PATTERN=$(utils/papi_component_avail | awk '/Active components:/{flag=1; next} flag' | grep "Name:" | sed 's/Name: //' | awk '{print $1}' | paste -sd'|' -)

# Find the test files, filtering for only the active components
COMPTESTS=$(find components/*/tests -perm -u+x -type f ! \( -name "*.[c|h]" -o -name "*.cu" -o -name "*.so" \) | grep -E "components/($ACTIVE_COMPONENTS_PATTERN)/")

# Find the test files, filtering for inactive components
INACTIVE_COMPTESTS=$(find components/*/tests -perm -u+x -type f ! \( -name "*.[c|h]" -o -name "*.cu" -o -name "*.so" \) | grep -vE "components/($ACTIVE_COMPONENTS_PATTERN)/")

#EXCLUDE=`grep --regexp=^# --invert-match run_tests_exclude.txt`
EXCLUDE_TXT=`grep -v -e '^#\|^$' run_tests_exclude.txt`
EXCLUDE="$EXCLUDE_TXT $INACTIVE_COMPTESTS";

ALLTESTS="$VTESTS $CTESTS $FTESTS $COMPTESTS";

PATH=./ctests:$PATH
export PATH

echo "Platform:"
uname -a

echo "Date:"
date

echo ""
if [ -r /proc/cpuinfo ]; then
   echo "Cpuinfo:"
   # only print info on first processor on x86
   sed '/^$/q' /proc/cpuinfo
fi

echo ""
if [ "x$VALGRIND" != "x" ]; then
  echo "The following test cases will be run using valgrind:";
else
  echo "The following test cases will be run:";
fi
echo ""

MATCH=0
LIST=""
for i in $ALLTESTS;
do
  for xtest in $EXCLUDE;
  do
    if [ "$i" = "$xtest" ]; then
      MATCH=1
      break
    fi;
  done
  if [ $MATCH -ne 1 ]; then
	LIST="$LIST $i"
  fi;
  MATCH=0
done
echo $LIST
echo ""

echo ""
echo "The following test cases will NOT be run:";
echo $EXCLUDE;

echo "";
echo "Running Tests";
echo ""

if [ "$LD_LIBRARY_PATH" = "" ]; then
  LD_LIBRARY_PATH=.:./libpfm4/lib
else
  LD_LIBRARY_PATH=.:./libpfm4/lib:"$LD_LIBRARY_PATH"
fi
export LD_LIBRARY_PATH
if [ "$LIBPATH" = "" ]; then
  LIBPATH=.:./libpfm4/lib
else
  LIBPATH=.:./libpfm4/lib:"$LIBPATH"
fi
export LIBPATH


if [ "$PERF_EVENT_ACTIVE" = "true" ]; then
  
  echo ""
  echo "Running Event Validation Tests";
  echo ""
  
  for i in $VTESTS;
  do
    for xtest in $EXCLUDE;
    do
      if [ "$i" = "$xtest" ]; then
        MATCH=1
        break
      fi;
    done
    if [ $MATCH -ne 1 ]; then
      if [ -x $i ]; then
        RAN="$i $RAN"
        printf "Running %-50s %s" $i:
        $VALGRIND ./$i $TESTS_QUIET
        
        #delete output folder for high-level tests
        case "$i" in
          *"_hl"*) rm -r papi_hl_output ;;
        esac
  
      fi;
    fi;
    MATCH=0
  done
  
  echo ""
  echo "Running C Tests";
  echo ""
  
  for i in $CTESTS;
  do
    for xtest in $EXCLUDE;
    do
      if [ "$i" = "$xtest" ]; then
        MATCH=1
        break
      fi;
    done
    if [ $MATCH -ne 1 ]; then
      if [ -x $i ]; then
        RAN="$i $RAN"
        printf "Running %-50s %s" $i:
        # For all_native_events an optional flag of --disable-cuda-events=<yes,no>
        # can be provided; however, passing this to ctests/calibrate.c will result
        # in the help message being displayed instead of running
        if [ "$i" = "ctests/all_native_events" ] || [ "$i" = "ctests/get_event_component" ]; then
          $VALGRIND ./$i $TESTS_QUIET $DISABLE_CUDA_EVENTS
        else
          $VALGRIND ./$i $TESTS_QUIET
        fi
  
        #delete output folder for high-level tests
        case "$i" in
          *"_hl"*) rm -r papi_hl_output ;;
        esac
  
      fi;
    fi;
    MATCH=0
  done
  
  echo ""
  echo "Running Fortran Tests";
  echo ""
  
  for i in $FTESTS;
  do
    for xtest in $EXCLUDE;
    do
      if [ "$i" = "$xtest" ]; then
        MATCH=1
        break
      fi;
    done
    if [ $MATCH -ne 1 ]; then
      if [ -x $i ]; then
        RAN="$i $RAN"
        printf "Running $i:\n"
        $VALGRIND ./$i $TESTS_QUIET
  
        #delete output folder for high-level tests
        case "$i" in
          *"_hl"*) rm -r papi_hl_output ;;
        esac
  
      fi;
    fi;
    MATCH=0
  done
fi

echo "";
echo "Running Component Tests";
echo ""

for i in $COMPTESTS;
do
  for xtest in $EXCLUDE;
  do
    if [ "$i" = "$xtest" ]; then
      MATCH=1
      break
    fi;
  done
  if [ $MATCH -ne 1 ]; then
    if [ -x $i ]; then
	    RAN="$i $RAN"
        printf "Running $i:\n";
        printf "%-59s" ""
        cmp=`echo $i | sed 's:components/::' | sed 's:/.*$::'`;
        if [ x$cmp = xsde ]; then
            LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PWD}/components/sde/sde_lib:${PWD}/components/sde/tests/lib $VALGRIND ./$i $TESTS_QUIET
        else
            $VALGRIND ./$i $TESTS_QUIET
        fi;
    fi;
  fi;
  MATCH=0
done

if [ "$RAN" = "" ]; then 
	echo "FAILED to run any tests. (you can safely ignore this if this was expected behavior)"
fi;
