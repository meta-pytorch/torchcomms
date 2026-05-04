#!/bin/bash

set -o errexit

: ${N:=0}
: ${CPUPROFILE:=""}
: ${CPUPROFBASE:=gperf.cpu.prof}
: ${PPROF_ARGS:=""}
: ${MALLOCSTATS:=1}
: ${CPUPROFILE_FREQUENCY:=250}
: ${INTERACTIVE:=0}

while [ -z "${CPUPROFILE}" ]
do
    TEST_FILE=${CPUPROFBASE}.${N}
    if [ ! -f "${TEST_FILE}" ]; then
        CPUPROFILE=${TEST_FILE}
    fi
    N=$((${N}+1))
done

export CPUPROFILE
export MALLOCSTATS
export CPUPROFILE_FREQUENCY
echo -e "\n\t--> Outputting profile to '${CPUPROFILE}'...\n"

# remove profile file if unsucessful execution
cleanup-failure() { set +v ; rm -f ${CPUPROFILE}; }
trap cleanup-failure SIGHUP SIGINT SIGQUIT SIGILL SIGABRT SIGKILL

# configure pre-loading of profiler library
LIBS=$(find $PWD | grep libptl | egrep -v '\.a$|\.dSYM')
LIBS=$(echo ${LIBS} | sed 's/ /:/g')
if [ "$(uname)" = "Darwin" ]; then
    for i in $(otool -L ${1} | egrep 'profiler' | awk '{print $1}')
    do
        LIBS=${LIBS}:${i}
    done
    LIBS=$(echo ${LIBS} | sed 's/^://g')
    if [ -n "${LIBS}" ]; then
        export DYLD_FORCE_FLAT_NAMESPACE=1
        export DYLD_INSERT_LIBRARIES=${LIBS}
        echo "DYLD_INSERT_LIBRARIES=${DYLD_INSERT_LIBRARIES}"
    fi
else
    for i in $(ldd ${1} | egrep 'profiler' | awk '{print $(NF-1)}')
    do
        LIBS=${LIBS}:${i}
    done
    LIBS=$(echo ${LIBS} | sed 's/^://g')
    if [ -n "${LIB}" ]; then
        export LD_PRELOAD=${LIBS}
        echo "LD_PRELOAD=${LD_PRELOAD}"
    fi
fi

# run the application
eval $@ | tee ${CPU_PROFILE}.log

# generate the results
EXT=so
if [ "$(uname)" = "Darwin" ]; then EXT=dylib; fi
if [ -f "${CPUPROFILE}" ]; then
    : ${PPROF:=$(which pprof)}
    if [ -n "${PPROF}" ]; then
        pprof --text --add_lib=libptl.${EXT} ${PPROF_ARGS} ${1} ${CPUPROFILE} | egrep -v ' 0x[0-9]' &> ${CPUPROFILE}.txt
        pprof --text --cum --add_lib=libptl.${EXT} ${PPROF_ARGS} ${1} ${CPUPROFILE} | egrep -v ' 0x[0-9]' &> ${CPUPROFILE}.cum.txt
        if [ "${INTERACTIVE}" -gt 0 ]; then
            pprof --add_lib=libptl.${EXT} ${PPROF_ARGS} ${1} ${CPUPROFILE}
        fi
    fi
fi
