#!/bin/bash

EXE=$(basename ${1})
DIR=heap.prof.${EXE}
mkdir -p ${DIR}

: ${N:=0}
: ${GPERF_PROFILE:=""}
: ${GPERF_PROFILE_BASE:=${DIR}/gperf}
: ${PPROF_ARGS:="--show_bytes --no_strip_temp"}
: ${MALLOCSTATS:=1}
: ${INTERACTIVE:=0}

while [ -z "${GPERF_PROFILE}" ]
do
    TEST_FILE=${GPERF_PROFILE_BASE}.${N}.0001.heap
    if [ ! -f "${TEST_FILE}" ]; then
        GPERF_PROFILE=${GPERF_PROFILE_BASE}.${N}
    fi
    N=$((${N}+1))
done

export MALLOCSTATS

echo -e "\n\t--> Outputting profile to '${GPERF_PROFILE}'...\n"

# remove profile file if unsucessful execution
cleanup-failure() { set +v ; echo "failure"; rm -f ${GPERF_PROFILE}; }
trap cleanup-failure SIGHUP SIGINT SIGQUIT SIGILL SIGABRT SIGKILL

ADD_LIBS()
{
    for i in $@
    do
        if [ -z "${ADD_LIB_LIST}" ]; then
            ADD_LIB_LIST="--add_lib=${i}"
        else
            ADD_LIB_LIST="${ADD_LIB_LIST} --add_lib=${i}"
        fi
    done
}

ADD_PRELOAD()
{
    for i in $@
    do
        if [ -z "${LIBS}" ]; then
            LIBS=${i}
        else
            LIBS="${LIBS}:${i}"
        fi
    done
}

# configure pre-loading of profiler library
PROJECT_LIBRARIES="$(find $PWD | egrep 'libtimemory|libctimemory' | egrep -v '\.a$|\.dSYM')"
ADD_LIBS ${PROJECT_LIBRARIES}
ADD_PRELOAD ${PROJECT_LIBRARIES}
if [ "$(uname)" = "Darwin" ]; then
    ADD_PRELOAD $(otool -L ${1} | egrep 'profiler' | awk '{print $1}')
    LIBS=$(echo ${LIBS} | sed 's/^://g')
    if [ -n "${LIBS}" ]; then
        export DYLD_FORCE_FLAT_NAMESPACE=1
        export DYLD_INSERT_LIBRARIES=${LIBS}
        echo "DYLD_INSERT_LIBRARIES=${DYLD_INSERT_LIBRARIES}"
    fi
else
    ADD_PRELOAD $(ldd ${1} | egrep 'profiler' | awk '{print $(NF-1)}')
    LIBS=$(echo ${LIBS} | sed 's/^://g')
    if [ -n "${LIB}" ]; then
        export LD_PRELOAD=${LIBS}
        echo "LD_PRELOAD=${LD_PRELOAD}"
    fi
fi

set -e
# run the application
eval HEAPPROFILE=${GPERF_PROFILE} $@ | tee ${GPERF_PROFILE}.log
set +e

# generate the results
shopt -s nullglob
FILES=${GPERF_PROFILE}*.heap
shopt -u nullglob

EXT=so
if [ "$(uname)" = "Darwin" ]; then EXT=dylib; fi
: ${PPROF:=$(which google-pprof)}
: ${PPROF:=$(which pprof)}
echo -e "Files: ${FILES}"
for i in ${FILES}
do
    if [ -n "${PPROF}" ]; then
        eval ${PPROF} --text ${ADD_LIB_LIST} ${PPROF_ARGS} --inuse_space   ${1} ${i} > ${i}.inuse_space.txt
        eval ${PPROF} --text ${ADD_LIB_LIST} ${PPROF_ARGS} --inuse_objects ${1} ${i} > ${i}.inuse_objects.txt
        eval ${PPROF} --text ${ADD_LIB_LIST} ${PPROF_ARGS} --alloc_space   ${1} ${i} > ${i}.alloc_space.txt
        eval ${PPROF} --text ${ADD_LIB_LIST} ${PPROF_ARGS} --alloc_objects ${1} ${i} > ${i}.alloc_objects.txt
        if [ "${INTERACTIVE}" -gt 0 ]; then
            eval ${PPROF} ${ADD_LIB_LIST} ${PPROF_ARGS} ${1} ${i}
        fi
    else
        echo -e "google-pprof/pprof not found!"
    fi
done
