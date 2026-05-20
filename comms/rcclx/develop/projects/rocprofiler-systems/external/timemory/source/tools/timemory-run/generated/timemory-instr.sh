#!/bin/bash -e

NAME=$(basename ${BASH_SOURCE[0]})
# path to timemory-run
: ${RUNEXE:="$(dirname $(realpath ${BASH_SOURCE[0]}))/timemory-run"}
# the libraries to generate instrumentation for
: ${TARGETS:="omp gomp iomp5 mpi pmpi mpicxx"}
# the output directory
: ${OUT:=${PWD}/instr}
# the executable
: ${EXE:=$(echo "$@" | sed 's/.* -- //g' | awk '{print $1}')}
# the timemory-run command
: ${CMD:=$(echo "$@" | sed 's/ -- .*//g' | awk '{$1=""}1' | awk '{$1=$1}1')}

usage()
{
    echo -e "\n${NAME} : automates adding instrumentation to libraries linked to executable using timemory-run\n"
    ${RUN_EXE} --help
    exit -1
}

info_msg()
{
    echo -e "\nAppend '${OUT}' to the runtime linker path before executing, e.g.\n"
    echo -e "\n\n\texport LD_LIBRARY_PATH=${OUT}:\${LD_LIBRARY_PATH}\n\n"
}

if [ ! -e "${RUNEXE}" ]; then
    RUNEXE=$(which timemory-run)
fi

if [ -z "${RUNEXE}" ]; then
    echo -e "\nError! 'timemory-run' could not be found. Please set RUNEXE\n"
    exit -1
fi

echo -e "\nInstrumented libraries \"${TARGETS}\" (if linked) will be in ${OUT}"

mkdir -p ${OUT}

ARGS="$@"

if [ -z "${EXE}" ]; then
    echo "\nCould not find '--' argument. Please provide EXE in environment or use '--'\n"
    usage
fi

for i in ${TARGETS}
do
    LIBS=$(ldd ${EXE} | grep "lib${i}" | awk '{print $(NF-1)'})
    for j in ${LIBS}
    do
	if [ -z "$(nm ${j} | egrep '_init|_fini')" ]; then
	    echo -e "\nWarning! Library '${j}' does not have '_init' or '_fini' symbols!\n"
            continue
	fi
        if [ ! -e "${j}" ]; then
            echo -e "\nWarning! Library '${j}' does not exist!\n"
            continue
        fi
        NAME=$(basename ${j})
        echo -e "\n##### Executing '${RUNEXE} ${CMD} -o ${OUT}/${NAME} -- ${j}'...\n"
        ${RUNEXE} ${CMD} -o ${OUT}/${NAME} -- ${j}
    done
done

info_msg
