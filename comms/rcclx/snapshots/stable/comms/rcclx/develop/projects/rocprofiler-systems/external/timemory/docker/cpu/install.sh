#!/bin/bash -e

: ${DEVELOP:=0}
: ${BRANCH:=develop}

if [ "${DEVELOP}" -gt 1 ]; then
    /tmp/apt.sh
else
    git clone https://github.com/NERSC/timemory.git timemory
    git checkout ${BRANCH}
    mkdir build
    pushd build
    cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo 
fi
