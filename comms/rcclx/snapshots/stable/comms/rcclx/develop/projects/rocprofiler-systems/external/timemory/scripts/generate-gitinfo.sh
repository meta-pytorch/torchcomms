#!/bin/bash

set -o errexit

_SCRIPT_DIR=$(bash -c "cd $(dirname ${BASH_SOURCE[0]}) && pwd")
_SOURCE_DIR=$(dirname ${_SCRIPT_DIR})

cd ${_SOURCE_DIR}

: ${GIT_EXECUTABLE:=git}
: ${OUTPUT:=${_SOURCE_DIR}/docs/.gitinfo}

if [ -n "$1" ]; then OUTPUT=${1}; shift; fi

${GIT_EXECUTABLE} rev-parse HEAD > ${OUTPUT}
${GIT_EXECUTABLE} describe --tags >> ${OUTPUT}
