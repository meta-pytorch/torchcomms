#!/bin/bash

set -o errexit

if [ ${PWD} = ${BASH_SOURCE[0]} ]; then
    cd ../..
fi

if [ -f "${PWD}/conda.yaml" ]; then
    conda-build .
fi
