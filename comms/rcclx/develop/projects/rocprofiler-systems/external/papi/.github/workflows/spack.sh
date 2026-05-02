#!/bin/bash -e

COMPILER=$1

[ -z "$COMPILER" ] && COMPILER=gcc@11

source /etc/profile
set +x
set -e
trap 'echo "# $BASH_COMMAND"' DEBUG
shopt -s expand_aliases

export HOME=`pwd`
git clone https://github.com/spack/spack spack || true
source spack/share/spack/setup-env.sh

SPEC="papi@master +lmsensors +powercap +sde +infiniband +rapl +cuda %$COMPILER ^cuda@12.5.1"
echo SPEC=$SPEC

module load $COMPILER
spack compiler find
spack install --fresh --only=dependencies $SPEC
spack dev-build -i --fresh --test=root $SPEC
spack test run papi
