#!/bin/bash -e

OUT=$1

echo Analysis output to $OUT

source /etc/profile
set +x
set -e
trap 'echo "# $BASH_COMMAND"' DEBUG
shopt -s expand_aliases

module load llvm
which scan-build
cd src
scan-build -o $OUT ./configure
scan-build -o $OUT make

