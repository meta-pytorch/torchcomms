#!/bin/bash -e

DEBUG=$1
SHLIB=$2
COMPILER=$3

[ -z "$COMPILER" ] && COMPILER=gcc@11

source /etc/profile
set +x
set -e
trap 'echo "# $BASH_COMMAND"' DEBUG
shopt -s expand_aliases

module load $COMPILER

cd src

# configuring and installing PAPI
if [ "$SHLIB" = "without" ]; then
    ./configure --prefix=$PWD/cat-ci --with-debug=$DEBUG --enable-warnings
else
    ./configure --prefix=$PWD/cat-ci --with-debug=$DEBUG --enable-warnings --with-shlib-tools
fi
make -j4 && make install

# set environment variables for CAT
export PAPI_DIR=$PWD/cat-ci
export LD_LIBRARY_PATH=${PAPI_DIR}/lib:$LD_LIBRARY_PATH
cd counter_analysis_toolkit

# check detected architecture was correct
# note that the make here will finish
DETECTED_ARCH=$(make | grep -o 'ARCH.*' | head -n 1)
if [ "$DETECTED_ARCH" != "ARCH=X86" ]; then
    echo "Failed to detect appropriate architecture."
    exit 1
fi

# create output directory
mkdir OUT_DIR
# create real and fake events to monitor
echo "BR_INST_RETIRED 0" > event_list.txt
echo "PAPI_CI_FAKE_EVENT 0" >> event_list.txt
./cat_collect -in event_list.txt -out OUT_DIR -branch

cd OUT_DIR
# we expect this file to exist and have values 
[ -f BR_INST_RETIRED.branch ]
[ -s BR_INST_RETIRED.branch ]
# we expect this file to exist but be empty 
[ -f PAPI_CI_FAKE_EVENT.branch ]
[ ! -s PAPI_CI_FAKE_EVENT.branch ]
