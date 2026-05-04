#!/usr/bin/env bash

#  USAGE: ./run_test.sh [--env name val ...] <APP> [<LOG>].
#   --env name val: set environment variable name to the given value
#    <LOG>: if present then it will output of the tests in LOG.
#  Added env SOLLVE_TIMELIMIT to control the timeout value through env

export OMP_THREAD_LIMIT=$(lscpu -p | grep -c "^[0-9]")

# Providing option to set time limit through SOLLVE_TIMELIMIT env
export SOLLVE_TIMELIMIT=${SOLLVE_TIMELIMIT:-60}

function report ()
{
  # $1= app, $2=status, $3=output
  msg="FAIL"
  if [ $2 -eq 0 ]; then
    msg="PASS"
  fi
  if [ $2 -eq 124 ]; then
    msg="FAIL: TEST HAS TIMEOUT"
  fi
  if [ -n "$3" ]; then
    echo "$1: $msg. exit code: $2"
    RED='\033[0;31m'
    NC='\033[0m' # No Color
    echo -e "${RED}$1:\n${3}${NC}" 1>&2
  else
    echo "$1: $msg. exit code: $2"
  fi
}

declare -a env_data
while [[ "$1" = "--env" ]]; do
  if [[ -z "$2" ]]; then
    echo 'ERROR: Expected 'name val' arguments with --env' 1>&2
    exit -1
  fi
  env_data+=("$2" "$3")
  shift 3
done

if [ "$#" -lt "1" ]; then
  exit -1
elif [ ! -f "$1" ]; then
  RED='\033[0;31m'
  NC='\033[0m' # No Color
  echo -e "${RED}$1: Test not found${3}${NC}" 1>&2
  exit -1
fi

app=$1
output=$(for ((idx=0; $idx < ${#env_data[*]}; idx=$((idx+2)))); do export "${env_data[$idx]}"="${env_data[$((idx+1))]}"; done; timeout $SOLLVE_TIMELIMIT "$app" 2>&1)
status=$?
output=$(printf '%s\n' "${output}" | uniq)

if [ -z $2 ]; then
  report $(basename $app) $status
else
  report $(basename $app) $status "$output"
fi

unset report
exit $status
