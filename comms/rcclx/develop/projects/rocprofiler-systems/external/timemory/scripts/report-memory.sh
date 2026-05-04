#!/bin/bash

run()
{
    echo ""
    date
    if [ $(uname) = "Darwin" ]; then
        ps -caxm -orss= | awk '{ sum += $1 } END { print "Resident Set Size: " sum/1024 " MiB" }'
        vm_stat -c 1 1
    else
        free -m
        vmstat -S m 1 1
    fi
    echo ""
}

F=""
N=1000
W=60

ulimit -a

if [ -f "$1" ]; then
    F="$1"
    N=0;
else
    if [ -n "$1" ]; then N=$1; fi
fi

if [ -n "$2" ]; then W=$2; fi

if [ "${N}" -eq 0 ]; then
    while [ -f "${F}" ]; do
        run
        sleep ${W}
    done
else
    for i in $(seq 0 1 ${N})
    do
        run
        sleep ${W}
    done
fi
