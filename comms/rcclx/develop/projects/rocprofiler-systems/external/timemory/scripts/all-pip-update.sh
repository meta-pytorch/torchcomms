#!/bin/bash

set -o errexit

get_pip()
{
    for i in $@
    do
        if eval command -v ${i} &> /dev/null; then
            echo ${i}
            return
        fi
    done
    echo ""
}

for i in 3.{7..4} 2.7
do
    
    PIP_CMD=$(get_pip pip-${i} pip${i})
    if [ -z "${PIP_CMD}" ]
    then
        continue
    fi
    
    echo -e "\n\n---> Building for Python version ${i}\n\n"
    
    eval ${PIP_CMD} uninstall -y timemory
    rm -rf TiMemory.egg-info dist build .eggs
    eval ${PIP_CMD} install -U timemory
    RET=$?
    if [ "${RET}" -gt 0 ]; then
        break
    fi
done
