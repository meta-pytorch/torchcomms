#!/bin/bash

set -o errexit

if [ ${PWD} = ${BASH_SOURCE[0]} ]; then
    cd ../..
fi

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

#for i in 2.{7,8,9} 3.{0,1,2,3,4,5,6,7}
for i in 2.7 3.4 3.5 3.6
do
    
    PIP_CMD=$(get_pip pip-${i} pip${i})
    if [ -z "${PIP_CMD}" ]
    then
        continue
    fi
    
    echo -e "\n\n---> Building for Python version ${i}\n\n"
    
    set +e
    eval ${PIP_CMD} uninstall -y timemory
    set -e
    rm -rf TiMemory.egg-info dist build .eggs
    eval ${PIP_CMD} install --no-cache-dir -U .
    RET=$?
    if [ "${RET}" -gt 0 ]; then
        break
    fi
done
