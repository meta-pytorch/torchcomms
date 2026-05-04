#!/bin/bash -e

set -e

docker-compose build --pull ${@}
# docker build ./cpu -t NERSC/timemory:cpu
# docker build ./gpu -t NERSC/timemory:gpu
