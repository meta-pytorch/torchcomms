#!/bin/bash

echo "Executing ${BASH_SOURCE[0]}..."
DIR=$1

if [ -z "${DIR}" ]; then echo "No directory provided!"; exit 1; fi
if [ ! -d "${DIR}" ]; then DIR=$(realpath ${DIR}); fi
if [ -z "${DIR}" ]; then echo "No directory provided!"; exit 1; fi

set -e

echo "Directory: ${DIR}..."
cd ${DIR}

echo "Generating coverage..."
lcov --directory . --capture --output-file coverage.info
echo "Removing coverage..."
lcov --remove coverage.info '/usr/*' '/tmp/*' "${HOME}"'/.cache/*' '*/external/*' '*/examples/*' '*/tests/*' '*/tools/*' '*/python/*' '*/timemory/tpls/*' '*/signals.hpp' '*/popen.cpp' --output-file coverage.info
echo "Listing coverage..."
lcov --list coverage.info
echo "Submitting coverage..."
bash <(curl -s https://codecov.io/bash) -f coverage.info || echo "Codecov did not collect coverage reports"
