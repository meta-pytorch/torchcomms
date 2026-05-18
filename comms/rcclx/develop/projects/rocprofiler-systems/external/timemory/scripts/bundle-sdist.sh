#!/bin/bash

set -o errexit

: ${PYTHON_EXE:=python3}

_SCRIPT_DIR=$(bash -c "cd $(dirname ${BASH_SOURCE[0]}) && pwd")
_SOURCE_DIR=$(dirname ${_SCRIPT_DIR})

cd ${_SOURCE_DIR}

rm -f docs/.gitinfo
./scripts/generate-gitinfo.sh
echo "############### git info ###############"
cat ./docs/.gitinfo
echo "########################################"

rm -rf dist
${PYTHON_EXE} setup.py sdist
cd dist
sha256sum *
gpg --detach-sign -a *
# twine upload *
