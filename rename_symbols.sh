#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

set -ex

FILES="$1"

for FILE in ${FILES//;/ }
do
    echo "RENAMING: $FILE"
    RENAME_FILE=$(mktemp /tmp/torchcomms_rename_syms.txt.XXXXXX)

    nm -A "$FILE" | awk '{print $NF}' | grep -e '^nccl' -e '^pnccl' | \
    awk '{print $1 " torchcomms_" $1}' | sort | uniq > "$RENAME_FILE"

    cat "$RENAME_FILE"

    SO_FILE=$(mktemp /tmp/torchcomms_rename.o.XXXXXX)
    objcopy --redefine-syms="$RENAME_FILE" "$FILE" "$SO_FILE"
    mv "$SO_FILE" "$FILE"
    rm "$RENAME_FILE"
done
