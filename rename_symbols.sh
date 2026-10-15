#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Prefixes every nccl*/pnccl* symbol in the given object files with
# torchcomms_, so the statically linked NCCLX does not clash with the NCCL
# PyTorch bundles. An optional second argument names a shared library whose
# exported symbols are left alone: they are resolved from that library at load
# (libcommsutils.so defines ncclCvarInit and friends), and a renamed reference
# would have nothing to bind to.

set -exo pipefail

FILES="$1"
KEEP_LIB="${2:-}"

# grep exits 1 when it selects no line, which is a legitimate outcome here (an
# object without nccl symbols, or one whose nccl symbols are all kept); any
# other status is a real failure and must stop the build.
grep_ok() {
    grep "$@" || [ $? -eq 1 ]
}

KEEP_FILE=
RENAME_FILE=
NCCL_SYMS=
SO_FILE=
rc=
for sig in 1 2 3 13 15; do eval "trap 'exit $((sig + 128))' $sig"; done
trap 'rc=$?; set +e; rm -f "$KEEP_FILE" "$RENAME_FILE" "$NCCL_SYMS" "$SO_FILE"; exit $rc' EXIT

KEEP_FILE=$(mktemp /tmp/torchcomms_keep_syms.txt.XXXXXX)
if [ -n "$KEEP_LIB" ]; then
    nm -D --defined-only "$KEEP_LIB" | awk '{print $NF}' | sort -u > "$KEEP_FILE"
    if [ ! -s "$KEEP_FILE" ]; then
        echo "rename_symbols.sh: $KEEP_LIB exports no symbols" >&2
        exit 1
    fi
fi

for FILE in ${FILES//;/ }
do
    echo "RENAMING: $FILE"
    RENAME_FILE=$(mktemp /tmp/torchcomms_rename_syms.txt.XXXXXX)
    NCCL_SYMS=$(mktemp /tmp/torchcomms_nccl_syms.txt.XXXXXX)

    nm -A "$FILE" | awk '{print $NF}' | grep_ok -e '^nccl' -e '^pnccl' | sort -u > "$NCCL_SYMS"
    grep_ok -vxF -f "$KEEP_FILE" "$NCCL_SYMS" | awk '{print $1 " torchcomms_" $1}' > "$RENAME_FILE"

    # A single object may legitimately rename nothing: it defines no nccl
    # symbol, or only header-inline ones the keep library also exports. The
    # static NCCLX archive always defines the public nccl* entry points, so an
    # empty list for it means the keep set swallowed everything and the
    # archive would ship unprefixed, the clash this script exists to prevent.
    if [[ "$FILE" == *.a ]] && [ ! -s "$RENAME_FILE" ]; then
        echo "rename_symbols.sh: nothing to rename in $FILE; every nccl symbol it defines is exported by $KEEP_LIB" >&2
        exit 1
    fi

    cat "$RENAME_FILE"

    SO_FILE=$(mktemp /tmp/torchcomms_rename.o.XXXXXX)
    objcopy --redefine-syms="$RENAME_FILE" "$FILE" "$SO_FILE"
    mv "$SO_FILE" "$FILE"
    SO_FILE=
    rm "$RENAME_FILE" "$NCCL_SYMS"
    RENAME_FILE=
    NCCL_SYMS=
done
