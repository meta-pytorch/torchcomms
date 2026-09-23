#!/bin/bash
# Hipify external comms/ headers that are not part of RCCL's source tree
# but are included via -I/data/users/.../fbcode

HIPIFY_PERL=$1
FBCODE_DIR=$2
SHIM_DIR=$3

if [ -z "$HIPIFY_PERL" ] || [ -z "$FBCODE_DIR" ] || [ -z "$SHIM_DIR" ]; then
  echo "Usage: $0 <hipify-perl> <fbcode_dir> <shim_dir>"
  exit 1
fi

mkdir -p "$SHIM_DIR"

for dir in comms/common comms/utils comms/torchcomms; do
  src="$FBCODE_DIR/$dir"
  [ -d "$src" ] || continue
  while read -r f; do
    rel="${f#$FBCODE_DIR/}"
    out="$SHIM_DIR/$rel"
    mkdir -p "$(dirname "$out")"
    if ! $HIPIFY_PERL -quiet-warnings "$f" -o "$out"; then
      echo "ERROR: hipify-perl failed on $f" >&2
      exit 1
    fi
  done < <(find "$src" -type f \( -name '*.h' -o -name '*.cuh' -o -name '*.hpp' \))
done
