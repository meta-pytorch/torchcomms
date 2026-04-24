#!/bin/bash
# sync_to_oss.sh — Copy MCCL sources from fbsource to a torchcomms OSS clone.
# Internal-only helper for validating the OSS build on Meta devservers.
#
# Usage:
#   bash sync_to_oss.sh                           # sync to ~/local/torchcomms
#   bash sync_to_oss.sh /path/to/torchcomms       # sync to custom path
#   bash sync_to_oss.sh --build                   # sync + build
#   bash sync_to_oss.sh /path/to/torchcomms --build
#
# NOTE: generate_cvars writes nccl_cvars.{cc,h} into the synced comms/utils/cvars/.
#   This modifies the OSS clone, not fbsource. To restore the OSS repo:
#     cd ~/local/torchcomms && git checkout .

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FBCODE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- Parse arguments ---
TORCHCOMMS_DIR="$HOME/local/torchcomms"
DO_BUILD=false
for arg in "$@"; do
  case "$arg" in
    --build) DO_BUILD=true ;;
    *) TORCHCOMMS_DIR="$arg" ;;
  esac
done

if [ ! -d "$TORCHCOMMS_DIR/comms" ]; then
    echo "ERROR: torchcomms repo not found at $TORCHCOMMS_DIR"
    echo "Clone it: git clone https://github.com/meta-pytorch/torchcomms $TORCHCOMMS_DIR"
    exit 1
fi

echo "=== Syncing MCCL sources: $FBCODE_DIR → $TORCHCOMMS_DIR ==="

RSYNC_EXCLUDE=(
    --exclude='BUCK'
    --exclude='PACKAGE'
    --exclude='.claude'
)

# --- Core MCCL C++ library (MCCL-owned, use --delete) ---
# tests/ is included: needed by CMakeLists.txt for McclLinkTest and BUILD_TESTS.
# integration_tests/ is excluded: Buck-only, not part of the OSS cmake build.
rsync -a --delete \
    "${RSYNC_EXCLUDE[@]}" \
    --exclude='coordinator' \
    --exclude='experimental' \
    --exclude='fb' \
    --exclude='conda' \
    --exclude='benchmarks' \
    --exclude='integration_tests' \
    --exclude='build/*.sh' \
    --exclude='MetaJobMetadataProvider*' \
    "$FBCODE_DIR/comms/mccl/" "$TORCHCOMMS_DIR/comms/mccl/"

# --- Python bindings: torchcomms/fb/mccl/ → torchcomms/mccl/ (MCCL-owned, use --delete) ---
rsync -a --delete \
    "${RSYNC_EXCLUDE[@]}" \
    "$FBCODE_DIR/comms/torchcomms/fb/mccl/" "$TORCHCOMMS_DIR/comms/torchcomms/mccl/"

# --- Shared dependencies (not MCCL-owned, no --delete to preserve OSS-only files) ---
for dir in utils ctran common; do
    rsync -a \
        "${RSYNC_EXCLUDE[@]}" \
        --exclude='integration_tests' \
        "$FBCODE_DIR/comms/$dir/" "$TORCHCOMMS_DIR/comms/$dir/"
done

# --- torchcomms sources referenced by MCCL CMakeLists.txt (no --delete) ---
# CudaApi.cpp (line 333-334), Utils.cpp, TracingGuard.cpp (lines 337-338)
rsync -a \
    "${RSYNC_EXCLUDE[@]}" \
    "$FBCODE_DIR/comms/torchcomms/device/" "$TORCHCOMMS_DIR/comms/torchcomms/device/"
rsync -a \
    "${RSYNC_EXCLUDE[@]}" \
    "$FBCODE_DIR/comms/torchcomms/utils/" "$TORCHCOMMS_DIR/comms/torchcomms/utils/"

# --- TCPDM deps (optional — ENABLE_TCPDM=True enables tcp_devmem + ynl) ---
if [ -d "$FBCODE_DIR/comms/tcp_devmem" ]; then
    rsync -a \
        "${RSYNC_EXCLUDE[@]}" \
        "$FBCODE_DIR/comms/tcp_devmem/" "$TORCHCOMMS_DIR/comms/tcp_devmem/"
fi
if [ -d "$FBCODE_DIR/ynl" ]; then
    rsync -a \
        "${RSYNC_EXCLUDE[@]}" \
        "$FBCODE_DIR/ynl/" "$TORCHCOMMS_DIR/ynl/"
fi

# --- Build infra ---
cp "$SCRIPT_DIR/build_mccl.sh" "$TORCHCOMMS_DIR/build_mccl.sh"
mkdir -p "$TORCHCOMMS_DIR/fb/mccl"
cp "$SCRIPT_DIR/fb/mccl/CMakeLists.txt" "$TORCHCOMMS_DIR/fb/mccl/CMakeLists.txt"
cp "$SCRIPT_DIR/fb/mccl/setup.py" "$TORCHCOMMS_DIR/fb/mccl/setup.py"

echo "=== Sync complete ==="
echo "NOTE: fb/mccl/ and comms/mccl/ are new untracked dirs in the OSS repo."
echo "      This is expected — they will be committed when open-sourced."

if $DO_BUILD; then
    echo "=== Starting build ==="
    cd "$TORCHCOMMS_DIR"
    bash build_mccl.sh
fi
