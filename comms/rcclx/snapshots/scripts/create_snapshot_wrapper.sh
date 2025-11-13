#!/bin/bash
# Wrapper script to create snapshots from buck2 build outputs
# This script must be run from the fbcode directory

set -e

# Check if we're in the right directory
if [[ ! -f "comms/rcclx/BUCK" ]]; then
    echo "Error: This script must be run from the fbcode directory"
    exit 1
fi

# Parse command line arguments
ROCM_VERSION=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --rocm-version)
            ROCM_VERSION="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --rocm-version <6.2|6.4|7.0>"
            exit 1
            ;;
    esac
done

if [[ -z "$ROCM_VERSION" ]]; then
    echo "Error: --rocm-version is required"
    echo "Usage: $0 --rocm-version <6.2|6.4|7.0>"
    exit 1
fi

# Display warning and ask for confirmation
echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                              ⚠️  WARNING  ⚠️                                ║"
echo "╠════════════════════════════════════════════════════════════════════════════╣"
echo "║                                                                            ║"
echo "║  THIS SCRIPT WILL UPDATE THE MANIFOLD STABLE ARTIFACT REFERENCE!          ║"
echo "║                                                                            ║"
echo "║  The following operations will be performed:                               ║"
echo "║  1. Archive current last-stable to archives (Manifold + repo metadata)    ║"
echo "║  2. Move current stable to last-stable (Manifold + repo metadata)         ║"
echo "║  3. Upload NEW artifact to stable in Manifold                             ║"
echo "║                                                                            ║"
echo "║  This will affect the stable artifact for ROCm ${ROCM_VERSION}                            ║"
echo "║                                                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
read -r -p "Do you want to proceed? (yes/no): " CONFIRM

# Convert to lowercase for comparison
CONFIRM=$(echo "$CONFIRM" | tr '[:upper:]' '[:lower:]')

if [[ "$CONFIRM" != "yes" ]]; then
    echo "Operation cancelled by user."
    exit 0
fi

echo ""
echo "Proceeding with snapshot creation..."
echo ""

# Determine ROCm constraint
case $ROCM_VERSION in
    6.2)
        ROCM_CONSTRAINT="ovr_config//third-party/rocm/constraints:6.2.1"
        ;;
    6.4)
        ROCM_CONSTRAINT="ovr_config//third-party/rocm/constraints:6.4.2"
        ;;
    7.0)
        ROCM_CONSTRAINT="ovr_config//third-party/rocm/constraints:7.0"
        ;;
    *)
        echo "Error: Unsupported ROCm version: $ROCM_VERSION"
        echo "Supported versions: 6.2, 6.4, 7.0"
        exit 1
        ;;
esac

echo "Creating snapshot for ROCm $ROCM_VERSION..."
echo "Building rcclx-dev library..."

# Build the rcclx-dev library
buck2 build @fbcode//mode/opt-amd-gpu -m rcclx_dev -m "$ROCM_CONSTRAINT" fbcode//comms/rcclx:rcclx-dev

# Get the library path from buck2 (relative to fbsource root)
LIBRARY_PATH_RELATIVE=$(buck2 build @fbcode//mode/opt-amd-gpu -m rcclx_dev -m "$ROCM_CONSTRAINT" fbcode//comms/rcclx:rcclx-dev --show-output | grep -oP 'fbcode//comms/rcclx:rcclx-dev\s+\K.*')

# Convert to absolute path (buck-out is at fbsource level, we're in fbcode)
LIBRARY_PATH="../$LIBRARY_PATH_RELATIVE"

if [[ ! -f "$LIBRARY_PATH" ]]; then
    echo "Error: Could not find built library at $LIBRARY_PATH"
    echo "Tried relative path: $LIBRARY_PATH_RELATIVE"
    exit 1
fi

echo "Library built at: $LIBRARY_PATH"
echo "Running snapshot script..."

# Run the snapshot creation script using buck2 run to get proper Python dependencies
buck2 run fbcode//comms/rcclx/snapshots/scripts:create_snapshot -- \
    --library "$LIBRARY_PATH" \
    --rocm-version "$ROCM_VERSION" \
    --snapshots-root "$(pwd)/comms/rcclx/snapshots" \
    --rcclx-repo "$(pwd)/comms/rcclx"

echo "Snapshot created successfully for ROCm $ROCM_VERSION"
echo ""
echo "Generating stable_checksums.bzl from metadata files..."

# Generate the checksums.bzl file from metadata
buck2 run fbcode//comms/rcclx/snapshots/scripts:generate_checksums_bzl -- \
    --snapshots-root "$(pwd)/comms/rcclx/snapshots" \
    --output "$(pwd)/comms/rcclx/stable_checksums.bzl"

echo ""
echo "✓ Snapshot creation completed successfully!"
echo "  - Artifact uploaded to Manifold: rcclx_prebuilt_artifacts/tree/stable/$ROCM_VERSION/librcclx-dev.a"
echo "  - Metadata saved to: comms/rcclx/snapshots/stable/$ROCM_VERSION/metadata.txt"
echo "  - Checksums updated in: comms/rcclx/stable_checksums.bzl"
echo ""
echo "The BUCK file will automatically use the checksum from stable_checksums.bzl"
