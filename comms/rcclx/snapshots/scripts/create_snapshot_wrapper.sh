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
ROCM_VERSIONS=""
COPY_STABLE=false
ONLY_STABLE=false
SKIP_BACKUP=false

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options for creating a new snapshot (default mode):"
    echo "  --rocm-version <version>   ROCm version (6.2, 6.4, or 7.0) - REQUIRED"
    echo "  --only-stable              Update only stable, leave last-stable unchanged"
    echo "  --skip-backup              Skip backup before making changes (not recommended)"
    echo ""
    echo "Options for copying stable to last-stable (--copy-stable mode):"
    echo "  --copy-stable              Copy current stable to last-stable for all versions"
    echo "  --rocm-versions <list>     Comma-separated ROCm versions (default: 6.2,6.4,7.0)"
    echo "  --skip-backup              Skip backup before making changes (not recommended)"
    echo ""
    echo "Examples:"
    echo "  # Create a new snapshot for ROCm 6.4"
    echo "  $0 --rocm-version 6.4"
    echo ""
    echo "  # Create a new snapshot, only updating stable (not rotating to last-stable)"
    echo "  $0 --rocm-version 6.4 --only-stable"
    echo ""
    echo "  # Copy stable to last-stable for all ROCm versions"
    echo "  $0 --copy-stable"
    echo ""
    echo "  # Copy stable to last-stable for specific ROCm versions"
    echo "  $0 --copy-stable --rocm-versions 6.4,7.0"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --rocm-version)
            ROCM_VERSION="$2"
            shift 2
            ;;
        --rocm-versions)
            ROCM_VERSIONS="$2"
            shift 2
            ;;
        --copy-stable)
            COPY_STABLE=true
            shift
            ;;
        --only-stable)
            ONLY_STABLE=true
            shift
            ;;
        --skip-backup)
            SKIP_BACKUP=true
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate arguments based on mode
if [[ "$COPY_STABLE" == "true" ]]; then
    # Copy stable mode
    if [[ "$ONLY_STABLE" == "true" ]]; then
        echo "Error: --copy-stable and --only-stable are mutually exclusive"
        exit 1
    fi
    if [[ -n "$ROCM_VERSION" ]]; then
        echo "Warning: --rocm-version is ignored in --copy-stable mode. Use --rocm-versions instead."
    fi
else
    # Normal snapshot creation mode
    if [[ -z "$ROCM_VERSION" ]]; then
        echo "Error: --rocm-version is required for snapshot creation"
        show_usage
        exit 1
    fi
fi

# Build extra args for create_snapshot
EXTRA_ARGS=""
if [[ "$SKIP_BACKUP" == "true" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --skip-backup"
fi

if [[ "$COPY_STABLE" == "true" ]]; then
    # =====================================================================
    # COPY STABLE MODE
    # =====================================================================
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║                         COPY STABLE TO LAST-STABLE                         ║"
    echo "╠════════════════════════════════════════════════════════════════════════════╣"
    echo "║                                                                            ║"
    echo "║  This will copy current stable snapshots to last-stable.                   ║"
    echo "║                                                                            ║"
    echo "║  Operations:                                                               ║"
    echo "║  1. Create a backup of all stable and last-stable artifacts                ║"
    echo "║  2. Copy stable to last-stable for each ROCm version                       ║"
    echo "║                                                                            ║"
    if [[ -n "$ROCM_VERSIONS" ]]; then
        echo "║  ROCm versions: $ROCM_VERSIONS"
    else
        echo "║  ROCm versions: 6.2,6.4,7.0 (default)                                      ║"
    fi
    echo "║                                                                            ║"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    read -r -p "Do you want to proceed? (yes/no): " CONFIRM

    CONFIRM=$(echo "$CONFIRM" | tr '[:upper:]' '[:lower:]')

    if [[ "$CONFIRM" != "yes" ]]; then
        echo "Operation cancelled by user."
        exit 0
    fi

    echo ""
    echo "Proceeding with copy-stable operation..."
    echo ""

    # Build the command
    CMD="buck2 run fbcode//comms/rcclx/snapshots/scripts:create_snapshot -- \
        --snapshots-root $(pwd)/comms/rcclx/snapshots \
        --copy-stable"

    if [[ -n "$ROCM_VERSIONS" ]]; then
        CMD="$CMD --rocm-versions $ROCM_VERSIONS"
    fi

    if [[ -n "$EXTRA_ARGS" ]]; then
        CMD="$CMD $EXTRA_ARGS"
    fi

    eval $CMD

    echo ""
    echo "Generating stable_checksums.bzl from metadata files..."

    buck2 run fbcode//comms/rcclx/snapshots/scripts:generate_checksums_bzl -- \
        --snapshots-root "$(pwd)/comms/rcclx/snapshots" \
        --output "$(pwd)/comms/rcclx/stable_checksums.bzl"

    echo ""
    echo "✓ Copy-stable operation completed successfully!"
    echo "  - Stable snapshots copied to last-stable"
    echo "  - Checksums updated in: comms/rcclx/stable_checksums.bzl"

else
    # =====================================================================
    # NORMAL SNAPSHOT CREATION MODE
    # =====================================================================

    # Display warning and ask for confirmation
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║                              ⚠️  WARNING  ⚠️                                ║"
    echo "╠════════════════════════════════════════════════════════════════════════════╣"
    echo "║                                                                            ║"
    echo "║  THIS SCRIPT WILL UPDATE THE MANIFOLD STABLE ARTIFACT REFERENCE!          ║"
    echo "║                                                                            ║"
    if [[ "$ONLY_STABLE" == "true" ]]; then
        echo "║  Mode: ONLY-STABLE (last-stable will NOT be modified)                     ║"
        echo "║                                                                            ║"
        echo "║  The following operations will be performed:                               ║"
        echo "║  1. Create a backup of all stable and last-stable artifacts                ║"
        echo "║  2. Upload NEW artifact to stable in Manifold                              ║"
        echo "║  3. Update stable metadata in repo                                         ║"
        echo "║                                                                            ║"
        echo "║  Last-stable will remain UNCHANGED.                                        ║"
    else
        echo "║  Mode: NORMAL (stable will rotate to last-stable)                          ║"
        echo "║                                                                            ║"
        echo "║  The following operations will be performed:                               ║"
        echo "║  1. Create a backup of all stable and last-stable artifacts                ║"
        echo "║  2. Move current stable to last-stable (Manifold + repo metadata)          ║"
        echo "║  3. Upload NEW artifact to stable in Manifold                              ║"
    fi
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
    echo "Note: First-time builds will take 20-30 minutes as we build from source without cache"

    # Build the rcclx-dev library
    # Use --no-remote-cache to disable remote cache queries/writes
    # Use --local-only to force local execution, preventing use of remote execution
    # which could otherwise provide prebuilt artifacts. This ensures we always do a
    # fresh local build when creating snapshots.
    buck2 build @fbcode//mode/opt-amd-gpu -m rcclx_dev -m "$ROCM_CONSTRAINT" --no-remote-cache --local-only fbcode//comms/rcclx:rcclx-dev

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

    # Define header path
    HEADER_PATH="$(pwd)/comms/rcclx/develop/src/rccl.h"

    if [[ ! -f "$HEADER_PATH" ]]; then
        echo "Error: Could not find rccl.h header at $HEADER_PATH"
        exit 1
    fi

    echo "Header found at: $HEADER_PATH"
    echo "Running snapshot script..."

    # Build extra args for only-stable mode
    if [[ "$ONLY_STABLE" == "true" ]]; then
        EXTRA_ARGS="$EXTRA_ARGS --only-stable"
    fi

    # Run the snapshot creation script using buck2 run to get proper Python dependencies
    buck2 run fbcode//comms/rcclx/snapshots/scripts:create_snapshot -- \
        --library "$LIBRARY_PATH" \
        --rocm-version "$ROCM_VERSION" \
        --snapshots-root "$(pwd)/comms/rcclx/snapshots" \
        --rcclx-repo "$(pwd)/comms/rcclx" \
        --header "$HEADER_PATH" \
        $EXTRA_ARGS

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
    if [[ "$ONLY_STABLE" == "true" ]]; then
        echo "  - Last-stable was NOT modified (--only-stable mode)"
    fi
    echo ""
    echo "The BUCK file will automatically use the checksum from stable_checksums.bzl"
fi
