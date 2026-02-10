# RCCLX Snapshots

This directory contains the infrastructure for managing snapshots of rcclx sources for the `rcclx-stable` and `rcclx-last-stable` targets.

## Overview

The snapshot system uses **pre-extracted sources** committed to the repository. When building `rcclx-stable` or `rcclx-last-stable`, the sources are compiled from the snapshot directory rather than the main rcclx directory.

This approach **eliminates ABI mismatch issues** because all code (rcclx + dependencies) compiles together at build time with current external headers (folly, scribe, thrift, etc.).

## Quick Start

### Creating a New Snapshot

```bash
cd fbcode

# Snapshot from specific commit
python3 comms/rcclx/snapshots/scripts/create_snapshot.py \
    --stage stable \
    --commit <commit-hash> \
    --snapshots-root comms/rcclx/snapshots \
    --repo-root /path/to/fbsource

# Snapshot from current HEAD
python3 comms/rcclx/snapshots/scripts/create_snapshot.py \
    --stage stable \
    --snapshots-root comms/rcclx/snapshots \
    --repo-root /path/to/fbsource

# Rotate stable → last-stable, then create new stable
python3 comms/rcclx/snapshots/scripts/create_snapshot.py \
    --stage stable \
    --commit <commit-hash> \
    --rotate \
    --snapshots-root comms/rcclx/snapshots \
    --repo-root /path/to/fbsource
```

### Building with Snapshots

```bash
# Build with stable snapshot
buck2 build //comms/rcclx:rcclx-stable --modifier=rocm70

# Build with last-stable snapshot
buck2 build //comms/rcclx:rcclx-last-stable --modifier=rocm70

# Build using modifier to select snapshot
buck2 build -m rcclx_stable //comms/rcclx:rcclx --modifier=rocm70
```

## Directory Structure

```
snapshots/
├── stable/
│   ├── rcclx/                    # Pre-extracted rcclx sources
│   │   ├── BUCK
│   │   ├── defs.bzl
│   │   ├── rccl_build_config.bzl
│   │   └── develop/
│   │       ├── src/
│   │       ├── meta/
│   │       └── ...
│   └── metadata.txt              # Commit hash, timestamp
├── last-stable/
│   ├── rcclx/                    # Pre-extracted rcclx sources
│   └── metadata.txt
├── scripts/
│   ├── create_snapshot.py        # Main snapshot creation script
│   └── constants.py              # Configuration constants
├── DESIGN_PRE_EXTRACTED_SOURCES.md  # Design document
└── README.md                     # This file
```

## How It Works

### Build-Time Flow

```
User runs: buck2 build //comms/rcclx:rcclx-stable --modifier=rocm70

Buck2 resolves:
  //comms/rcclx:rcclx-stable
    → alias to //comms/rcclx/snapshots/stable/rcclx:rcclx-dev

Buck2 builds from:
  snapshots/stable/rcclx/BUCK (frozen rcclx source)
    ├── //comms/ctran:...        (from HEAD)
    ├── //comms/utils:...        (from HEAD)
    ├── //comms/common:...       (from HEAD)
    └── //folly:..., //scribe:.. (from HEAD)

Result:
  All code compiled together with current headers
  → No ABI mismatch
```

### Why This Solves ABI Issues

The previous "bundled dependencies" approach precompiled internal dependencies (`librcclxdeps.a`) which caused ABI mismatches when external dependencies (folly, scribe, thrift) changed.

With pre-extracted sources:
- **rcclx sources** are frozen at snapshot time
- **Internal deps** (ctran, utils, logger) compile from HEAD
- **External deps** (folly, scribe, thrift) compile from HEAD
- **All code uses the same headers** at compile time = No ABI mismatch

## Script Reference

### create_snapshot.py

Main script for creating source snapshots.

**Arguments:**
| Argument | Required | Description |
|----------|----------|-------------|
| `--stage` | Yes | Snapshot stage: `stable` or `last-stable` |
| `--commit` | No | Commit hash to snapshot from (default: current HEAD) |
| `--snapshots-root` | Yes | Path to snapshots directory |
| `--repo-root` | Yes | Path to repository root |
| `--rotate` | No | If creating stable, first copy current stable to last-stable |

**Examples:**
```bash
# Create stable snapshot from specific commit
python3 create_snapshot.py \
    --stage stable \
    --commit abc123def456 \
    --snapshots-root comms/rcclx/snapshots \
    --repo-root /path/to/fbsource

# Rotate and create new stable
python3 create_snapshot.py \
    --stage stable \
    --commit abc123def456 \
    --rotate \
    --snapshots-root comms/rcclx/snapshots \
    --repo-root /path/to/fbsource
```

## Metadata

Each snapshot includes a `metadata.txt` file recording:
- Commit hash
- Commit date
- Commit description
- Snapshot creation timestamp

Example:
```
# RCCLX Snapshot Metadata
commit: abc123def456...
commit_date: 2026-02-01 12:00:00
commit_description: Fix critical bug in rcclx
snapshot_created: 2026-02-01T12:30:00.000000
created_by: create_snapshot.py
```

## Testing

After creating or updating snapshots:

```bash
# Test stable builds for all ROCm versions
buck2 build //comms/rcclx:rcclx-stable --modifier=rocm621
buck2 build //comms/rcclx:rcclx-stable --modifier=rocm64
buck2 build //comms/rcclx:rcclx-stable --modifier=rocm70

# Test last-stable builds
buck2 build //comms/rcclx:rcclx-last-stable --modifier=rocm70

# Run tests with stable snapshot
buck2 test @fbcode//mode/opt-amd-gpu -m rocm70 -m rcclx_stable \
    fbcode//param_bench/train/comms/cpp/rccl-tests/src:
```

## Design Documents

- **[DESIGN_PRE_EXTRACTED_SOURCES.md](DESIGN_PRE_EXTRACTED_SOURCES.md)** - Current design (pre-extracted sources)
- **[DESIGN_BUNDLED_DEPS.md](DESIGN_BUNDLED_DEPS.md)** - Deprecated design (bundled dependencies)

## Benefits

| Benefit | Description |
|---------|-------------|
| **No ABI mismatches** | All code compiles together at build time with current headers |
| **Reuses existing BUCK files** | Snapshot includes rcclx's BUCK, used directly |
| **Same build as rcclx-dev** | `rcclx-stable` is alias to `snapshots/stable/rcclx:rcclx-dev` |
| **Full Buck caching** | Standard Buck compilation, incremental builds work |
| **Simple to maintain** | Update = extract rcclx/ from a commit |
| **Easy to debug** | Standard Buck build, standard tools |
| **No Manifold dependency** | Sources committed to repo |
