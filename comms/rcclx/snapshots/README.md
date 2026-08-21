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

# Rotate only: mirror stable → last-stable, leaving stable untouched
python3 comms/rcclx/snapshots/scripts/create_snapshot.py \
    --rotate-only \
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
│   ├── comms/rcclx/              # Pre-extracted rcclx sources
│   │   ├── BUCK
│   │   ├── defs.bzl
│   │   ├── rccl_build_config.bzl
│   │   └── develop/
│   │       ├── src/
│   │       ├── meta/
│   │       └── ...
│   └── metadata.txt              # Commit hash, timestamp
├── last-stable/
│   ├── comms/rcclx/              # Pre-extracted rcclx sources
│   └── metadata.txt
├── scripts/
│   ├── create_snapshot.py        # Snapshot creation / rotation script
│   └── README.md                 # Script usage reference
└── README.md                     # This file
```

The `comms/rcclx/` nesting is not redundant: `sl archive` preserves the repository path,
and includes such as `#include "comms/rcclx/develop/meta/lib/CollTraceUtils.h"` resolve
against it via the snapshot's `-I fbcode/comms/rcclx/snapshots/<stage>` flag.

## How It Works

### Build-Time Flow

```
User runs: buck2 build //comms/rcclx:rcclx-stable --modifier=rocm70

Buck2 resolves:
  //comms/rcclx:rcclx-stable
    → alias to //comms/rcclx/snapshots/stable/comms/rcclx:rcclx-dev

Buck2 builds from:
  snapshots/stable/comms/rcclx/BUCK (frozen rcclx source)
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
| `--snapshots-root` | Yes | Path to snapshots directory |
| `--repo-root` | Yes | Path to repository root |
| `--stage` | Unless `--rotate-only` | Snapshot stage: `stable` or `last-stable` |
| `--commit` | No | Commit hash to snapshot from (default: current HEAD) |
| `--rotate` | No | If creating stable, first copy current stable to last-stable |
| `--rotate-only` | No | Only mirror stable into last-stable, then exit. Cannot be combined with `--stage`/`--commit`/`--rotate` |

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

# Rotate only
python3 create_snapshot.py \
    --rotate-only \
    --snapshots-root comms/rcclx/snapshots \
    --repo-root /path/to/fbsource
```

### Rotation Semantics

Rotation **copies the `stable` tree** into `last-stable` rather than re-extracting from
`stable/metadata.txt`'s commit. Snapshots accumulate in-repo patches after they are cut
(ROCm bumps, build fixes, backported upstream changes), so re-extraction would silently
discard them.

After the copy, stage-specific Buck load paths and the `-I` include path are retargeted
from `stable` to `last-stable`, and the script verifies no stale `stable` references
remain. A correct rotation leaves the two trees byte-identical except for those path
references.

## Metadata

Each snapshot includes a `metadata.txt` file recording two hashes:

| Field | Meaning |
|-------|---------|
| `source_commit` | The commit these sources were originally extracted from |
| `fbsource_revision` | The landed fbsource revision the snapshot was last created or rotated at |
| `snapshot_created` | Timestamp of the last create/rotate |
| `rotated_from` | Present on rotated snapshots only |

Both hashes are needed because a snapshot is not a frozen copy of one commit.
Patches land on the tree after it is cut, so `source_commit` alone understates the
contents, while `fbsource_revision` alone loses the drop it came from. Run `sl log`
on the snapshot directory to see the patches applied in between.

`fbsource_revision` is the latest *landed* ancestor rather than the working copy
revision, which is usually a draft commit whose hash is rewritten on land.

Rotation carries `source_commit` across from stable rather than regenerating it,
so the record of which drop the sources came from survives.

Example:
```
# RCCLX Snapshot Metadata
source_commit: 6f840ba5d808af9407979bca2132790a72898887
source_commit_date: 2026-01-30 10:51 -0800
source_commit_description: DDA nranks check patch

fbsource_revision: 34f53cfe1715f09b9c6750ed51e38e49eb204dfc
fbsource_revision_date: 2026-08-20 11:14 -0700

snapshot_created: 2026-08-20T12:29:49.732133
created_by: create_snapshot.py
rotated_from: stable
```

## Testing

After creating or updating snapshots:

```bash
# Test stable builds for all ROCm versions
buck2 build //comms/rcclx:rcclx-stable --modifier=rocm70

# Test last-stable builds
buck2 build //comms/rcclx:rcclx-last-stable --modifier=rocm70

# Run tests with stable snapshot
buck2 test @fbcode//mode/opt-amd-gpu -m rocm70 -m rcclx_stable \
    fbcode//param_bench/train/comms/cpp/rccl-tests/src:
```

## Benefits

| Benefit | Description |
|---------|-------------|
| **No ABI mismatches** | All code compiles together at build time with current headers |
| **Reuses existing BUCK files** | Snapshot includes rcclx's BUCK, used directly |
| **Same build as rcclx-dev** | `rcclx-stable` is alias to `snapshots/stable/comms/rcclx:rcclx-dev` |
| **Full Buck caching** | Standard Buck compilation, incremental builds work |
| **Simple to maintain** | Update = extract comms/rcclx/ from a commit |
| **Easy to debug** | Standard Buck build, standard tools |
| **No Manifold dependency** | Sources committed to repo |
