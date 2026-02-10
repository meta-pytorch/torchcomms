# rcclx Snapshot Scripts

This directory contains scripts for managing rcclx pre-extracted source snapshots. These snapshots allow users to build `rcclx-stable` and `rcclx-last-stable` from committed source code, ensuring ABI compatibility by compiling all code together with current external dependencies.

## Overview

The snapshot system extracts rcclx source code from specific commits and stores it in the repository. At build time, the sources are compiled fresh with whatever external dependencies (folly, scribe, thrift, etc.) are current in fbsource. This eliminates ABI mismatch issues that can occur with precompiled binaries.

### Snapshot Directories

```
snapshots/
├── stable/
│   ├── comms/rcclx/     # Extracted rcclx sources (preserves repo path)
│   └── metadata.txt     # Commit hash, timestamp
└── last-stable/
    ├── comms/rcclx/     # Previous stable snapshot
    └── metadata.txt
```

Note: The `sl archive` command preserves the `comms/rcclx/` path structure from the repository.

## Quick Start

All commands should be run from the `fbcode` directory.

### Create a New Snapshot from Current HEAD

```bash
python3 comms/rcclx/snapshots/scripts/create_snapshot.py \
    --stage stable \
    --snapshots-root comms/rcclx/snapshots \
    --repo-root /data/users/$USER/fbsource
```

### Create a Snapshot from a Specific Commit

```bash
python3 comms/rcclx/snapshots/scripts/create_snapshot.py \
    --stage stable \
    --commit abc123def456 \
    --snapshots-root comms/rcclx/snapshots \
    --repo-root /data/users/$USER/fbsource
```

### Rotate Stable to Last-Stable and Create New Stable

```bash
python3 comms/rcclx/snapshots/scripts/create_snapshot.py \
    --stage stable \
    --commit abc123def456 \
    --rotate \
    --snapshots-root comms/rcclx/snapshots \
    --repo-root /data/users/$USER/fbsource
```

## Command Reference

```
Usage: create_snapshot.py [OPTIONS]

Required:
  --stage <stable|last-stable>    Snapshot stage to create
  --snapshots-root <path>         Path to snapshots directory
  --repo-root <path>              Path to repository root (for sl commands)

Optional:
  --commit <hash>                 Commit hash to snapshot from (default: HEAD)
  --rotate                        If creating stable, first copy stable to last-stable
```

## How It Works

1. **Extract Sources**: Uses `sl archive` to extract `fbcode/comms/rcclx/` from the specified commit
2. **Store in Repo**: Sources are committed to `snapshots/{stage}/rcclx/`
3. **Write Metadata**: Records commit hash and timestamp in `metadata.txt`
4. **Build at Use Time**: When users build `rcclx-stable`, Buck compiles from the snapshot sources

## Benefits

| Feature | Description |
|---------|-------------|
| **ABI Compatible** | All code compiles together with current headers |
| **Same Build Process** | Uses identical BUCK targets as `rcclx-dev` |
| **Full Caching** | Buck caches intermediate outputs normally |
| **Easy to Debug** | Source code is readable in the repo |
| **Simple Updates** | Just run `create_snapshot.py` with new commit |

## Script Files

| Script | Purpose |
|--------|---------|
| `create_snapshot.py` | Main script for creating source snapshots |
| `constants.py` | Configuration constants |

## Related Files

- `/comms/rcclx/BUCK` - Build targets including `rcclx-stable` and `rcclx-last-stable` aliases
- `/comms/rcclx/snapshots/DESIGN_PRE_EXTRACTED_SOURCES.md` - Detailed design documentation
- `/comms/rcclx/snapshots/stable/metadata.txt` - Current stable snapshot info
- `/comms/rcclx/snapshots/last-stable/metadata.txt` - Previous stable snapshot info
