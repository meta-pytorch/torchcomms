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

### Snapshots Are Maintained In-Repo

A snapshot is *not* a read-only mirror of its recorded commit. Once created, snapshots
receive compatibility patches directly in the repo (ROCm version bumps, build fixes,
backported upstream patches). This matters for rotation: `stable` → `last-stable` is a
**tree copy**, not a re-extraction of `stable/metadata.txt`'s commit. Re-extracting would
silently discard every patch applied since the snapshot was cut.

Because of this, `metadata.txt` records **two** hashes: `source_commit`, the drop
the sources were originally extracted from, and `fbsource_revision`, the landed
revision the snapshot was last created or rotated at. Rotation carries
`source_commit` across from stable rather than regenerating it.

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

### Rotate Only (Mirror Stable into Last-Stable)

Promotes the current `stable` tree into `last-stable` and stops. Use this when you want
to archive the current stable without cutting a new one yet.

```bash
python3 comms/rcclx/snapshots/scripts/create_snapshot.py \
    --rotate-only \
    --snapshots-root comms/rcclx/snapshots \
    --repo-root /data/users/$USER/fbsource
```

## Command Reference

```
Usage: create_snapshot.py [OPTIONS]

Required:
  --snapshots-root <path>         Path to snapshots directory
  --repo-root <path>              Path to repository root (for sl commands)
  --stage <stable|last-stable>    Snapshot stage to create (not used with --rotate-only)

Optional:
  --commit <hash>                 Commit hash to snapshot from (default: HEAD)
  --rotate                        If creating stable, first copy stable to last-stable
  --rotate-only                   Only mirror stable into last-stable, then exit.
                                  Cannot be combined with --stage/--commit/--rotate.
```

## How It Works

### Creating a snapshot (`--stage`)

1. **Extract Sources**: Uses `sl archive` to extract `fbcode/comms/rcclx/` from the specified commit
2. **Store in Repo**: Sources are committed to `snapshots/{stage}/comms/rcclx/`
3. **Retarget Paths**: Rewrites Buck load paths and the `-I` include path to point inside the snapshot
4. **Write Metadata**: Records commit hash and timestamp in `metadata.txt`
5. **Build at Use Time**: When users build `rcclx-stable`, Buck compiles from the snapshot sources

### Rotating (`--rotate` / `--rotate-only`)

1. **Copy Tree**: `stable/comms/rcclx/` is copied verbatim to `last-stable/comms/rcclx/`,
   preserving symlinks and file modes
2. **Retarget Stage**: Path-like references to `comms/rcclx/snapshots/stable` are rewritten
   to `last-stable` in `BUCK` and `.bzl` files, then verified to leave no stale references.
   Runtime package-name checks such as `elif "snapshots/stable" in pkg:` in `def_build.bzl`
   are deliberately left alone — they must keep matching every stage.
3. **Write Metadata**: Both snapshots record `source_commit` (carried across from
   stable) and `fbsource_revision` (the latest landed ancestor); `last-stable` also
   gets `rotated_from: stable`
4. **Lint**: `arc lint -a` runs over the rotated `BUCK`/`.bzl` files

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
| `create_snapshot.py` | Main script for creating and rotating source snapshots |

## Related Files

- `/comms/rcclx/BUCK` - Build targets including `rcclx-stable` and `rcclx-last-stable` aliases
- `/comms/rcclx/snapshots/README.md` - Snapshot system overview
- `/comms/rcclx/snapshots/stable/metadata.txt` - Current stable snapshot info
- `/comms/rcclx/snapshots/last-stable/metadata.txt` - Previous stable snapshot info
