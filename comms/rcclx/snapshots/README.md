# RCCLX Snapshots

This directory contains the infrastructure for managing snapshots of rcclx static library builds across different ROCm versions.

## Storage Architecture

The snapshot system uses a **dual-storage approach** to optimize for both size and accessibility:

- **Artifacts (large `.a` files)**: Stored in Manifold bucket `rcclx_prebuilt_artifacts`
- **Metadata (small text files)**: Stored in the repository for easy access and version control

### Manifold Structure (Artifacts)

```
rcclx_prebuilt_artifacts/tree/
├── stable/
│   ├── 6.2/librcclx-dev.a
│   ├── 6.4/librcclx-dev.a
│   └── 7.0/librcclx-dev.a
├── last_stable/
│   ├── 6.2/librcclx-dev.a
│   ├── 6.4/librcclx-dev.a
│   └── 7.0/librcclx-dev.a
└── archives/
    ├── 6.2/
    │   ├── 20250107_143000_librcclx-dev.a
    │   ├── 20250108_091500_librcclx-dev.a
    │   └── ...
    ├── 6.4/
    │   ├── 20250107_143000_librcclx-dev.a
    │   └── ...
    └── 7.0/
        ├── 20250107_143000_librcclx-dev.a
        └── ...
```

### Repository Structure (Metadata)

```
snapshots/
├── scripts/           # Automation scripts
├── stable/           # Current stable metadata
│   ├── 6.2/metadata.txt
│   ├── 6.4/metadata.txt
│   └── 7.0/metadata.txt
├── last-stable/      # Previous stable metadata
│   ├── 6.2/metadata.txt
│   ├── 6.4/metadata.txt
│   └── 7.0/metadata.txt
└── archives/         # Historical metadata with timestamp-prefixed files
    ├── 6.2/
    │   ├── 20250107_143000_metadata.txt
    │   ├── 20250108_091500_metadata.txt
    │   └── ...
    ├── 6.4/
    │   ├── 20250107_143000_metadata.txt
    │   └── ...
    └── 7.0/
        ├── 20250107_143000_metadata.txt
        └── ...
```

## Overview

The snapshot system allows you to:
1. Build rcclx with specific ROCm versions
2. Create tagged stable snapshots of the built archives (stored in Manifold)
3. Maintain version history with automatic rotation (stable → last-stable → archives)
4. Track metadata including commit hashes and dependencies (stored in repo)
5. Use pre-built stable snapshots for faster builds and validated releases

### Build Targets

There are two main build targets for rcclx:

- **`rcclx-dev`**: Builds rcclx from source (current top-of-tree)
  - Use when you're developing new features or testing changes
  - Always builds from the latest source code
  - Takes longer to build as it compiles everything from scratch

- **`rcclx-stable`**: Uses pre-built stable snapshots
  - Use for production deployments or when you need validated builds
  - Much faster as it skips compilation
  - Automatically selects the correct snapshot based on ROCm version constraint
  - Requires that a stable snapshot exists for the specified ROCm version

## Usage

### Using Stable Snapshots (rcclx-stable)

The `rcclx-stable` target allows you to use pre-built stable snapshots instead of building from source. This is faster and ensures you're using validated builds. The artifacts are automatically fetched from Manifold during the build:

```bash
# Use stable snapshot for ROCm 6.2
buck2 build @fbcode//mode/opt-amd-gpu -m rcclx_dev -m ovr_config//third-party/rocm/constraints:6.2.1 fbcode//comms/rcclx:rcclx-stable

# Use stable snapshot for ROCm 6.4
buck2 build @fbcode//mode/opt-amd-gpu -m rcclx_dev -m ovr_config//third-party/rocm/constraints:6.4.2 fbcode//comms/rcclx:rcclx-stable

# Use stable snapshot for ROCm 7.0
buck2 build @fbcode//mode/opt-amd-gpu -m rcclx_dev -m ovr_config//third-party/rocm/constraints:7.0 fbcode//comms/rcclx:rcclx-stable
```

The build system automatically:
1. Selects the correct stable snapshot based on the ROCm version constraint
2. Downloads the artifact from Manifold at `rcclx_prebuilt_artifacts/tree/stable/<version>/librcclx-dev.a`
3. Uses the downloaded artifact for linking

If a stable snapshot doesn't exist in Manifold for the specified version, the build will fail with a clear error message.

### Creating a Snapshot

To create a new snapshot for a specific ROCm version, use the wrapper script from the fbcode directory:

```bash
# From fbcode directory
cd /path/to/fbsource/fbcode

# For ROCm 6.2
./comms/rcclx/snapshots/scripts/create_snapshot_wrapper.sh --rocm-version 6.2

# For ROCm 6.4
./comms/rcclx/snapshots/scripts/create_snapshot_wrapper.sh --rocm-version 6.4

# For ROCm 7.0
./comms/rcclx/snapshots/scripts/create_snapshot_wrapper.sh --rocm-version 7.0
```

The wrapper script will:
1. Build the rcclx-dev library with the specified ROCm version
2. Run the snapshot creation script to rotate archives and create metadata
3. Upload the artifact to Manifold at `rcclx_prebuilt_artifacts/tree/stable/<rocm_version>/`
4. Save metadata to the repo at `comms/rcclx/snapshots/stable/<rocm_version>/metadata.txt`

### What Happens During Snapshot Creation

When you run a snapshot build, the following operations occur in order:

1. **Build**: The rcclx library is built with the specified ROCm version constraints

2. **Archive Previous (if last-stable exists)**:
   - **Metadata**: Moved from `last-stable/<version>/` to `archives/<version>/<timestamp>/` in repo
   - **Artifact**: Copied from Manifold `last_stable/<version>/` to `archives/<version>/<timestamp>/` in Manifold

3. **Rotate Current (if stable exists)**:
   - **Metadata**: Moved from `stable/<version>/` to `last-stable/<version>/` in repo
   - **Artifact**: Copied from Manifold `stable/<version>/` to `last_stable/<version>/` in Manifold

4. **Calculate Checksums**: SHA1 and SHA256 checksums are calculated for the newly built library

5. **Install New**:
   - **Artifact**: Newly built `librcclx-dev.a` uploaded to Manifold at `stable/<version>/`
   - **Metadata**: Created in repo at `stable/<version>/metadata.txt`

6. **Generate Metadata**: A `metadata.txt` file is created in the repo containing:
   - Snapshot creation timestamp
   - ROCm version
   - SHA1 and SHA256 checksums of the artifact
   - Commit hashes for rcclx and fbsource repositories

### Getting Checksums for BUCK File

The snapshot creation workflow automatically generates checksums and updates the BUCK file configuration. No manual intervention is required!

When you create a snapshot using the wrapper script:

```bash
cd /path/to/fbsource/fbcode
./comms/rcclx/snapshots/scripts/create_snapshot_wrapper.sh --rocm-version 6.4
```

The script automatically:
1. Builds the rcclx library for the specified ROCm version
2. Calculates SHA256 checksums for the built library
3. Stores checksums in `snapshots/stable/<version>/metadata.txt`
4. Uploads the artifact to Manifold
5. **Generates `comms/rcclx/stable_checksums.bzl`** from all metadata files

The BUCK file loads checksums from `stable_checksums.bzl`:

```python
load(":stable_checksums.bzl", "RCCLX_STABLE_CHECKSUMS")

manifold_get(
    name = "rcclx-stable-6.4-artifact",
    ...
    sha256 = RCCLX_STABLE_CHECKSUMS.get("6.4", "0" * 64),
    ...
)
```

This approach eliminates manual checksum management - the checksums are automatically read from metadata files and applied to BUCK targets.

**Manual Checksum Regeneration (if needed)**

If you need to regenerate `stable_checksums.bzl` manually:

```bash
buck2 run fbcode//comms/rcclx/snapshots/scripts:generate_checksums_bzl -- \
    --snapshots-root "$(pwd)/comms/rcclx/snapshots" \
    --output "$(pwd)/comms/rcclx/stable_checksums.bzl"
```

### Accessing Artifacts from Manifold

To download a pre-built artifact from Manifold, you can use the Manifold API or command-line tools:

```python
from manifold.clients.python import ManifoldClient
from datetime import timedelta
import io

# Download stable artifact for ROCm 6.4
async def download_artifact():
    with ManifoldClient(bucket="rcclx_prebuilt_artifacts", apiKey="rcclx_prebuilt_artifacts-key") as client:
        buffer = io.BytesIO()
        await client.get(
            path="tree/stable/6.4/librcclx-dev.a",
            output=buffer,
            timeout=timedelta(seconds=600),
        )

        # Save to local file
        with open("librcclx-dev.a", "wb") as f:
            f.write(buffer.getvalue())
```

### Snapshot Metadata

Each stable snapshot includes a `metadata.txt` file with information about the build:

```
Snapshot created: 2025-01-15 14:30:00 UTC
ROCm version: 6.4

Commit Hashes:
  fbsource: abc123def456...
  rcclx: 789ghi012jkl...
```

This metadata allows you to trace each snapshot back to its exact source code state.

## Directory Purposes

- **scripts/**: Contains automation scripts used by the build system
  - `create_snapshot.py`: Main script that handles snapshot creation and rotation

- **stable/**: Contains the current stable snapshot for each ROCm version
  - This is where you should find the latest validated builds
  - Each subdirectory contains the library archive and metadata

- **last-stable/**: Contains the previous stable snapshot for each ROCm version
  - Provides a quick rollback option if issues are found with the current stable
  - Automatically populated when creating a new stable snapshot

- **archives/**: Contains historical snapshots organized by ROCm version and timestamp
  - Format: `archives/<rocm_version>/<YYYYMMDD_HHMMSS>/`
  - Provides long-term version history
  - Automatically populated when rotating snapshots

## Example Workflow

1. Developer makes changes to rcclx
2. Changes are tested and validated
3. Run snapshot build for ROCm 6.4:
   ```bash
   cd /path/to/fbsource/fbcode
   ./comms/rcclx/snapshots/scripts/create_snapshot_wrapper.sh --rocm-version 6.4
   ```
4. The snapshot system:
   - Archives the old last-stable to `archives/6.4/<timestamp>/` (metadata in repo, artifact in Manifold)
   - Moves current stable to `last-stable/6.4/` (metadata in repo, artifact in Manifold)
   - Installs new build to `stable/6.4/` in Manifold
   - Creates metadata file with commit hashes in repo

## Notes

- **Artifacts** (.a files) are stored in Manifold to avoid bloating the repository with large binary files
- **Metadata** (metadata.txt) is stored in the repository for easy access and version control
- Each ROCm version maintains its own independent snapshot history
- The snapshot system is designed to work with the existing rcclx-dev build target
- All snapshot operations are atomic and safe (creates backups before modifications)
- Timestamp format `YYYYMMDD_HHMMSS` ensures consistent naming across repo and Manifold archives
