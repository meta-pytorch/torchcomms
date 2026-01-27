#!/usr/bin/env python3
# pyre-strict
"""
Script to create snapshots of rcclx static libraries and headers.

This script handles:
1. Creating a backup of all current artifacts in Manifold (checkpoint for recovery)
2. Moving last-stable to archives with timestamp (metadata and headers in repo, artifacts in Manifold)
3. Moving stable to last-stable (metadata and headers in repo, artifacts in Manifold)
4. Copying newly built library to stable (metadata and headers in repo, artifacts in Manifold)
5. Creating metadata file with commit hashes

Artifacts (.a files) are stored in Manifold at:
  rcclx_prebuilt_artifacts/tree/stable/{rocm_version}/librcclx-dev.a
  rcclx_prebuilt_artifacts/tree/last_stable/{rocm_version}/librcclx-dev.a
  rcclx_prebuilt_artifacts/tree/archives/{rocm_version}/{timestamp}_librcclx-dev.a

Metadata and header files are stored in the repo at:
  snapshots/stable/{rocm_version}/metadata.txt
  snapshots/stable/{rocm_version}/rccl.h
  snapshots/last-stable/{rocm_version}/metadata.txt
  snapshots/last-stable/{rocm_version}/rccl.h
  snapshots/archives/{rocm_version}/{timestamp}_metadata.txt
  snapshots/archives/{rocm_version}/{timestamp}_rccl.h
"""

import argparse
import hashlib
import io
import logging
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from ai_infra.privacy_security.security.file_vault.meta.manifold_client.python import (
    BarnFSManifoldClient,
)
from libfb.py.asyncio.await_utils import await_sync

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)

MANIFOLD_BUCKET: str = "rcclx_prebuilt_artifacts"
MANIFOLD_PATH_PREFIX: str = "tree/"
MANIFOLD_TIMEOUT_SEC: int = 600  # 10 minutes
MANIFOLD_NUM_RETRIES: int = 10

# Default ROCm versions for backup
DEFAULT_ROCM_VERSIONS: List[str] = ["6.2", "6.4", "7.0"]

# Stages to backup
BACKUP_STAGES: List[str] = ["stable", "last_stable"]

# Library filename
LIBRARY_NAME: str = "librcclx-dev.a"


def get_backup_path(
    backup_name: str, stage: str, rocm_version: str, filename: str = ""
) -> str:
    """
    Construct a Manifold backup path for artifacts.

    Args:
        backup_name: Name of the backup directory (e.g., 'backup_20260127_012000')
        stage: One of 'stable', 'last_stable'
        rocm_version: ROCm version (e.g., '6.2')
        filename: Optional filename to append

    Returns:
        Full Manifold backup path (without bucket prefix)
    """
    if filename:
        return f"{MANIFOLD_PATH_PREFIX}archives/{backup_name}/{stage}/{rocm_version}/{filename}"
    return f"{MANIFOLD_PATH_PREFIX}archives/{backup_name}/{stage}/{rocm_version}"


def generate_backup_name() -> str:
    """
    Generate a backup directory name with timestamp.

    Returns:
        Backup name in format 'backup_YYYYMMDD_HHMMSS'
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"backup_{timestamp}"


def create_manifold_directory_for_backup(dir_path: str) -> bool:
    """
    Create a directory in Manifold (tree namespace) for backup.

    Args:
        dir_path: Directory path to create (without bucket prefix)

    Returns:
        True if successful or already exists
    """

    async def async_mkdir() -> bool:
        client = BarnFSManifoldClient.get_client(
            bucket=MANIFOLD_BUCKET, apiKey=f"{MANIFOLD_BUCKET}-key"
        )
        with client:
            await client.mkdir(path=dir_path)
            return True

    return await_sync(async_mkdir())


def copy_manifold_file_for_backup(src_path: str, dst_path: str) -> bool:
    """
    Copy a file within Manifold for backup purposes.

    Args:
        src_path: Source Manifold path (without bucket prefix)
        dst_path: Destination Manifold path (without bucket prefix)

    Returns:
        True if successful, False if source doesn't exist
    """

    async def async_copy() -> bool:
        client = BarnFSManifoldClient.get_client(
            bucket=MANIFOLD_BUCKET, apiKey=f"{MANIFOLD_BUCKET}-key"
        )
        with client:
            # Check if source exists
            if not await client.exists(path=src_path):
                return False

            # Download from source
            buffer = io.BytesIO()
            await client.get(
                path=src_path,
                output=buffer,
                timeout=timedelta(seconds=MANIFOLD_TIMEOUT_SEC),
                numRetries=MANIFOLD_NUM_RETRIES,
            )

            # Upload to destination
            buffer.seek(0)
            await client.put(
                path=dst_path,
                input=buffer,
                timeout=timedelta(seconds=MANIFOLD_TIMEOUT_SEC),
                numRetries=MANIFOLD_NUM_RETRIES,
            )

            return True

    return await_sync(async_copy())


def backup_all_artifacts(rocm_versions: List[str]) -> str:
    """
    Create a backup of all stable and last-stable artifacts in Manifold.

    This creates a checkpoint that can be used to restore artifacts if
    something goes wrong during snapshot creation.

    Args:
        rocm_versions: List of ROCm versions to backup

    Returns:
        The backup path name that was created
    """
    backup_name = generate_backup_name()
    backup_base_path = f"{MANIFOLD_PATH_PREFIX}archives/{backup_name}"

    logger.info("=" * 60)
    logger.info("Creating backup checkpoint before snapshot creation")
    logger.info("=" * 60)
    logger.info(f"Backup name: {backup_name}")
    logger.info(f"Backup path: {MANIFOLD_BUCKET}/{backup_base_path}/")
    logger.info(f"Stages to backup: {', '.join(BACKUP_STAGES)}")
    logger.info(f"ROCm versions: {', '.join(rocm_versions)}")
    logger.info("")

    # Create the directory structure in Manifold before copying files
    logger.info("Creating backup directory structure in Manifold...")
    try:
        # Create the backup root directory
        create_manifold_directory_for_backup(backup_base_path)
        logger.info(f"  -> Created: {backup_base_path}")

        # Create stage and version directories
        for stage in BACKUP_STAGES:
            stage_path = f"{backup_base_path}/{stage}"
            create_manifold_directory_for_backup(stage_path)
            logger.info(f"  -> Created: {stage_path}")

            for rocm_version in rocm_versions:
                version_path = f"{stage_path}/{rocm_version}"
                create_manifold_directory_for_backup(version_path)
                logger.info(f"  -> Created: {version_path}")
    except Exception as e:
        logger.error(f"Error creating backup directory structure: {e}")
        raise

    logger.info("")

    total_copied = 0
    total_missing = 0

    for stage in BACKUP_STAGES:
        for rocm_version in rocm_versions:
            src_path = get_manifold_path(stage, rocm_version, LIBRARY_NAME)
            dst_path = get_backup_path(backup_name, stage, rocm_version, LIBRARY_NAME)

            logger.info(f"Backing up {stage}/{rocm_version}/{LIBRARY_NAME}...")

            try:
                if copy_manifold_file_for_backup(src_path, dst_path):
                    logger.info(f"  -> Copied to {dst_path}")
                    total_copied += 1
                else:
                    logger.warning(f"  -> Source not found (skipping): {src_path}")
                    total_missing += 1
            except Exception as e:
                logger.error(f"  -> Error copying: {e}")
                total_missing += 1

    logger.info("")
    logger.info(f"Backup complete: {total_copied} copied, {total_missing} missing/skipped")
    logger.info(f"Backup checkpoint: {backup_name}")
    logger.info("")
    logger.info("To restore from this backup if needed, run:")
    logger.info(
        f"  buck2 run fbcode//comms/rcclx/snapshots/scripts:restore_manifold_artifacts -- \\"
    )
    logger.info(f"      --backup-name {backup_name}")
    logger.info("=" * 60)
    logger.info("")

    return backup_name


def get_git_commit_hash(repo_path: str) -> Optional[str]:
    """Get the current git commit hash for a repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to get git commit hash for {repo_path}: {e}")
        return None


def get_hg_commit_hash(repo_path: str) -> Optional[str]:
    """Get the current hg commit hash for a repository."""
    try:
        result = subprocess.run(
            ["hg", "log", "-r", ".", "-T", "{node}"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to get hg commit hash for {repo_path}: {e}")
        return None


def get_dependency_info(rcclx_repo_path: str) -> Dict[str, str]:
    """
    Get commit hashes for rcclx and its direct dependencies.

    Args:
        rcclx_repo_path: Path to the rcclx repository

    Returns:
        Dictionary mapping repo names to commit hashes
    """
    deps_info = {}

    # Get rcclx commit (using hg since this is fbsource)
    rcclx_hash = get_hg_commit_hash(rcclx_repo_path)
    if rcclx_hash:
        deps_info["rcclx"] = rcclx_hash

    # Get fbsource root commit
    fbsource_root = Path(rcclx_repo_path)
    while fbsource_root.parent != fbsource_root:
        if (fbsource_root / ".hg").exists():
            break
        fbsource_root = fbsource_root.parent

    fbsource_hash = get_hg_commit_hash(str(fbsource_root))
    if fbsource_hash:
        deps_info["fbsource"] = fbsource_hash

    return deps_info


def calculate_file_checksums(file_path: Path) -> Dict[str, str]:
    """
    Calculate SHA1 and SHA256 checksums for a file.

    Args:
        file_path: Path to the file to checksum

    Returns:
        Dictionary with 'sha1' and 'sha256' keys
    """
    sha1_hash = hashlib.sha1()
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(4096 * 1024), b""):
            sha1_hash.update(chunk)
            sha256_hash.update(chunk)

    return {
        "sha1": sha1_hash.hexdigest(),
        "sha256": sha256_hash.hexdigest(),
    }


def create_metadata_file(
    metadata_path: Path,
    rocm_version: str,
    deps_info: Dict[str, str],
    checksums: Dict[str, str],
) -> None:
    """
    Create metadata file with commit hashes, checksums, and build information.

    Args:
        metadata_path: Path where metadata.txt should be created
        rocm_version: ROCm version used for the build
        deps_info: Dictionary of dependency commit hashes
        checksums: Dictionary with 'sha1' and 'sha256' checksums
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    with open(metadata_path, "w") as f:
        f.write(f"Snapshot created: {timestamp}\n")
        f.write(f"ROCm version: {rocm_version}\n")
        f.write("\n")
        f.write("Checksums:\n")
        f.write(f"  sha1: {checksums['sha1']}\n")
        f.write(f"  sha256: {checksums['sha256']}\n")
        f.write("\n")
        f.write("Commit Hashes:\n")
        for repo_name, commit_hash in sorted(deps_info.items()):
            f.write(f"  {repo_name}: {commit_hash}\n")

    logger.info(f"Created metadata file: {metadata_path}")


def upload_file_to_manifold(local_path: Path, manifold_path: str) -> None:
    """
    Upload a file to Manifold.

    Args:
        local_path: Local file path to upload
        manifold_path: Manifold path (without bucket prefix)
    """
    logger.info(f"Uploading {local_path.name} to Manifold at {manifold_path}")

    async def async_upload() -> None:
        with open(local_path, "rb") as f:
            file_data = f.read()

        client = BarnFSManifoldClient.get_client(
            bucket=MANIFOLD_BUCKET, apiKey=f"{MANIFOLD_BUCKET}-key"
        )
        with client:
            await client.put(
                path=manifold_path,
                input=io.BytesIO(file_data),
                timeout=timedelta(seconds=MANIFOLD_TIMEOUT_SEC),
                numRetries=MANIFOLD_NUM_RETRIES,
            )

    await_sync(async_upload())
    logger.info(f"Successfully uploaded {local_path.name} to Manifold")


def download_file_from_manifold(manifold_path: str, local_path: Path) -> None:
    """
    Download a file from Manifold.

    Args:
        manifold_path: Manifold path (without bucket prefix)
        local_path: Local file path to save to
    """
    logger.info(f"Downloading from Manifold {manifold_path} to {local_path}")

    async def async_download() -> None:
        client = BarnFSManifoldClient.get_client(
            bucket=MANIFOLD_BUCKET, apiKey=f"{MANIFOLD_BUCKET}-key"
        )
        with client:
            buffer = io.BytesIO()
            await client.get(
                path=manifold_path,
                output=buffer,
                timeout=timedelta(seconds=MANIFOLD_TIMEOUT_SEC),
                numRetries=MANIFOLD_NUM_RETRIES,
            )

            # Create parent directory if needed
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to file
            with open(local_path, "wb") as f:
                f.write(buffer.getvalue())

    await_sync(async_download())
    logger.info(f"Successfully downloaded to {local_path}")


def check_manifold_path_exists(manifold_path: str) -> bool:
    """
    Check if a path exists in Manifold.

    Args:
        manifold_path: Manifold path (without bucket prefix)

    Returns:
        True if path exists, False otherwise
    """

    async def async_exists() -> bool:
        client = BarnFSManifoldClient.get_client(
            bucket=MANIFOLD_BUCKET, apiKey=f"{MANIFOLD_BUCKET}-key"
        )
        with client:
            return await client.exists(path=manifold_path)

    return await_sync(async_exists())


def copy_manifold_file(src_path: str, dst_path: str) -> None:
    """
    Copy a file within Manifold by downloading and re-uploading.
    Deletes destination first if it exists to avoid conflicts.

    Args:
        src_path: Source Manifold path (without bucket prefix)
        dst_path: Destination Manifold path (without bucket prefix)
    """
    logger.info(f"Copying in Manifold from {src_path} to {dst_path}")

    async def async_copy() -> None:
        client = BarnFSManifoldClient.get_client(
            bucket=MANIFOLD_BUCKET, apiKey=f"{MANIFOLD_BUCKET}-key"
        )
        with client:
            # Delete destination first if it exists
            if await client.exists(path=dst_path):
                logger.info(f"Destination {dst_path} exists, deleting first")
                await client.rm(path=dst_path)

            # Download from source
            buffer = io.BytesIO()
            await client.get(
                path=src_path,
                output=buffer,
                timeout=timedelta(seconds=MANIFOLD_TIMEOUT_SEC),
                numRetries=MANIFOLD_NUM_RETRIES,
            )

            # Upload to destination
            buffer.seek(0)
            await client.put(
                path=dst_path,
                input=buffer,
                timeout=timedelta(seconds=MANIFOLD_TIMEOUT_SEC),
                numRetries=MANIFOLD_NUM_RETRIES,
            )

    await_sync(async_copy())
    logger.info("Successfully copied in Manifold")


def delete_manifold_file(manifold_path: str) -> None:
    """
    Delete a file from Manifold.

    Args:
        manifold_path: Manifold path (without bucket prefix)
    """
    logger.info(f"Deleting from Manifold: {manifold_path}")

    async def async_delete() -> None:
        client = BarnFSManifoldClient.get_client(
            bucket=MANIFOLD_BUCKET, apiKey=f"{MANIFOLD_BUCKET}-key"
        )
        with client:
            await client.rm(path=manifold_path)

    await_sync(async_delete())
    logger.info("Successfully deleted from Manifold")


def move_directory_contents(src_dir: Path, dst_dir: Path) -> None:
    """
    Move all contents from src_dir to dst_dir (metadata only, no artifacts).

    Args:
        src_dir: Source directory
        dst_dir: Destination directory
    """
    if not src_dir.exists():
        return

    # Create destination directory if it doesn't exist
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Move all files and directories
    for item in src_dir.iterdir():
        dst_item = dst_dir / item.name
        if dst_item.exists():
            if dst_item.is_dir():
                shutil.rmtree(dst_item)
            else:
                dst_item.unlink()
        shutil.move(str(item), str(dst_item))

    logger.info(f"Moved contents from {src_dir} to {dst_dir}")


def is_directory_empty(directory: Path) -> bool:
    """Check if a directory is empty (ignoring hidden files)."""
    if not directory.exists():
        return True
    return not any(directory.iterdir())


def get_manifold_path(stage: str, rocm_version: str, filename: str = "") -> str:
    """
    Construct a Manifold path for artifacts.

    Args:
        stage: One of 'stable', 'last_stable', 'archives'
        rocm_version: ROCm version (e.g., '6.2')
        filename: Optional filename to append

    Returns:
        Full Manifold path (without bucket prefix)
    """
    if filename:
        return f"{MANIFOLD_PATH_PREFIX}{stage}/{rocm_version}/{filename}"
    return f"{MANIFOLD_PATH_PREFIX}{stage}/{rocm_version}"


def create_snapshot(
    library_path: Path,
    rocm_version: str,
    snapshots_root: Path,
    rcclx_repo_path: Path,
    header_path: Path,
    skip_backup: bool = False,
    only_stable: bool = False,
) -> None:
    """
    Create a snapshot of the rcclx library and header.

    This function handles the rotation of snapshots and saves artifacts to Manifold
    while keeping metadata and headers in the repo.

    Args:
        library_path: Path to the built library file (e.g., librcclx-dev.a)
        rocm_version: ROCm version (e.g., "6.2", "6.4", "7.0")
        snapshots_root: Root path of the snapshots directory
        rcclx_repo_path: Path to the rcclx repository for getting commit info
        header_path: Path to the rccl.h header file to snapshot
        skip_backup: If True, skip the backup step (not recommended)
        only_stable: If True, only update stable without rotating to last-stable
    """
    if only_stable:
        logger.info(f"Creating snapshot for ROCm {rocm_version} (only-stable mode)")
    else:
        logger.info(f"Creating snapshot for ROCm {rocm_version}")

    # Validate inputs
    if not library_path.exists():
        raise FileNotFoundError(f"Library file not found: {library_path}")

    if not header_path.exists():
        raise FileNotFoundError(f"Header file not found: {header_path}")

    if not snapshots_root.exists():
        raise FileNotFoundError(f"Snapshots directory not found: {snapshots_root}")

    # Step 0: Create a backup of all artifacts before making any changes
    if not skip_backup:
        backup_name = backup_all_artifacts(DEFAULT_ROCM_VERSIONS)
        logger.info(f"Backup checkpoint created: {backup_name}")
    else:
        logger.warning("Skipping backup step (--skip-backup flag was set)")

    # Define directories for metadata (in repo)
    stable_dir = snapshots_root / "stable" / rocm_version
    last_stable_dir = snapshots_root / "last-stable" / rocm_version

    # Create directories if they don't exist
    stable_dir.mkdir(parents=True, exist_ok=True)
    if not only_stable:
        last_stable_dir.mkdir(parents=True, exist_ok=True)

    library_name = library_path.name
    header_name = header_path.name

    # Step 1: Move stable to last-stable (unless only_stable mode)
    if not only_stable:
        # The backup already preserves the old last-stable, so we can safely overwrite
        if not is_directory_empty(stable_dir):
            logger.info(f"Moving stable to last-stable/{rocm_version}/")

            # Move metadata in repo (overwrites existing last-stable)
            move_directory_contents(stable_dir, last_stable_dir)

            # Move artifact in Manifold (if exists)
            stable_manifold_path = get_manifold_path("stable", rocm_version, library_name)
            if check_manifold_path_exists(stable_manifold_path):
                last_stable_manifold_path = get_manifold_path(
                    "last_stable", rocm_version, library_name
                )
                copy_manifold_file(stable_manifold_path, last_stable_manifold_path)
                delete_manifold_file(stable_manifold_path)
                logger.info(
                    f"Moved artifact in Manifold to last_stable/{rocm_version}/{library_name}"
                )
    else:
        logger.info("Skipping stable-to-last-stable rotation (--only-stable mode)")
        # Clear existing stable metadata files before creating new ones
        if not is_directory_empty(stable_dir):
            for item in stable_dir.iterdir():
                if item.is_file():
                    item.unlink()

    # Step 2: Calculate checksums for the library before uploading
    logger.info(f"Calculating checksums for {library_path}")
    checksums = calculate_file_checksums(library_path)
    logger.info(f"SHA1: {checksums['sha1']}")
    logger.info(f"SHA256: {checksums['sha256']}")

    # Step 3: Upload newly built library to Manifold stable
    stable_manifold_path = get_manifold_path("stable", rocm_version, library_name)
    upload_file_to_manifold(library_path, stable_manifold_path)

    # Step 4: Copy header file to repo stable
    stable_header_path = stable_dir / header_name
    shutil.copy2(header_path, stable_header_path)
    logger.info(f"Copied {header_name} to {stable_header_path}")

    # Step 5: Create metadata file in repo stable with checksums
    deps_info = get_dependency_info(str(rcclx_repo_path))
    metadata_path = stable_dir / "metadata.txt"
    create_metadata_file(metadata_path, rocm_version, deps_info, checksums)

    logger.info(f"Snapshot creation completed successfully for ROCm {rocm_version}")
    logger.info(
        f"Artifact stored in Manifold at: {MANIFOLD_BUCKET}/{stable_manifold_path}"
    )
    logger.info(f"Header stored in repo at: {stable_header_path}")
    logger.info(f"Metadata stored in repo at: {metadata_path}")
    if only_stable:
        logger.info("Last-stable was not modified (--only-stable mode)")


def copy_stable_to_last_stable(
    rocm_version: str,
    snapshots_root: Path,
    skip_backup: bool = False,
) -> None:
    """
    Copy current stable snapshot to last-stable.

    This function rotates snapshots without creating a new stable:
    1. Creates a backup checkpoint (unless skipped)
    2. Copies current stable to last-stable (overwrites existing last-stable)
    3. Stable remains unchanged

    Args:
        rocm_version: ROCm version (e.g., "6.2", "6.4", "7.0")
        snapshots_root: Root path of the snapshots directory
        skip_backup: If True, skip the backup step (not recommended)
    """
    logger.info(f"Copying stable to last-stable for ROCm {rocm_version}")

    if not snapshots_root.exists():
        raise FileNotFoundError(f"Snapshots directory not found: {snapshots_root}")

    # Step 0: Create a backup of all artifacts before making any changes
    if not skip_backup:
        backup_name = backup_all_artifacts(DEFAULT_ROCM_VERSIONS)
        logger.info(f"Backup checkpoint created: {backup_name}")
    else:
        logger.warning("Skipping backup step (--skip-backup flag was set)")

    # Define directories for metadata (in repo)
    stable_dir = snapshots_root / "stable" / rocm_version
    last_stable_dir = snapshots_root / "last-stable" / rocm_version

    # Validate that stable exists
    if is_directory_empty(stable_dir):
        raise ValueError(
            f"Stable directory is empty for ROCm {rocm_version}. "
            "Cannot copy stable to last-stable."
        )

    # Check if stable artifact exists in Manifold
    stable_manifold_path = get_manifold_path("stable", rocm_version, LIBRARY_NAME)
    if not check_manifold_path_exists(stable_manifold_path):
        raise ValueError(
            f"Stable artifact not found in Manifold for ROCm {rocm_version}. "
            "Cannot copy stable to last-stable."
        )

    # Create last-stable directory if it doesn't exist
    last_stable_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Copy stable to last-stable (overwrites existing last-stable)
    # The backup already preserves the old last-stable, so we can safely overwrite
    logger.info(f"Copying stable to last-stable/{rocm_version}/")

    # Clear existing last-stable files and copy from stable
    for item in last_stable_dir.iterdir():
        if item.is_file():
            item.unlink()

    # Copy metadata in repo
    for item in stable_dir.iterdir():
        if item.is_file():
            dst_item = last_stable_dir / item.name
            shutil.copy2(str(item), str(dst_item))
            logger.info(f"Copied {item.name} to {dst_item}")

    # Copy artifact in Manifold (overwrites existing)
    last_stable_manifold_path = get_manifold_path(
        "last_stable", rocm_version, LIBRARY_NAME
    )
    copy_manifold_file(stable_manifold_path, last_stable_manifold_path)
    logger.info(
        f"Copied artifact in Manifold to last_stable/{rocm_version}/{LIBRARY_NAME}"
    )

    logger.info(
        f"Successfully copied stable to last-stable for ROCm {rocm_version}"
    )
    logger.info(f"Stable remains at: {stable_dir}")
    logger.info(f"Last-stable updated at: {last_stable_dir}")


def copy_stable_all_versions(
    snapshots_root: Path,
    rocm_versions: List[str],
    skip_backup: bool = False,
) -> None:
    """
    Copy stable to last-stable for all specified ROCm versions.

    Args:
        snapshots_root: Root path of the snapshots directory
        rocm_versions: List of ROCm versions to process
        skip_backup: If True, skip the backup step (not recommended)
    """
    logger.info("=" * 60)
    logger.info("Copying stable to last-stable for all ROCm versions")
    logger.info("=" * 60)
    logger.info(f"ROCm versions: {', '.join(rocm_versions)}")
    logger.info("")

    # Step 0: Create a single backup for all versions before making any changes
    if not skip_backup:
        backup_name = backup_all_artifacts(DEFAULT_ROCM_VERSIONS)
        logger.info(f"Backup checkpoint created: {backup_name}")
    else:
        logger.warning("Skipping backup step (--skip-backup flag was set)")

    # Process each ROCm version (skip backup since we already did it once)
    for rocm_version in rocm_versions:
        logger.info("")
        logger.info(f"Processing ROCm {rocm_version}...")
        try:
            copy_stable_to_last_stable(
                rocm_version=rocm_version,
                snapshots_root=snapshots_root,
                skip_backup=True,  # Already backed up above
            )
        except Exception as e:
            logger.error(f"Error processing ROCm {rocm_version}: {e}")
            raise

    logger.info("")
    logger.info("=" * 60)
    logger.info("Successfully copied stable to last-stable for all ROCm versions")
    logger.info("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a snapshot of rcclx static library and header"
    )
    parser.add_argument(
        "--library",
        type=Path,
        help="Path to the built library file (e.g., librcclx-dev.a). "
        "Required unless --copy-stable is used.",
    )
    parser.add_argument(
        "--rocm-version",
        type=str,
        help="ROCm version (e.g., 6.2, 6.4, 7.0). "
        "Required unless --copy-stable is used.",
    )
    parser.add_argument(
        "--snapshots-root",
        type=Path,
        required=True,
        help="Root path of the snapshots directory",
    )
    parser.add_argument(
        "--rcclx-repo",
        type=Path,
        help="Path to the rcclx repository root. "
        "Required unless --copy-stable is used.",
    )
    parser.add_argument(
        "--header",
        type=Path,
        help="Path to the rccl.h header file to snapshot. "
        "Required unless --copy-stable is used.",
    )
    parser.add_argument(
        "--skip-backup",
        action="store_true",
        help="Skip the backup step before creating snapshot (not recommended)",
    )
    parser.add_argument(
        "--only-stable",
        action="store_true",
        help="Only update stable snapshot without rotating to last-stable. "
        "Last-stable will remain unchanged.",
    )
    parser.add_argument(
        "--copy-stable",
        action="store_true",
        help="Copy current stable to last-stable for all ROCm versions. "
        "Does not create a new stable snapshot.",
    )
    parser.add_argument(
        "--rocm-versions",
        type=str,
        default=",".join(DEFAULT_ROCM_VERSIONS),
        help=f"Comma-separated list of ROCm versions for --copy-stable mode "
        f"(default: {','.join(DEFAULT_ROCM_VERSIONS)})",
    )

    args = parser.parse_args()

    try:
        if args.copy_stable:
            # Copy stable mode - rotate stable to last-stable for all versions
            if args.only_stable:
                parser.error("--copy-stable and --only-stable are mutually exclusive")

            rocm_versions = [
                v.strip() for v in args.rocm_versions.split(",") if v.strip()
            ]
            if not rocm_versions:
                logger.error("No ROCm versions specified")
                return 1

            copy_stable_all_versions(
                snapshots_root=args.snapshots_root,
                rocm_versions=rocm_versions,
                skip_backup=args.skip_backup,
            )
        else:
            # Normal snapshot creation mode - validate required args
            if not args.library:
                parser.error("--library is required unless --copy-stable is used")
            if not args.rocm_version:
                parser.error("--rocm-version is required unless --copy-stable is used")
            if not args.rcclx_repo:
                parser.error("--rcclx-repo is required unless --copy-stable is used")
            if not args.header:
                parser.error("--header is required unless --copy-stable is used")

            create_snapshot(
                library_path=args.library,
                rocm_version=args.rocm_version,
                snapshots_root=args.snapshots_root,
                rcclx_repo_path=args.rcclx_repo,
                header_path=args.header,
                skip_backup=args.skip_backup,
                only_stable=args.only_stable,
            )
        return 0
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
