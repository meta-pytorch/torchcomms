#!/usr/bin/env python3
# pyre-strict
"""
Script to create snapshots of rcclx static libraries.

This script handles:
1. Moving last-stable to archives with timestamp (metadata in repo, artifacts in Manifold)
2. Moving stable to last-stable (metadata in repo, artifacts in Manifold)
3. Copying newly built library to stable (metadata in repo, artifacts in Manifold)
4. Creating metadata file with commit hashes

Artifacts (.a files) are stored in Manifold at:
  rcclx_prebuilt_artifacts/tree/stable/{rocm_version}/librcclx-dev.a
  rcclx_prebuilt_artifacts/tree/last_stable/{rocm_version}/librcclx-dev.a
  rcclx_prebuilt_artifacts/tree/archives/{rocm_version}/{timestamp}_librcclx-dev.a

Metadata files are stored in the repo at:
  snapshots/stable/{rocm_version}/metadata.txt
  snapshots/last-stable/{rocm_version}/metadata.txt
  snapshots/archives/{rocm_version}/{timestamp}_metadata.txt
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
from typing import Dict, Optional

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
) -> None:
    """
    Create a snapshot of the rcclx library.

    This function handles the rotation of snapshots and saves artifacts to Manifold
    while keeping metadata in the repo.

    Args:
        library_path: Path to the built library file (e.g., librcclx-dev.a)
        rocm_version: ROCm version (e.g., "6.2", "6.4", "7.0")
        snapshots_root: Root path of the snapshots directory
        rcclx_repo_path: Path to the rcclx repository for getting commit info
    """
    logger.info(f"Creating snapshot for ROCm {rocm_version}")

    # Validate inputs
    if not library_path.exists():
        raise FileNotFoundError(f"Library file not found: {library_path}")

    if not snapshots_root.exists():
        raise FileNotFoundError(f"Snapshots directory not found: {snapshots_root}")

    # Define directories for metadata (in repo)
    stable_dir = snapshots_root / "stable" / rocm_version
    last_stable_dir = snapshots_root / "last-stable" / rocm_version
    archives_dir = snapshots_root / "archives" / rocm_version

    # Create directories if they don't exist
    stable_dir.mkdir(parents=True, exist_ok=True)
    last_stable_dir.mkdir(parents=True, exist_ok=True)
    archives_dir.mkdir(parents=True, exist_ok=True)

    library_name = library_path.name

    # Step 1: Move last-stable to archives with timestamp-prefixed filenames
    # Check if last-stable metadata exists in repo
    if not is_directory_empty(last_stable_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(
            f"Archiving last-stable to archives/{rocm_version}/{timestamp}_metadata.txt"
        )

        # Archive metadata in repo with timestamp prefix
        for item in last_stable_dir.iterdir():
            if item.is_file():
                archive_filename = f"{timestamp}_{item.name}"
                archive_path = archives_dir / archive_filename
                shutil.move(str(item), str(archive_path))
                logger.info(f"Archived metadata: {archive_path}")

        # Archive artifact in Manifold (if exists) with timestamp prefix
        last_stable_manifold_path = get_manifold_path(
            "last_stable", rocm_version, library_name
        )
        if check_manifold_path_exists(last_stable_manifold_path):
            archive_library_name = f"{timestamp}_{library_name}"
            archive_manifold_path = get_manifold_path(
                "archives", rocm_version, archive_library_name
            )
            copy_manifold_file(last_stable_manifold_path, archive_manifold_path)
            delete_manifold_file(last_stable_manifold_path)
            logger.info(f"Archived artifact in Manifold: {archive_manifold_path}")

    # Step 2: Move stable to last-stable
    # Check if stable metadata exists in repo
    if not is_directory_empty(stable_dir):
        logger.info(f"Moving stable to last-stable/{rocm_version}/")

        # Move metadata in repo
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

    # Step 3: Calculate checksums for the library before uploading
    logger.info(f"Calculating checksums for {library_path}")
    checksums = calculate_file_checksums(library_path)
    logger.info(f"SHA1: {checksums['sha1']}")
    logger.info(f"SHA256: {checksums['sha256']}")

    # Step 4: Upload newly built library to Manifold stable
    stable_manifold_path = get_manifold_path("stable", rocm_version, library_name)
    upload_file_to_manifold(library_path, stable_manifold_path)

    # Step 5: Create metadata file in repo stable with checksums
    deps_info = get_dependency_info(str(rcclx_repo_path))
    metadata_path = stable_dir / "metadata.txt"
    create_metadata_file(metadata_path, rocm_version, deps_info, checksums)

    logger.info(f"Snapshot creation completed successfully for ROCm {rocm_version}")
    logger.info(
        f"Artifact stored in Manifold at: {MANIFOLD_BUCKET}/{stable_manifold_path}"
    )
    logger.info(f"Metadata stored in repo at: {metadata_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a snapshot of rcclx static library"
    )
    parser.add_argument(
        "--library",
        type=Path,
        required=True,
        help="Path to the built library file (e.g., librcclx-dev.a)",
    )
    parser.add_argument(
        "--rocm-version",
        type=str,
        required=True,
        help="ROCm version (e.g., 6.2, 6.4, 7.0)",
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
        required=True,
        help="Path to the rcclx repository root",
    )

    args = parser.parse_args()

    try:
        create_snapshot(
            library_path=args.library,
            rocm_version=args.rocm_version,
            snapshots_root=args.snapshots_root,
            rcclx_repo_path=args.rcclx_repo,
        )
        return 0
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
