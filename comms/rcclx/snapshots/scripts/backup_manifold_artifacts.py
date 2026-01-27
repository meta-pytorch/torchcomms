#!/usr/bin/env python3
# pyre-strict
"""
Script to backup rcclx prebuilt artifacts within Manifold.

This script copies stable and last-stable artifacts from their current
locations in Manifold to an archives backup path with a timestamp prefix.

The backup structure in Manifold:
  rcclx_prebuilt_artifacts/tree/archives/
  └── backup_20260127_012000/
      ├── stable/
      │   ├── 6.2/librcclx-dev.a
      │   ├── 6.4/librcclx-dev.a
      │   └── 7.0/librcclx-dev.a
      └── last_stable/
          ├── 6.2/librcclx-dev.a
          ├── 6.4/librcclx-dev.a
          └── 7.0/librcclx-dev.a

Usage:
    buck2 run fbcode//comms/rcclx/snapshots/scripts:backup_manifold_artifacts

    # Or with specific ROCm versions:
    buck2 run fbcode//comms/rcclx/snapshots/scripts:backup_manifold_artifacts -- \\
        --rocm-versions 6.2,6.4,7.0
"""

import argparse
import io
import logging
import sys
from datetime import datetime, timedelta
from typing import List

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

# Default ROCm versions to backup
DEFAULT_ROCM_VERSIONS: List[str] = ["6.2", "6.4", "7.0"]

# Stages to backup
STAGES: List[str] = ["stable", "last_stable"]

# Library filename
LIBRARY_NAME: str = "librcclx-dev.a"


def get_manifold_path(stage: str, rocm_version: str, filename: str = "") -> str:
    """
    Construct a Manifold path for artifacts.

    Args:
        stage: One of 'stable', 'last_stable'
        rocm_version: ROCm version (e.g., '6.2')
        filename: Optional filename to append

    Returns:
        Full Manifold path (without bucket prefix)
    """
    if filename:
        return f"{MANIFOLD_PATH_PREFIX}{stage}/{rocm_version}/{filename}"
    return f"{MANIFOLD_PATH_PREFIX}{stage}/{rocm_version}"


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


def create_manifold_directory(dir_path: str) -> bool:
    """
    Create a directory in Manifold (tree namespace).

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


def copy_manifold_file(src_path: str, dst_path: str) -> bool:
    """
    Copy a file within Manifold by downloading and re-uploading.

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


def generate_backup_name() -> str:
    """
    Generate a backup directory name with timestamp.

    Returns:
        Backup name in format 'backup_YYYYMMDD_HHMMSS'
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"backup_{timestamp}"


def backup_artifacts(
    rocm_versions: List[str],
) -> str:
    """
    Backup all artifacts within Manifold to an archives backup path.

    Args:
        rocm_versions: List of ROCm versions to backup

    Returns:
        The backup path name that was created
    """
    backup_name = generate_backup_name()
    backup_base_path = f"{MANIFOLD_PATH_PREFIX}archives/{backup_name}"

    logger.info("=" * 60)
    logger.info("Backing up Manifold artifacts")
    logger.info("=" * 60)
    logger.info(f"Backup name: {backup_name}")
    logger.info(f"Backup path: {MANIFOLD_BUCKET}/{backup_base_path}/")
    logger.info(f"ROCm versions: {', '.join(rocm_versions)}")
    logger.info(f"Stages: {', '.join(STAGES)}")
    logger.info("")

    # Create the directory structure in Manifold before copying files
    logger.info("Creating backup directory structure in Manifold...")
    try:
        # Create the backup root directory
        create_manifold_directory(backup_base_path)
        logger.info(f"  -> Created: {backup_base_path}")

        # Create stage and version directories
        for stage in STAGES:
            stage_path = f"{backup_base_path}/{stage}"
            create_manifold_directory(stage_path)
            logger.info(f"  -> Created: {stage_path}")

            for rocm_version in rocm_versions:
                version_path = f"{stage_path}/{rocm_version}"
                create_manifold_directory(version_path)
                logger.info(f"  -> Created: {version_path}")
    except Exception as e:
        logger.error(f"Error creating directory structure: {e}")
        raise

    logger.info("")

    total_copied = 0
    total_missing = 0

    for stage in STAGES:
        for rocm_version in rocm_versions:
            src_path = get_manifold_path(stage, rocm_version, LIBRARY_NAME)
            dst_path = get_backup_path(backup_name, stage, rocm_version, LIBRARY_NAME)

            logger.info(f"Backing up {stage}/{rocm_version}/{LIBRARY_NAME}...")

            try:
                if copy_manifold_file(src_path, dst_path):
                    logger.info(f"  -> Copied to {dst_path}")
                    total_copied += 1
                else:
                    logger.warning(f"  -> Source not found: {src_path}")
                    total_missing += 1
            except Exception as e:
                logger.error(f"  -> Error copying: {e}")
                total_missing += 1

    logger.info("")
    logger.info("=" * 60)
    logger.info("Backup Summary")
    logger.info("=" * 60)
    logger.info(f"Total copied: {total_copied}")
    logger.info(f"Total missing/failed: {total_missing}")
    logger.info("")
    logger.info(f"Backup location: {MANIFOLD_BUCKET}/{backup_base_path}/")
    logger.info("")

    if total_copied > 0:
        logger.info("Backup structure:")
        for stage in STAGES:
            for rocm_version in rocm_versions:
                backup_path = get_backup_path(
                    backup_name, stage, rocm_version, LIBRARY_NAME
                )
                if check_manifold_path_exists(backup_path):
                    logger.info(f"  {backup_name}/{stage}/{rocm_version}/{LIBRARY_NAME}")

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"BACKUP PATH: {backup_name}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Use this path with restore_manifold_artifacts to restore:")
    logger.info(
        f"  buck2 run fbcode//comms/rcclx/snapshots/scripts:restore_manifold_artifacts -- \\"
    )
    logger.info(f"      --backup-name {backup_name}")
    logger.info("")

    return backup_name


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backup rcclx prebuilt artifacts within Manifold"
    )
    parser.add_argument(
        "--rocm-versions",
        type=str,
        default=",".join(DEFAULT_ROCM_VERSIONS),
        help=f"Comma-separated list of ROCm versions to backup (default: {','.join(DEFAULT_ROCM_VERSIONS)})",
    )

    args = parser.parse_args()

    # Parse ROCm versions
    rocm_versions = [v.strip() for v in args.rocm_versions.split(",") if v.strip()]

    if not rocm_versions:
        logger.error("No ROCm versions specified")
        return 1

    try:
        backup_name = backup_artifacts(rocm_versions=rocm_versions)
        print(f"\n{backup_name}")
        return 0
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
