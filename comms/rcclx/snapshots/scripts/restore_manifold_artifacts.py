#!/usr/bin/env python3
# pyre-strict
"""
Script to restore rcclx prebuilt artifacts from a Manifold backup.

This script restores stable and/or last-stable artifacts from a named
backup location in Manifold archives back to their original locations.

Backup structure in Manifold:
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
    # Restore both stable and last_stable from a backup
    buck2 run fbcode//comms/rcclx/snapshots/scripts:restore_manifold_artifacts -- \\
        --backup-name backup_20260127_012000

    # Dry run to see what would be restored
    buck2 run fbcode//comms/rcclx/snapshots/scripts:restore_manifold_artifacts -- \\
        --backup-name backup_20260127_012000 \\
        --dry-run

    # Restore only stable artifacts
    buck2 run fbcode//comms/rcclx/snapshots/scripts:restore_manifold_artifacts -- \\
        --backup-name backup_20260127_012000 \\
        --stages stable

    # Restore only last_stable artifacts
    buck2 run fbcode//comms/rcclx/snapshots/scripts:restore_manifold_artifacts -- \\
        --backup-name backup_20260127_012000 \\
        --stages last_stable

    # Restore specific ROCm versions
    buck2 run fbcode//comms/rcclx/snapshots/scripts:restore_manifold_artifacts -- \\
        --backup-name backup_20260127_012000 \\
        --rocm-versions 6.2,6.4
"""

import argparse
import io
import logging
import sys
from datetime import timedelta
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

# Default ROCm versions
DEFAULT_ROCM_VERSIONS: List[str] = ["6.2", "6.4", "7.0"]

# Default stages to restore
DEFAULT_STAGES: List[str] = ["stable", "last_stable"]

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


def copy_manifold_file(src_path: str, dst_path: str, overwrite: bool = True) -> bool:
    """
    Copy a file within Manifold by downloading and re-uploading.

    Args:
        src_path: Source Manifold path (without bucket prefix)
        dst_path: Destination Manifold path (without bucket prefix)
        overwrite: If True, delete existing destination file before copy

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

            # Delete destination if it exists and overwrite is enabled
            if overwrite and await client.exists(path=dst_path):
                logger.info(f"  Deleting existing file at {dst_path}")
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

            return True

    return await_sync(async_copy())


def restore_artifacts(
    backup_name: str,
    rocm_versions: List[str],
    stages: List[str],
    dry_run: bool = False,
) -> None:
    """
    Restore artifacts from a backup location in Manifold.

    Args:
        backup_name: Name of the backup directory (e.g., 'backup_20260127_012000')
        rocm_versions: List of ROCm versions to restore
        stages: List of stages to restore ('stable', 'last_stable')
        dry_run: If True, only log what would be done
    """
    backup_base_path = f"{MANIFOLD_PATH_PREFIX}archives/{backup_name}"

    logger.info("=" * 60)
    logger.info("Restoring artifacts from Manifold backup")
    logger.info("=" * 60)
    logger.info(f"Backup name: {backup_name}")
    logger.info(f"Backup path: {MANIFOLD_BUCKET}/{backup_base_path}/")
    logger.info(f"ROCm versions: {', '.join(rocm_versions)}")
    logger.info(f"Stages: {', '.join(stages)}")
    logger.info(f"Dry run: {dry_run}")
    logger.info("")

    total_restored = 0
    total_missing = 0
    total_failed = 0

    for stage in stages:
        for rocm_version in rocm_versions:
            src_path = get_backup_path(backup_name, stage, rocm_version, LIBRARY_NAME)
            dst_path = get_manifold_path(stage, rocm_version, LIBRARY_NAME)

            logger.info(f"Restoring {stage}/{rocm_version}/{LIBRARY_NAME}...")

            if dry_run:
                if check_manifold_path_exists(src_path):
                    logger.info(f"  [DRY RUN] Would copy from {src_path} to {dst_path}")
                    total_restored += 1
                else:
                    logger.warning(f"  [DRY RUN] Backup not found: {src_path}")
                    total_missing += 1
            else:
                try:
                    if copy_manifold_file(src_path, dst_path):
                        logger.info(f"  -> Restored to {dst_path}")
                        total_restored += 1
                    else:
                        logger.warning(f"  -> Backup not found: {src_path}")
                        total_missing += 1
                except Exception as e:
                    logger.error(f"  -> Error restoring: {e}")
                    total_failed += 1

    logger.info("")
    logger.info("=" * 60)
    logger.info("Restore Summary")
    logger.info("=" * 60)
    logger.info(f"Total restored: {total_restored}")
    logger.info(f"Total missing (not in backup): {total_missing}")
    logger.info(f"Total failed: {total_failed}")

    if dry_run:
        logger.info("")
        logger.info("This was a dry run. No changes were made to Manifold.")
        logger.info("Run without --dry-run to apply changes.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Restore rcclx prebuilt artifacts from Manifold backup"
    )
    parser.add_argument(
        "--backup-name",
        type=str,
        required=True,
        help="Name of the backup directory (e.g., 'backup_20260127_012000')",
    )
    parser.add_argument(
        "--rocm-versions",
        type=str,
        default=",".join(DEFAULT_ROCM_VERSIONS),
        help=f"Comma-separated list of ROCm versions (default: {','.join(DEFAULT_ROCM_VERSIONS)})",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default=",".join(DEFAULT_STAGES),
        help=f"Comma-separated list of stages to restore (default: {','.join(DEFAULT_STAGES)})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only log what would be done, without making changes to Manifold",
    )

    args = parser.parse_args()

    # Parse ROCm versions
    rocm_versions = [v.strip() for v in args.rocm_versions.split(",") if v.strip()]

    if not rocm_versions:
        logger.error("No ROCm versions specified")
        return 1

    # Parse stages
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    if not stages:
        logger.error("No stages specified")
        return 1

    # Validate stages
    valid_stages = {"stable", "last_stable"}
    for stage in stages:
        if stage not in valid_stages:
            logger.error(f"Invalid stage: {stage}. Valid stages: {valid_stages}")
            return 1

    try:
        restore_artifacts(
            backup_name=args.backup_name,
            rocm_versions=rocm_versions,
            stages=stages,
            dry_run=args.dry_run,
        )
        return 0
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
