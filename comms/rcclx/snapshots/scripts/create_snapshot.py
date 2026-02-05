#!/usr/bin/env python3
# pyre-strict
"""
Script to create snapshots of rcclx sources from a specific commit.

This script extracts the rcclx directory from a specified commit hash and
saves it to the snapshots directory. The extracted sources are then built
from source at build time, which ensures ABI compatibility by compiling
all code together with current external dependencies.

Usage:
    # Snapshot from specific commit
    python3 create_snapshot.py \\
        --stage stable \\
        --commit abc123def \\
        --snapshots-root fbcode/comms/rcclx/snapshots \\
        --repo-root /path/to/fbsource

    # Snapshot from current HEAD
    python3 create_snapshot.py \\
        --stage stable \\
        --snapshots-root fbcode/comms/rcclx/snapshots \\
        --repo-root /path/to/fbsource

    # Rotate: copy stable → last-stable, then create new stable
    python3 create_snapshot.py \\
        --stage stable \\
        --commit abc123def \\
        --rotate \\
        --snapshots-root fbcode/comms/rcclx/snapshots \\
        --repo-root /path/to/fbsource

The snapshot structure is:
    snapshots/stable/rcclx/          # Extracted rcclx sources
    snapshots/stable/metadata.txt    # Commit hash, timestamp
    snapshots/last-stable/rcclx/
    snapshots/last-stable/metadata.txt
"""

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger: logging.Logger = logging.getLogger(__name__)


def get_current_commit(repo_root: Path) -> str:
    """Get the current commit hash from the repository."""
    result = subprocess.run(
        ["sl", "log", "-r", ".", "-T", "{node}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def get_commit_info(commit: str, repo_root: Path) -> str:
    """Get commit information (hash, date, description) for metadata."""
    result = subprocess.run(
        ["sl", "log", "-r", commit, "-T", "{node}\\n{date|isodate}\\n{desc|firstline}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def extract_rcclx_from_commit(
    commit: str,
    dest: Path,
    repo_root: Path,
    stage: str,
) -> None:
    """
    Extract core rcclx source files from a specific commit using sl archive.

    Only extracts the essential AMD drop bits:
    - BUCK
    - defs.bzl
    - develop/
    - rccl_build_config.bzl
    - utils.bzl

    Args:
        commit: The commit hash to extract from
        dest: Destination directory for extracted sources
        repo_root: Path to the repository root
        stage: The snapshot stage name ("stable" or "last-stable")
    """
    logger.info(f"Extracting rcclx sources from commit {commit[:12]}...")

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / "rcclx.tar"

        # Use sl archive to extract only the core rcclx files from specific commit
        # Only include: BUCK, defs.bzl, develop/, rccl_build_config.bzl, utils.bzl
        cmd = [
            "sl",
            "archive",
            "--rev",
            commit,
            "--include",
            "fbcode/comms/rcclx/BUCK",
            "--include",
            "fbcode/comms/rcclx/defs.bzl",
            "--include",
            "fbcode/comms/rcclx/develop/**",
            "--include",
            "fbcode/comms/rcclx/rccl_build_config.bzl",
            "--include",
            "fbcode/comms/rcclx/utils.bzl",
            "--exclude",
            "**/tests/**",
            "--exclude",
            "**/test/**",
            "--exclude",
            "**/__pycache__/**",
            "--exclude",
            "**/buck-out/**",
            "--exclude",
            "**/.buckd/**",
            "-t",
            "tar",
            str(archive_path),
        ]

        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"sl archive failed: {result.stderr}")
            raise RuntimeError(f"Failed to archive rcclx from commit {commit}: {result.stderr}")

        # Create destination directory
        # We want the final structure to be dest/rcclx/ containing the files
        dest.mkdir(parents=True, exist_ok=True)

        # Extract the archive
        # The archive contains <archive-name>/fbcode/comms/rcclx/..., we want to extract to dest/
        # Strip 2 components (<archive-name>/fbcode/) so comms/rcclx/ directory is preserved
        # Final result: dest/comms/rcclx/BUCK, dest/comms/rcclx/develop/, etc.
        # This is required because includes like "comms/rcclx/develop/..." need the full path
        extract_cmd = [
            "tar",
            "-xf",
            str(archive_path),
            "-C",
            str(dest),
            "--strip-components=2",  # Strip <archive-name>/fbcode/ prefix, keep comms/rcclx/
        ]

        logger.info(f"Extracting archive to {dest}")
        result = subprocess.run(
            extract_cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"tar extraction failed: {result.stderr}")
            raise RuntimeError(f"Failed to extract archive: {result.stderr}")

    logger.info(f"Successfully extracted rcclx sources to {dest / 'comms' / 'rcclx'}")

    # Fix absolute load paths to be relative within the snapshot
    fix_load_paths_in_snapshot(dest, stage)


def fix_load_paths_in_snapshot(dest: Path, stage: str) -> None:
    """
    Fix absolute load paths in .bzl and BUCK files to point within the snapshot.

    The extracted sources contain load statements like:
        load("@fbcode//comms/rcclx/develop:def_build.bzl", ...)
        load("@fbcode//comms/rcclx:utils.bzl", ...)
        load("@fbcode//comms/rcclx:rccl_build_config.bzl", ...)

    These need to be changed to point to the snapshot's own files:
        load("//comms/rcclx/snapshots/{stage}/comms/rcclx/develop:def_build.bzl", ...)
        load("//comms/rcclx/snapshots/{stage}/comms/rcclx:utils.bzl", ...)
        load("//comms/rcclx/snapshots/{stage}/comms/rcclx:rccl_build_config.bzl", ...)

    Additionally, we add compiler flags to suppress warnings that were not errors
    in older code but are treated as errors with current compiler settings.

    This ensures rcclx-stable uses the snapshot's .bzl files, not HEAD's.

    Args:
        dest: Path to the snapshot stage directory (e.g., snapshots/stable/)
        stage: The snapshot stage name ("stable" or "last-stable")
    """
    # Files are extracted to dest/comms/rcclx/
    # (BUCK, defs.bzl, develop/, rccl_build_config.bzl, utils.bzl)
    rcclx_dir = dest / "comms" / "rcclx"

    if not (rcclx_dir / "rccl_build_config.bzl").exists():
        logger.warning(f"rccl_build_config.bzl not found at {rcclx_dir}, skipping load path fixes")
        return

    # Build the snapshot package path for this stage
    # Snapshot is at //comms/rcclx/snapshots/{stage}/comms/rcclx
    snapshot_pkg = f"//comms/rcclx/snapshots/{stage}/comms/rcclx"

    # Fix load paths in rccl_build_config.bzl
    _fix_load_paths(rcclx_dir / "rccl_build_config.bzl", snapshot_pkg)

    # Add compiler flag to suppress deprecated-this-capture warning
    _add_snapshot_compiler_flags(rcclx_dir / "rccl_build_config.bzl")

    # Add include path flag so that "comms/rcclx/..." includes resolve within the snapshot
    _add_snapshot_include_path(rcclx_dir / "rccl_build_config.bzl", stage)

    # Remove obsolete manifold-related targets from BUCK file
    _remove_manifold_targets_from_buck(rcclx_dir)

    # Fix load paths in all BUCK files recursively (subdirectories like develop/meta/lib/BUCK)
    _fix_load_paths_in_all_buck_files(rcclx_dir, snapshot_pkg)

    # Fix defs.bzl API compatibility issues (e.g., target_sdk_version -> minimum_os_version)
    _fix_defs_bzl_compatibility(rcclx_dir)

    # Fix the :rcclx alias to point directly to :rcclx-dev (snapshots only have one version)
    _fix_rcclx_alias_in_buck(rcclx_dir)

    # Clean up unused imports from the main BUCK file
    _cleanup_unused_imports_in_buck(rcclx_dir)

    # Fix genrule exec_compatible_with for cross-platform compatibility
    _fix_genrule_exec_compatibility(rcclx_dir)

    # Fix duplicate top-level assignment lint error in rccl_build_config.bzl
    _fix_duplicate_top_level_assign(rcclx_dir / "rccl_build_config.bzl")


def _fix_load_paths(file_path: Path, snapshot_pkg: str) -> None:
    """Fix absolute load paths in a .bzl file."""
    if not file_path.exists():
        logger.warning(f"File not found, skipping load path fixes: {file_path}")
        return

    # Patterns to replace: absolute paths -> snapshot-specific paths
    # With flat structure, snapshot_pkg is //comms/rcclx/snapshots/{stage}
    replacements = [
        # @fbcode//comms/rcclx/develop:def_build.bzl -> //comms/rcclx/snapshots/{stage}/develop:def_build.bzl
        (
            '"@fbcode//comms/rcclx/develop:def_build.bzl"',
            f'"{snapshot_pkg}/develop:def_build.bzl"',
        ),
        # //comms/rcclx/develop:def_build.bzl -> //comms/rcclx/snapshots/{stage}/develop:def_build.bzl
        (
            '"//comms/rcclx/develop:def_build.bzl"',
            f'"{snapshot_pkg}/develop:def_build.bzl"',
        ),
        # @fbcode//comms/rcclx:utils.bzl -> //comms/rcclx/snapshots/{stage}:utils.bzl
        (
            '"@fbcode//comms/rcclx:utils.bzl"',
            f'"{snapshot_pkg}:utils.bzl"',
        ),
        # //comms/rcclx:utils.bzl -> //comms/rcclx/snapshots/{stage}:utils.bzl
        (
            '"//comms/rcclx:utils.bzl"',
            f'"{snapshot_pkg}:utils.bzl"',
        ),
    ]

    logger.info(f"Fixing load paths in {file_path}")

    with open(file_path, "r") as f:
        content = f.read()

    original_content = content
    for old_str, new_str in replacements:
        content = content.replace(old_str, new_str)

    if content != original_content:
        with open(file_path, "w") as f:
            f.write(content)
        logger.info(f"  Updated load paths in {file_path.name}")
    else:
        logger.info(f"  No load path changes needed in {file_path.name}")


def _fix_load_paths_in_all_buck_files(rcclx_dir: Path, snapshot_pkg: str) -> None:
    """
    Fix absolute load paths in all BUCK files throughout the snapshot recursively.

    This handles BUCK files in subdirectories like develop/meta/lib/BUCK which
    may also have load statements referencing the main rcclx package.

    Additionally, this function updates targets to use COMMON_PRE_COMPILER_FLAGS
    for preprocessor_flags, which contains the include path needed for snapshot builds.

    Args:
        rcclx_dir: Path to the rcclx directory within the snapshot
        snapshot_pkg: The snapshot package path (e.g., //comms/rcclx/snapshots/stable/comms/rcclx)
    """
    logger.info(f"Fixing load paths in all BUCK files under {rcclx_dir}")

    # Patterns to replace in BUCK files
    # These load statements reference the main rcclx package and need to be updated
    # to use the snapshot's own files
    replacements = [
        # @fbcode//comms/rcclx:rccl_build_config.bzl -> snapshot:rccl_build_config.bzl
        (
            '"@fbcode//comms/rcclx:rccl_build_config.bzl"',
            f'"{snapshot_pkg}:rccl_build_config.bzl"',
        ),
        # //comms/rcclx:rccl_build_config.bzl -> snapshot:rccl_build_config.bzl
        (
            '"//comms/rcclx:rccl_build_config.bzl"',
            f'"{snapshot_pkg}:rccl_build_config.bzl"',
        ),
        # @fbcode//comms/rcclx:defs.bzl -> snapshot:defs.bzl
        (
            '"@fbcode//comms/rcclx:defs.bzl"',
            f'"{snapshot_pkg}:defs.bzl"',
        ),
        # //comms/rcclx:defs.bzl -> snapshot:defs.bzl
        (
            '"//comms/rcclx:defs.bzl"',
            f'"{snapshot_pkg}:defs.bzl"',
        ),
        # @fbcode//comms/rcclx:utils.bzl -> snapshot:utils.bzl
        (
            '"@fbcode//comms/rcclx:utils.bzl"',
            f'"{snapshot_pkg}:utils.bzl"',
        ),
        # //comms/rcclx:utils.bzl -> snapshot:utils.bzl
        (
            '"//comms/rcclx:utils.bzl"',
            f'"{snapshot_pkg}:utils.bzl"',
        ),
        # @fbcode//comms/rcclx/develop:def_build.bzl -> snapshot/develop:def_build.bzl
        (
            '"@fbcode//comms/rcclx/develop:def_build.bzl"',
            f'"{snapshot_pkg}/develop:def_build.bzl"',
        ),
        # //comms/rcclx/develop:def_build.bzl -> snapshot/develop:def_build.bzl
        (
            '"//comms/rcclx/develop:def_build.bzl"',
            f'"{snapshot_pkg}/develop:def_build.bzl"',
        ),
    ]

    # Find all BUCK files recursively
    buck_files_updated = 0
    for buck_file in rcclx_dir.rglob("BUCK"):
        with open(buck_file, "r") as f:
            content = f.read()

        original_content = content

        # Apply load path replacements
        for old_str, new_str in replacements:
            content = content.replace(old_str, new_str)

        # Update load statements to also import COMMON_PRE_COMPILER_FLAGS
        # This is needed so targets can use it in preprocessor_flags for include paths
        content = _add_common_pre_compiler_flags_import(content)

        # Update targets to use COMMON_PRE_COMPILER_FLAGS in preprocessor_flags
        # This ensures the snapshot include path is available to all targets
        content = _update_preprocessor_flags(content)

        if content != original_content:
            with open(buck_file, "w") as f:
                f.write(content)
            logger.info(f"  Updated load paths in {buck_file.relative_to(rcclx_dir.parent.parent)}")
            buck_files_updated += 1

    logger.info(f"  Updated {buck_files_updated} BUCK file(s)")


def _add_common_pre_compiler_flags_import(content: str) -> str:
    """
    Update load statements to also import COMMON_PRE_COMPILER_FLAGS.

    If the BUCK file loads COMMON_COMPILER_FLAGS from rccl_build_config.bzl,
    also import COMMON_PRE_COMPILER_FLAGS which contains the snapshot include path.
    """
    import re

    # Pattern to match load statement that imports COMMON_COMPILER_FLAGS but not COMMON_PRE_COMPILER_FLAGS
    # Match: load("...rccl_build_config.bzl", "COMMON_COMPILER_FLAGS")
    pattern = r'(load\([^)]*:rccl_build_config\.bzl",\s*"COMMON_COMPILER_FLAGS")(\))'

    # Check if COMMON_PRE_COMPILER_FLAGS is already imported
    if "COMMON_PRE_COMPILER_FLAGS" in content:
        return content

    # Add COMMON_PRE_COMPILER_FLAGS to the import
    replacement = r'\1, "COMMON_PRE_COMPILER_FLAGS"\2'
    content = re.sub(pattern, replacement, content)

    return content


def _update_preprocessor_flags(content: str) -> str:
    """
    Update targets to include COMMON_PRE_COMPILER_FLAGS in preprocessor_flags.

    This handles targets that have hardcoded preprocessor_flags like:
        preprocessor_flags = ["-DBUILD_META_INTERNAL"],

    And updates them to:
        preprocessor_flags = COMMON_PRE_COMPILER_FLAGS + ["-DBUILD_META_INTERNAL"],

    This ensures the snapshot include path (defined in COMMON_PRE_COMPILER_FLAGS)
    is available to all targets that need it.

    IMPORTANT: Only modifies files that import COMMON_PRE_COMPILER_FLAGS, to avoid
    breaking files that don't have access to this variable.
    """
    import re

    # Only modify files that have COMMON_PRE_COMPILER_FLAGS imported
    # This prevents breaking BUCK files that don't load from rccl_build_config.bzl
    if "COMMON_PRE_COMPILER_FLAGS" not in content:
        return content

    # Pattern to match preprocessor_flags = [...] that doesn't already include COMMON_PRE_COMPILER_FLAGS
    # We need to be careful to only match literal list definitions, not variable references
    pattern = r'preprocessor_flags\s*=\s*(\[[^\]]*\])'

    def replace_preprocessor_flags(match: re.Match[str]) -> str:
        flags_list = match.group(1)
        # Don't modify if it's already using COMMON_PRE_COMPILER_FLAGS
        if "COMMON_PRE_COMPILER_FLAGS" in match.group(0):
            return match.group(0)
        # Prepend COMMON_PRE_COMPILER_FLAGS to the existing flags
        return f"preprocessor_flags = COMMON_PRE_COMPILER_FLAGS + {flags_list}"

    content = re.sub(pattern, replace_preprocessor_flags, content)

    return content


def _fix_rcclx_alias_in_buck(rcclx_dir: Path) -> None:
    """
    Fix the :rcclx alias in snapshot BUCK files to point directly to :rcclx-dev.

    The original BUCK file has an alias that selects between different versions:
        fb_native.alias(
            name = "rcclx",
            actual = select({
                "DEFAULT": ":rcclx-stable",
                ...
            }),
        )

    But in a snapshot, we only have :rcclx-dev, so the alias should point directly to it.
    This replaces the select-based alias with a simple alias to :rcclx-dev.
    """
    buck_file = rcclx_dir / "BUCK"
    if not buck_file.exists():
        logger.warning(f"BUCK file not found at {buck_file}, skipping rcclx alias fix")
        return

    logger.info(f"Fixing :rcclx alias in {buck_file}")

    with open(buck_file, "r") as f:
        content = f.read()

    original_content = content

    # Pattern to match the entire rcclx alias block with select
    # This is a multi-line pattern matching from fb_native.alias to the closing )
    import re

    # Match the rcclx alias with select statement
    # The pattern captures the entire alias block including the select
    alias_pattern = r'''fb_native\.alias\(
    name = "rcclx",
    actual = select\(\{[^}]+\}\),
    visibility = \["PUBLIC"\],
\)'''

    # Replacement: simple alias pointing to rcclx-dev
    replacement = '''fb_native.alias(
    name = "rcclx",
    actual = ":rcclx-dev",
    visibility = ["PUBLIC"],
)'''

    content = re.sub(alias_pattern, replacement, content, flags=re.DOTALL)

    if content != original_content:
        with open(buck_file, "w") as f:
            f.write(content)
        logger.info(f"  Fixed :rcclx alias to point directly to :rcclx-dev")
    else:
        logger.info(f"  No :rcclx alias fix needed (may already be fixed)")


def _cleanup_unused_imports_in_buck(rcclx_dir: Path) -> None:
    """
    Remove unused imports from the main snapshot BUCK file.

    The original BUCK file imports symbols that are only used in the multi-version
    setup (e.g., get_internal_deps, get_internal_exported_deps, get_rccl_fbcode_exported_deps).
    These are not needed in the simplified snapshot BUCK file.

    This function also removes the comments about multi-version usage since they
    don't apply to snapshots.
    """
    buck_file = rcclx_dir / "BUCK"
    if not buck_file.exists():
        logger.warning(f"BUCK file not found at {buck_file}, skipping unused imports cleanup")
        return

    logger.info(f"Cleaning up unused imports in {buck_file}")

    with open(buck_file, "r") as f:
        content = f.read()

    original_content = content

    import re

    # Remove unused imports from the rccl_build_config.bzl load statement
    # These symbols are used in the multi-version BUCK but not in snapshots
    unused_imports = [
        '"get_internal_deps",',
        '"get_internal_exported_deps",',
        '"get_rccl_fbcode_exported_deps",',
    ]

    for unused_import in unused_imports:
        # Remove the line containing the unused import (with leading whitespace)
        pattern = rf'\n\s*{re.escape(unused_import)}'
        content = re.sub(pattern, '', content)

    # Remove multi-version usage comments that don't apply to snapshots
    multi_version_comment_pattern = r'''# Default rcclx library target - automatically selects the appropriate version based on constraints
# Usage:
#   buck2 build fbcode//comms/rcclx:rcclx                                 # Uses rcclx-dev \(default\)
#   buck2 build -m rcclx_dev fbcode//comms/rcclx:rcclx                   # Uses rcclx-dev
#   buck2 build -m rcclx_stable fbcode//comms/rcclx:rcclx                # Uses rcclx-stable
#   buck2 build -m rcclx_last_stable fbcode//comms/rcclx:rcclx           # Uses rcclx-last-stable'''

    snapshot_comment = '''# Default rcclx library target - in snapshots, points directly to rcclx-dev
# (Snapshots represent a single fixed version, not a selection between versions)'''

    content = re.sub(multi_version_comment_pattern, snapshot_comment, content)

    if content != original_content:
        with open(buck_file, "w") as f:
            f.write(content)
        logger.info(f"  Removed unused imports from BUCK file")
    else:
        logger.info(f"  No unused imports cleanup needed")


def _fix_genrule_exec_compatibility(rcclx_dir: Path) -> None:
    """
    Fix genrule exec_compatible_with for cross-platform compatibility.

    The buck_genrule for rcclx-shared uses `bash` but doesn't specify `cmd_exe`
    or `exec_compatible_with`, which causes BUCKLINT errors when cross-building
    (e.g., targeting posix from a Windows host).

    This adds `exec_compatible_with = ["ovr_config//os:linux"]` to specify that
    the genrule only works on Linux hosts.

    Args:
        rcclx_dir: Path to the rcclx directory within the snapshot
    """
    buck_file = rcclx_dir / "BUCK"
    if not buck_file.exists():
        logger.warning(f"BUCK file not found at {buck_file}, skipping genrule fix")
        return

    logger.info(f"Fixing genrule exec_compatible_with in {buck_file}")

    with open(buck_file, "r") as f:
        content = f.read()

    original_content = content

    # Check if already fixed
    if 'exec_compatible_with = ["ovr_config//os:linux"]' in content:
        logger.info(f"  genrule exec_compatible_with already present")
        return

    # Pattern to match the rcclx-shared genrule without exec_compatible_with
    old_genrule = '''buck_genrule(
    name = "rcclx-shared",
    out = "librccl.so.1",
    bash = "$(location fbsource//third-party/rocm:rocm_path)/llvm/bin/clang -shared -o $OUT -Wl,-soname,librccl.so.1 $(location :rcclx-dev)",
    visibility = ["PUBLIC"],
)'''

    new_genrule = '''buck_genrule(
    name = "rcclx-shared",
    out = "librccl.so.1",
    bash = "$(location fbsource//third-party/rocm:rocm_path)/llvm/bin/clang -shared -o $OUT -Wl,-soname,librccl.so.1 $(location :rcclx-dev)",
    exec_compatible_with = ["ovr_config//os:linux"],
    visibility = ["PUBLIC"],
)'''

    content = content.replace(old_genrule, new_genrule)

    if content != original_content:
        with open(buck_file, "w") as f:
            f.write(content)
        logger.info(f"  Added exec_compatible_with to rcclx-shared genrule")
    else:
        logger.info(f"  No genrule fix needed (pattern not found or already fixed)")


def _fix_duplicate_top_level_assign(file_path: Path) -> None:
    """
    Fix duplicate top-level assignment lint error for COMMON_PRE_COMPILER_FLAGS.

    The original rccl_build_config.bzl has a pattern like:
        COMMON_PRE_COMPILER_FLAGS = [...] + get_npkit_compiler_flags() + ...

        if read_bool("rccl", "inject_faults", False):
            COMMON_PRE_COMPILER_FLAGS += ["-DENABLE_FAULT_INJECTION"]

    This triggers a STARLARK lint error "duplicate-top-level-assign" because
    the variable is assigned and then modified with +=.

    This function refactors the code to use a helper function instead:
        def _get_fault_injection_flags():
            if read_bool("rccl", "inject_faults", False):
                return ["-DENABLE_FAULT_INJECTION"]
            return []

        COMMON_PRE_COMPILER_FLAGS = [...] + ... + _get_fault_injection_flags()

    Args:
        file_path: Path to the rccl_build_config.bzl file
    """
    if not file_path.exists():
        logger.warning(f"File not found, skipping duplicate assignment fix: {file_path}")
        return

    logger.info(f"Fixing duplicate top-level assignment in {file_path}")

    with open(file_path, "r") as f:
        content = f.read()

    original_content = content

    # Check if already fixed (helper function exists)
    if "_get_fault_injection_flags" in content:
        logger.info(f"  duplicate assignment already fixed")
        return

    # Pattern to match the += assignment block
    old_pattern = '''
# Inject random warp delay in device code
if read_bool("rccl", "inject_faults", False):
    COMMON_PRE_COMPILER_FLAGS += ["-DENABLE_FAULT_INJECTION"]'''

    # Check if the pattern exists
    if old_pattern not in content:
        logger.info(f"  No duplicate assignment pattern found")
        return

    # Remove the old += block
    content = content.replace(old_pattern, "")

    # Find where COMMON_PRE_COMPILER_FLAGS is defined and add the helper function before it
    # Look for the pattern that ends the COMMON_PRE_COMPILER_FLAGS definition
    import re

    # Find patterns like:
    # ] + get_npkit_compiler_flags() + get_rdma_core_compiler_flags() + get_host_uncahced_memory_flags()
    # and append + _get_fault_injection_flags()

    # Pattern to match the end of COMMON_PRE_COMPILER_FLAGS definition
    end_pattern = r'(\] \+ get_npkit_compiler_flags\(\) \+ get_rdma_core_compiler_flags\(\) \+ get_host_uncahced_memory_flags\(\))'

    if re.search(end_pattern, content):
        # Add the helper function call to the end
        content = re.sub(
            end_pattern,
            r'\1 + _get_fault_injection_flags()',
            content,
        )

        # Add the helper function definition before COMMON_PRE_COMPILER_FLAGS
        helper_function = '''def _get_fault_injection_flags():
    if read_bool("rccl", "inject_faults", False):
        return ["-DENABLE_FAULT_INJECTION"]
    return []

'''
        # Insert before COMMON_PRE_COMPILER_FLAGS definition
        common_flags_pattern = r'(COMMON_PRE_COMPILER_FLAGS = \[)'
        content = re.sub(
            common_flags_pattern,
            helper_function + r'\1',
            content,
            count=1,  # Only replace the first occurrence
        )

        if content != original_content:
            with open(file_path, "w") as f:
                f.write(content)
            logger.info(f"  Fixed duplicate top-level assignment")
        else:
            logger.info(f"  No changes made")
    else:
        logger.warning(f"  Could not find COMMON_PRE_COMPILER_FLAGS end pattern")


def _fix_defs_bzl_compatibility(rcclx_dir: Path) -> None:
    """
    Fix API compatibility issues in defs.bzl for older snapshots.

    The toolchain provider API has changed over time. This function patches
    the defs.bzl file to use the current API names.

    Known changes:
    - target_sdk_version was renamed to minimum_os_version
    """
    defs_file = rcclx_dir / "defs.bzl"
    if not defs_file.exists():
        logger.warning(f"defs.bzl not found at {defs_file}, skipping compatibility fixes")
        return

    logger.info(f"Fixing defs.bzl API compatibility in {defs_file}")

    with open(defs_file, "r") as f:
        content = f.read()

    original_content = content

    # Fix: target_sdk_version was renamed to minimum_os_version
    # Old: target_sdk_version = base_toolchain.target_sdk_version,
    # New: minimum_os_version = base_toolchain.minimum_os_version,
    if "target_sdk_version = base_toolchain.target_sdk_version" in content:
        content = content.replace(
            "target_sdk_version = base_toolchain.target_sdk_version",
            "minimum_os_version = base_toolchain.minimum_os_version"
        )
        logger.info("  Fixed: target_sdk_version -> minimum_os_version")

    if content != original_content:
        with open(defs_file, "w") as f:
            f.write(content)
        logger.info(f"  Updated defs.bzl with API compatibility fixes")
    else:
        logger.info(f"  No API compatibility fixes needed in defs.bzl")


def _add_snapshot_compiler_flags(file_path: Path) -> None:
    """
    Add compiler flags to suppress warnings that were not errors in older snapshots.

    Older snapshot code may have patterns that trigger warnings with current compiler
    settings (e.g., -Wdeprecated-this-capture). Since these weren't errors when the
    code was written, we suppress them for snapshot builds.
    """
    if not file_path.exists():
        logger.warning(f"File not found, skipping compiler flag additions: {file_path}")
        return

    logger.info(f"Adding snapshot compiler flags to {file_path}")

    with open(file_path, "r") as f:
        content = f.read()

    # Check if flags are already added (check for one of our snapshot flags)
    if "-Wno-unused-exception-parameter" in content:
        logger.info(f"  Snapshot compiler flags already present in {file_path.name}")
        return

    # Flags to add for snapshot compatibility
    snapshot_flags = (
        '    # Snapshot compatibility: suppress warnings that were not errors in older code\n'
        '    "-Wno-deprecated-this-capture",\n'
        '    "-Wno-unused-exception-parameter",\n'
    )

    # Try multiple possible ending patterns for COMMON_COMPILER_FLAGS list
    # Different commit versions may have different last flags
    possible_endings = [
        '    "-Wno-pointer-bool-conversion",\n]',
        '    "-Wno-cuda-compat",\n]',
    ]

    replaced = False
    for old_pattern in possible_endings:
        if old_pattern in content:
            # Insert snapshot flags before the closing bracket
            new_pattern = old_pattern[:-1] + snapshot_flags + ']'
            content = content.replace(old_pattern, new_pattern)
            replaced = True
            with open(file_path, "w") as f:
                f.write(content)
            logger.info(f"  Added snapshot compatibility flags to COMMON_COMPILER_FLAGS in {file_path.name}")
            break

    if not replaced:
        logger.warning(f"  Could not find COMMON_COMPILER_FLAGS pattern in {file_path.name}")


def _add_snapshot_include_path(file_path: Path, stage: str) -> None:
    """
    Add include path flag so that "comms/rcclx/..." includes resolve within the snapshot.

    The original include paths like:
        #include "comms/rcclx/develop/meta/lib/CollTraceUtils.h"

    Need to resolve within the snapshot directory. By adding an include path
    pointing to the snapshot root, these paths will resolve correctly:
        -I fbcode/comms/rcclx/snapshots/{stage}

    Note: The path must be prefixed with "fbcode/" because Buck compilation runs
    from the fbsource root, but snapshot files are in fbcode/comms/rcclx/...

    This way "comms/rcclx/develop/..." resolves to
    "fbcode/comms/rcclx/snapshots/{stage}/comms/rcclx/develop/..."

    The include path is added to BOTH:
    - COMMON_PRE_COMPILER_FLAGS (used as preprocessor_flags in main rccl targets)
    - COMMON_COMPILER_FLAGS (used by auxiliary targets like colltrace_utils)

    Args:
        file_path: Path to the rccl_build_config.bzl file
        stage: The snapshot stage name ("stable" or "last-stable")
    """
    if not file_path.exists():
        logger.warning(f"File not found, skipping include path additions: {file_path}")
        return

    logger.info(f"Adding snapshot include path to {file_path}")

    with open(file_path, "r") as f:
        content = f.read()

    # Check if include path is already added
    snapshot_include_marker = f'# Snapshot include path for "{stage}"'
    if snapshot_include_marker in content:
        logger.info(f"  Snapshot include path already present in {file_path.name}")
        return

    # The include path flag to add - points to the snapshot root so
    # "comms/rcclx/develop/..." resolves correctly
    # Note: Must use "fbcode/" prefix because compilation runs from fbsource root
    include_path_flag = f'"-I", "fbcode/comms/rcclx/snapshots/{stage}"'

    modified = False

    # Add to COMMON_PRE_COMPILER_FLAGS (after -DBUILD_META_INTERNAL)
    old_pattern = '"-DBUILD_META_INTERNAL",'
    new_pattern = (
        '"-DBUILD_META_INTERNAL",\n'
        f'    {snapshot_include_marker}\n'
        f'    {include_path_flag},'
    )

    if old_pattern in content:
        content = content.replace(old_pattern, new_pattern)
        modified = True
        logger.info(f"  Added snapshot include path to COMMON_PRE_COMPILER_FLAGS")
    else:
        logger.warning(f"  Could not find COMMON_PRE_COMPILER_FLAGS pattern")

    # Also add to COMMON_COMPILER_FLAGS (at the beginning after [)
    # This is needed for auxiliary targets like colltrace_utils that only use COMMON_COMPILER_FLAGS
    # The include path should be added at the start of COMMON_COMPILER_FLAGS
    common_flags_pattern = 'COMMON_COMPILER_FLAGS = [\n    "-fPIC",'
    common_flags_replacement = (
        'COMMON_COMPILER_FLAGS = [\n'
        f'    {snapshot_include_marker}\n'
        f'    {include_path_flag},\n'
        '    "-fPIC",'
    )

    if common_flags_pattern in content:
        content = content.replace(common_flags_pattern, common_flags_replacement)
        modified = True
        logger.info(f"  Added snapshot include path to COMMON_COMPILER_FLAGS")
    else:
        logger.warning(f"  Could not find COMMON_COMPILER_FLAGS pattern")

    if modified:
        with open(file_path, "w") as f:
            f.write(content)


def _remove_manifold_targets_from_buck(rcclx_dir: Path) -> None:
    """
    Remove obsolete manifold-related targets from the snapshot BUCK file.

    Since we're using a pre-extracted sources approach that builds from source,
    we no longer need the manifold_get targets and prebuilt_cxx_library targets
    that download prebuilt artifacts from Manifold.

    This removes:
    - The manifold.bzl load statement
    - The stable_checksums.bzl load statement
    - All manifold_get targets
    - The prebuilt_cxx_library targets for rcclx-stable and rcclx-last-stable

    Args:
        rcclx_dir: Path to the rcclx directory within the snapshot
    """
    buck_file = rcclx_dir / "BUCK"
    if not buck_file.exists():
        logger.warning(f"BUCK file not found at {buck_file}, skipping manifold cleanup")
        return

    logger.info(f"Removing manifold targets from {buck_file}")

    with open(buck_file, 'r') as f:
        content = f.read()

    original_content = content

    # Remove the manifold.bzl load statement
    content = _remove_line_containing(content, 'load("@fbsource//tools/build_defs:manifold.bzl"')

    # Remove the stable_checksums.bzl load statement
    content = _remove_line_containing(content, 'load(":stable_checksums.bzl"')

    # Remove all manifold_get targets (they span multiple lines)
    content = _remove_buck_targets(content, 'manifold_get')

    # Remove prebuilt_cxx_library targets for rcclx-stable and rcclx-last-stable
    # (the ones that use manifold artifacts, not the rcclx-dev ones)
    content = _remove_prebuilt_stable_targets(content)

    # Remove comments about manifold artifacts
    content = _remove_manifold_comments(content)

    if content != original_content:
        with open(buck_file, 'w') as f:
            f.write(content)
        logger.info(f"  Cleaned up manifold targets from BUCK file")
    else:
        logger.info(f"  No manifold targets found to remove")


def _remove_line_containing(content: str, pattern: str) -> str:
    """Remove lines containing the given pattern."""
    lines = content.split('\n')
    filtered_lines = [line for line in lines if pattern not in line]
    return '\n'.join(filtered_lines)


def _remove_buck_targets(content: str, target_type: str) -> str:
    """Remove all Buck targets of a specific type from the content."""
    import re

    # Pattern to match a complete target definition:
    # target_type(
    #     ...
    # )
    # This handles nested parentheses properly
    pattern = rf'{target_type}\s*\('

    result = []
    i = 0
    lines = content.split('\n')

    while i < len(lines):
        line = lines[i]
        if re.search(pattern, line):
            # Found start of target, skip until we find the closing paren
            paren_count = line.count('(') - line.count(')')
            i += 1
            while i < len(lines) and paren_count > 0:
                paren_count += lines[i].count('(') - lines[i].count(')')
                i += 1
            # Skip any trailing empty lines
            while i < len(lines) and lines[i].strip() == '':
                i += 1
        else:
            result.append(line)
            i += 1

    return '\n'.join(result)


def _remove_prebuilt_stable_targets(content: str) -> str:
    """Remove prebuilt_cxx_library targets for rcclx-stable and rcclx-last-stable."""
    import re

    lines = content.split('\n')
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]
        # Check if this is a prebuilt_cxx_library target for stable/last-stable
        if 'fb_native.prebuilt_cxx_library(' in line:
            # Look ahead to check if this is a stable or last-stable target
            paren_count = line.count('(') - line.count(')')
            block_lines = [line]
            i += 1

            while i < len(lines) and paren_count > 0:
                block_lines.append(lines[i])
                paren_count += lines[i].count('(') - lines[i].count(')')
                i += 1

            # Check if this block contains rcclx-stable or rcclx-last-stable
            block_text = '\n'.join(block_lines)
            if re.search(r'name\s*=\s*"rcclx-stable"', block_text) or \
               re.search(r'name\s*=\s*"rcclx-last-stable"', block_text):
                # Skip this target (don't add to result)
                # Also skip trailing empty lines
                while i < len(lines) and lines[i].strip() == '':
                    i += 1
            else:
                # Keep this target
                result.extend(block_lines)
        else:
            result.append(line)
            i += 1

    return '\n'.join(result)


def _remove_manifold_comments(content: str) -> str:
    """Remove comments related to manifold artifacts."""
    lines = content.split('\n')
    result = []

    skip_patterns = [
        '# Manifold artifacts for',
        '# Download prebuilt rcclx library artifacts from Manifold',
        '# Checksums are automatically loaded from stable_checksums.bzl',
        '# ROCm 6.2',
        '# ROCm 6.4',
        '# ROCm 7.0',
        '# ROCm 6.2 last-stable',
        '# ROCm 6.4 last-stable',
        '# ROCm 7.0 last-stable',
        '# Stable snapshot library target - uses pre-built stable',
        '# This allows using validated stable builds',
        '# Headers come from the snapshot directories',
        '# Usage: buck2 build -m ovr_config//third-party/rocm',
        '# Last-stable snapshot library target',
        '# This provides a quick way to roll back',
    ]

    for line in lines:
        stripped = line.strip()
        should_skip = any(stripped.startswith(p) for p in skip_patterns)
        if not should_skip:
            result.append(line)

    return '\n'.join(result)


def write_metadata(
    metadata_path: Path,
    commit: str,
    repo_root: Path,
) -> None:
    """Write metadata file with commit hash and timestamp."""
    commit_info = get_commit_info(commit, repo_root)
    lines = commit_info.split("\n")

    full_hash = lines[0] if len(lines) > 0 else commit
    commit_date = lines[1] if len(lines) > 1 else "unknown"
    commit_desc = lines[2] if len(lines) > 2 else "unknown"

    metadata_content = f"""# RCCLX Snapshot Metadata
# This file records the source commit for this snapshot.
# The rcclx/ directory contains sources extracted from this commit.

commit: {full_hash}
commit_date: {commit_date}
commit_description: {commit_desc}
snapshot_created: {datetime.now().isoformat()}
created_by: create_snapshot.py
"""

    with open(metadata_path, "w") as f:
        f.write(metadata_content)

    logger.info(f"Wrote metadata to {metadata_path}")


def rotate_stable_to_last_stable(snapshots_root: Path) -> None:
    """
    Copy stable snapshot to last-stable.

    This preserves the previous stable snapshot before creating a new one.
    """
    stable_rcclx = snapshots_root / "stable" / "rcclx"
    stable_meta = snapshots_root / "stable" / "metadata.txt"
    last_stable_rcclx = snapshots_root / "last-stable" / "rcclx"
    last_stable_meta = snapshots_root / "last-stable" / "metadata.txt"

    if not stable_rcclx.exists():
        logger.warning("No stable snapshot exists, skipping rotation")
        return

    logger.info("Rotating: copying stable → last-stable...")

    # Remove existing last-stable
    if last_stable_rcclx.exists():
        logger.info(f"Removing existing last-stable: {last_stable_rcclx}")
        shutil.rmtree(last_stable_rcclx)

    # Copy stable to last-stable
    logger.info(f"Copying {stable_rcclx} → {last_stable_rcclx}")
    shutil.copytree(stable_rcclx, last_stable_rcclx)

    # Copy metadata
    if stable_meta.exists():
        shutil.copy(stable_meta, last_stable_meta)
        logger.info(f"Copied metadata to {last_stable_meta}")

    logger.info("Rotation complete")


def create_snapshot(
    stage: str,
    commit: str,
    snapshots_root: Path,
    repo_root: Path,
    rotate: bool = False,
) -> None:
    """
    Create a source snapshot of rcclx from a specific commit.

    Args:
        stage: "stable" or "last-stable"
        commit: Git/Sapling commit hash to snapshot from
        snapshots_root: Path to snapshots directory
        repo_root: Path to repo root (for sl commands)
        rotate: If True and stage is "stable", first copy stable to last-stable
    """
    logger.info("=" * 60)
    logger.info(f"Creating {stage} snapshot from commit {commit[:12]}...")
    logger.info("=" * 60)

    # Handle rotation: stable → last-stable
    if rotate and stage == "stable":
        rotate_stable_to_last_stable(snapshots_root)

    # Prepare destination paths
    stage_dir = snapshots_root / stage
    dest_rcclx = stage_dir / "rcclx"
    metadata_path = stage_dir / "metadata.txt"

    # Remove old snapshot if exists
    if dest_rcclx.exists():
        logger.info(f"Removing existing snapshot: {dest_rcclx}")
        shutil.rmtree(dest_rcclx)

    # Ensure stage directory exists
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Extract rcclx from specified commit
    extract_rcclx_from_commit(commit, stage_dir, repo_root, stage)

    # Write metadata
    write_metadata(metadata_path, commit, repo_root)

    # Run arc lint to fix any formatting issues
    run_arc_lint(stage_dir, repo_root)

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Successfully created {stage} snapshot")
    logger.info(f"  Sources: {stage_dir}/rcclx/")
    logger.info(f"  Metadata: {metadata_path}")
    logger.info("=" * 60)


def run_arc_lint(stage_dir: Path, repo_root: Path) -> None:
    """
    Run arc lint -a on all BUCK and .bzl files in the snapshot directory.

    This auto-fixes any formatting issues introduced during snapshot creation.

    Args:
        stage_dir: Path to the snapshot stage directory (e.g., snapshots/stable/)
        repo_root: Path to the repository root
    """
    rcclx_dir = stage_dir / "comms" / "rcclx"
    if not rcclx_dir.exists():
        logger.warning(f"rcclx directory not found at {rcclx_dir}, skipping arc lint")
        return

    # Find all BUCK and .bzl files in the snapshot
    files_to_lint = []
    for pattern in ["**/BUCK", "**/*.bzl"]:
        files_to_lint.extend(rcclx_dir.glob(pattern))

    if not files_to_lint:
        logger.info("No BUCK or .bzl files found to lint")
        return

    # Convert to paths relative to repo root for arc lint
    # arc lint expects paths relative to the repo root
    rel_files = []
    for f in files_to_lint:
        try:
            rel_path = f.relative_to(repo_root)
            rel_files.append(str(rel_path))
        except ValueError:
            # If file is not relative to repo_root, use absolute path
            rel_files.append(str(f.resolve()))

    logger.info(f"Running arc lint -a on {len(rel_files)} files in {rcclx_dir}")

    # Run arc lint -a on all the files
    # Split into batches if there are too many files
    batch_size = 50
    for i in range(0, len(rel_files), batch_size):
        batch = rel_files[i:i + batch_size]
        result = subprocess.run(
            ["arc", "lint", "-a"] + batch,
            cwd=repo_root,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            # arc lint may return non-zero even when it successfully applies fixes
            # Log the output but don't fail
            logger.warning(f"arc lint returned non-zero exit code: {result.returncode}")
            if result.stdout:
                logger.debug(f"arc lint stdout: {result.stdout[:500]}")
            if result.stderr:
                logger.debug(f"arc lint stderr: {result.stderr[:500]}")

    logger.info("arc lint completed")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a source snapshot of rcclx from a specific commit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Snapshot from specific commit
    python3 create_snapshot.py --stage stable --commit abc123def \\
        --snapshots-root fbcode/comms/rcclx/snapshots \\
        --repo-root /path/to/fbsource

    # Snapshot from current HEAD
    python3 create_snapshot.py --stage stable \\
        --snapshots-root fbcode/comms/rcclx/snapshots \\
        --repo-root /path/to/fbsource

    # Rotate stable → last-stable, then create new stable
    python3 create_snapshot.py --stage stable --commit abc123def --rotate \\
        --snapshots-root fbcode/comms/rcclx/snapshots \\
        --repo-root /path/to/fbsource
        """,
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["stable", "last-stable"],
        help="Snapshot stage: stable or last-stable",
    )
    parser.add_argument(
        "--commit",
        type=str,
        default=None,
        help="Commit hash to snapshot from (default: current HEAD)",
    )
    parser.add_argument(
        "--snapshots-root",
        type=Path,
        required=True,
        help="Path to snapshots directory",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        required=True,
        help="Path to repository root (for sl commands)",
    )
    parser.add_argument(
        "--rotate",
        action="store_true",
        help="If creating stable, first copy current stable to last-stable",
    )

    args = parser.parse_args()

    try:
        # Validate paths
        if not args.repo_root.exists():
            parser.error(f"Repository root does not exist: {args.repo_root}")

        # Resolve commit
        commit: str
        if args.commit:
            commit = args.commit
        else:
            logger.info("No commit specified, using current HEAD")
            commit = get_current_commit(args.repo_root)
            logger.info(f"Current HEAD: {commit[:12]}")

        # Validate rotation option
        if args.rotate and args.stage != "stable":
            parser.error("--rotate can only be used with --stage stable")

        # Create snapshot
        create_snapshot(
            stage=args.stage,
            commit=commit,
            snapshots_root=args.snapshots_root,
            repo_root=args.repo_root,
            rotate=args.rotate,
        )

        return 0

    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
