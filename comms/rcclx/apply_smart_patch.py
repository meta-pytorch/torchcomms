#!/usr/bin/env python3
"""
Smart patch applier for the rcclx AMD February drop.

Parses a unified diff and handles five cases:
  CREATE  (--- /dev/null)     : writes file content directly, bypassing patch's
                                 silent-skip when the destination already exists.
  MODIFY  (--- a/ +++ b/)     : applies hunk via `patch -p1`, pauses on conflict.
  DELETE  (+++ /dev/null)     : removes file from disk.
  RENAME  (rename from/to)    : moves file; handles Phase-1 path divergence.
  EMPTY-DELETE (deleted file mode, no hunks): removes file.

Usage:
  python3 /tmp/apply_smart_patch.py ~/tmp/rccl_feb_drop_modified.diff [--repo-root PATH]
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT_DEFAULT = Path("/data/users/srinathb/fbsource")


# ── patch splitting ───────────────────────────────────────────────────────────

def split_into_file_patches(diff_text: str) -> list[str]:
    parts = re.split(r"(?=^diff --git )", diff_text, flags=re.MULTILINE)
    return [p for p in parts if p.strip()]


# ── patch classification ──────────────────────────────────────────────────────

def classify(patch: str) -> str:
    """
    Return one of: 'create', 'delete', 'rename', 'empty_delete', 'modify', 'skip'
    """
    lines = patch.splitlines()

    # No diff --git header → commit metadata block, safe to skip
    if not any(l.startswith("diff --git ") for l in lines):
        return "skip"

    has_minus3 = any(l.startswith("--- ") for l in lines)
    has_plus3  = any(l.startswith("+++ ") for l in lines)
    has_rename = any(l.startswith("rename from ") for l in lines)
    has_deleted_mode = any("deleted file mode" in l for l in lines)
    has_hunks  = any(l.startswith("@@") for l in lines)

    if has_rename and not has_hunks:
        return "rename"
    if has_minus3 and any(l == "--- /dev/null" for l in lines):
        return "create"
    if has_plus3 and any(l == "+++ /dev/null" for l in lines):
        return "delete"
    if has_deleted_mode and not has_minus3 and not has_plus3:
        return "empty_delete"
    if is_binary_patch(patch):
        return "binary"
    if has_minus3 and has_plus3:
        return "modify"

    return "skip"


def is_binary_patch(patch: str) -> bool:
    return bool(re.search(r"^Binary files ", patch, re.MULTILINE))


# ── path extraction ───────────────────────────────────────────────────────────

def target_path(patch: str) -> str | None:
    """+++ b/<path>"""
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            return line[6:]
        if line.startswith("+++ ") and not line.startswith("+++ /dev/null"):
            return line[4:].lstrip("/")
    return None


def source_path(patch: str) -> str | None:
    """--- a/<path>"""
    for line in patch.splitlines():
        if line.startswith("--- a/"):
            return line[6:]
        if line.startswith("--- ") and not line.startswith("--- /dev/null"):
            return line[4:].lstrip("/")
    return None


def git_header_paths(patch: str) -> tuple[str | None, str | None]:
    """Extract (a_path, b_path) from 'diff --git a/X b/Y'."""
    m = re.search(r"^diff --git a/(\S+) b/(\S+)", patch, re.MULTILINE)
    if m:
        return m.group(1), m.group(2)
    return None, None


def rename_paths(patch: str) -> tuple[str | None, str | None]:
    """Return (rename_from, rename_to) paths."""
    frm = to = None
    for line in patch.splitlines():
        if line.startswith("rename from "):
            frm = line[len("rename from "):]
        elif line.startswith("rename to "):
            to = line[len("rename to "):]
    return frm, to


def file_mode(patch: str) -> int | None:
    m = re.search(r"^new file mode (\d+)", patch, re.MULTILINE)
    return int(m.group(1), 8) if m else None


# ── content extraction ────────────────────────────────────────────────────────

def extract_created_content(patch: str) -> tuple[str, bool]:
    lines: list[str] = []
    no_newline = False
    in_hunk = False
    for line in patch.splitlines():
        if line.startswith("@@"):
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith("\\ "):
            no_newline = True
            continue
        if line.startswith("+"):
            lines.append(line[1:])
    content = "\n".join(lines)
    if not no_newline:
        content += "\n"
    return content, not no_newline


# ── patch application ─────────────────────────────────────────────────────────

def apply_modify_patch(patch_text: str, repo_root: Path) -> tuple[bool, list[str]]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
        f.write(patch_text)
        tmp = f.name
    try:
        result = subprocess.run(
            ["patch", "--batch", "-p1", "-i", tmp],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        rej_files = re.findall(r"saving rejects to file (\S+)", result.stdout)
        success = result.returncode == 0 and not rej_files
        if not success and not rej_files:
            print(f"    patch stderr: {result.stderr[:400]}")
        return success, rej_files
    finally:
        os.unlink(tmp)


# ── conflict prompt ───────────────────────────────────────────────────────────

def prompt_after_conflict(rel_path: str, rej_files: list[str]) -> str:
    print()
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │  CONFLICT — manual resolution required                  │")
    print("  └─────────────────────────────────────────────────────────┘")
    print(f"  File   : {rel_path}")
    for r in rej_files:
        print(f"  Reject : {r}")
    print()
    print("  Steps:")
    print(f"    1. Edit {rel_path} to apply the rejected hunks from the .rej file")
    print(f"    2. Delete the .rej file when done")
    print()
    while True:
        ans = input("  [c]ontinue / [s]kip this file / [q]uit → ").strip().lower()
        if ans in ("c", "s", "q"):
            return {"c": "continue", "s": "skip", "q": "quit"}[ans]
        print("  Please enter c, s, or q.")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("diff_file", type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT_DEFAULT)
    args = parser.parse_args()

    diff_path = args.diff_file.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()

    if not diff_path.exists():
        sys.exit(f"Error: diff file not found: {diff_path}")
    if not repo_root.exists():
        sys.exit(f"Error: repo root not found: {repo_root}")

    diff_text = diff_path.read_text(errors="replace")
    patches = split_into_file_patches(diff_text)
    total = len(patches)
    print(f"Parsed {total} file patches from {diff_path.name}\n")

    counts = dict(created=0, modified=0, deleted=0, renamed=0,
                  conflicts=0, skipped=0, binary=0)
    conflict_files: list[str] = []

    for idx, patch in enumerate(patches, 1):
        ptype = classify(patch)
        prefix = f"[{idx:5d}/{total}]"

        # ── skip (commit header blocks) ───────────────────────────────────
        if ptype == "skip":
            # Completely silent — these are just HG commit metadata headers
            counts["skipped"] += 1
            continue

        # ── binary ────────────────────────────────────────────────────────
        if ptype == "binary":
            _, b_path = git_header_paths(patch)
            label = b_path or "?"
            print(f"{prefix} BINARY   (skipped)         : {label}")
            counts["binary"] += 1
            continue

        # ── create ────────────────────────────────────────────────────────
        if ptype == "create":
            tgt = target_path(patch)
            if tgt is None:
                print(f"{prefix} CREATE   (no path, skipped)")
                counts["skipped"] += 1
                continue
            dest = repo_root / tgt
            existed = dest.exists()
            dest.parent.mkdir(parents=True, exist_ok=True)
            content, _ = extract_created_content(patch)
            dest.write_text(content, errors="replace")
            mode = file_mode(patch)
            if mode is not None:
                dest.chmod(mode)
            tag = "overwrite" if existed else "new file "
            print(f"{prefix} CREATE   ({tag})        : {tgt}")
            counts["created"] += 1
            continue

        # ── delete (+++ /dev/null) ────────────────────────────────────────
        if ptype == "delete":
            src = source_path(patch)
            if src is None:
                print(f"{prefix} DELETE   (no path, skipped)")
                counts["skipped"] += 1
                continue
            target_abs = repo_root / src
            if target_abs.exists():
                target_abs.unlink()
                print(f"{prefix} DELETE                     : {src}")
                counts["deleted"] += 1
            else:
                print(f"{prefix} DELETE   (already absent)  : {src}")
                counts["skipped"] += 1
            continue

        # ── empty_delete (deleted file mode, no content) ─────────────────
        if ptype == "empty_delete":
            a_path, _ = git_header_paths(patch)
            if a_path is None:
                print(f"{prefix} EMPTY-DEL(no path, skipped)")
                counts["skipped"] += 1
                continue
            target_abs = repo_root / a_path
            if target_abs.exists():
                target_abs.unlink()
                print(f"{prefix} EMPTY-DEL                  : {a_path}")
                counts["deleted"] += 1
            else:
                print(f"{prefix} EMPTY-DEL(already absent)  : {a_path}")
                counts["skipped"] += 1
            continue

        # ── rename (pure, no content change) ─────────────────────────────
        if ptype == "rename":
            frm, to = rename_paths(patch)
            if frm is None or to is None:
                print(f"{prefix} RENAME   (no paths, skipped)")
                counts["skipped"] += 1
                continue

            src_abs  = repo_root / frm
            dest_abs = repo_root / to

            if dest_abs.exists():
                # Already at destination (Phase 1 or prior run)
                print(f"{prefix} RENAME   (dest exists, skip): {frm} → {to}")
                counts["skipped"] += 1
            elif src_abs.exists():
                # Source still at original location — do the move
                dest_abs.parent.mkdir(parents=True, exist_ok=True)
                src_abs.rename(dest_abs)
                print(f"{prefix} RENAME                     : {frm} → {to}")
                counts["renamed"] += 1
            else:
                # Source gone (Phase 1 moved it somewhere else).
                # Try to find the filename under rcclx to give a useful hint.
                filename = Path(frm).name
                hits = list((repo_root / "fbcode/comms/rcclx").rglob(filename))
                if hits:
                    hint = " | found at: " + str(hits[0].relative_to(repo_root))
                else:
                    hint = " | not found in rcclx"
                print(f"{prefix} RENAME   (src absent{hint}): {frm} → {to}")
                counts["skipped"] += 1
            continue

        # ── modify ────────────────────────────────────────────────────────
        tgt = target_path(patch)
        if tgt is None:
            # Last-resort: extract from diff --git header
            _, b_path = git_header_paths(patch)
            tgt = b_path
        if tgt is None:
            print(f"{prefix} MODIFY   (no path, skipped)")
            counts["skipped"] += 1
            continue

        target_abs = repo_root / tgt
        if not target_abs.exists():
            print(f"{prefix} MODIFY   (file absent, skip): {tgt}")
            counts["skipped"] += 1
            continue

        success, rej_files = apply_modify_patch(patch, repo_root)
        if success:
            print(f"{prefix} MODIFY                     : {tgt}")
            counts["modified"] += 1
        else:
            print(f"{prefix} MODIFY   CONFLICT          : {tgt}")
            counts["conflicts"] += 1
            conflict_files.append(tgt)
            decision = prompt_after_conflict(tgt, rej_files)
            if decision == "quit":
                print("\nAborted by user.")
                break
            if decision == "skip":
                counts["skipped"] += 1

    # ── summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  Created   (new + overwrites) : {counts['created']}")
    print(f"  Modified  (clean)            : {counts['modified']}")
    print(f"  Deleted                      : {counts['deleted']}")
    print(f"  Renamed                      : {counts['renamed']}")
    print(f"  Conflicts (needs resolve)    : {counts['conflicts']}")
    print(f"  Skipped                      : {counts['skipped']}")
    print(f"  Binary    (skipped)          : {counts['binary']}")
    if conflict_files:
        print()
        print("  Files needing resolution:")
        for f in conflict_files:
            print(f"    {f}.rej")
    print("=" * 65)
    sys.exit(1 if counts["conflicts"] > 0 else 0)


if __name__ == "__main__":
    main()
