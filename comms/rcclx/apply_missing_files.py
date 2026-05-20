#!/usr/bin/env python3
"""
Apply missing files from rccl_feb_drop_modified.diff.
- NEW files: applied completely (all content is in + lines)
- MODIFIED files missing from disk: skipped (can't reliably reconstruct
  from a modification diff without the original file content)
- Binary files: skipped (can't restore from text diff)
- Files that already exist: skipped
"""
import os
import sys
import re

DIFF_FILE = os.path.expanduser("~/tmp/rccl_feb_drop_modified.diff")
REPO_ROOT = "/data/users/srinathb/fbsource"
SCOPE = "fbcode/comms/rcclx/"

BINARY_EXTS = {'.hsaco', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg',
               '.so', '.a', '.o', '.pyc', '.pdf', '.zip', '.tar', '.gz',
               '.bz2', '.xz', '.whl', '.bin', '.exe', '.dll', '.dylib'}

def is_binary(filepath):
    ext = '.' + filepath.rsplit('.', 1)[-1] if '.' in filepath.rsplit('/', 1)[-1] else ''
    return ext.lower() in BINARY_EXTS

def apply_missing_files(diff_path, repo_root, scope):
    created = []
    skipped_exists = []
    skipped_binary = []
    skipped_modified_missing = []
    failed = []

    current_file = None
    current_status = None  # 'new', 'modified', 'deleted'
    current_lines = []
    in_content = False

    def flush_file():
        if current_file and current_file.startswith(scope) and current_status == 'new':
            full_path = os.path.join(repo_root, current_file)
            if os.path.exists(full_path):
                skipped_exists.append(current_file)
            elif is_binary(current_file):
                skipped_binary.append(current_file)
            else:
                try:
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)
                    content = '\n'.join(current_lines)
                    # Preserve trailing newline if lines are non-empty
                    if current_lines:
                        content += '\n'
                    with open(full_path, 'w', errors='replace') as f:
                        f.write(content)
                    created.append(current_file)
                except Exception as e:
                    failed.append((current_file, str(e)))
        elif current_file and current_file.startswith(scope) and current_status == 'modified':
            full_path = os.path.join(repo_root, current_file)
            if not os.path.exists(full_path):
                skipped_modified_missing.append(current_file)

    print(f"Processing diff file (this will take several minutes for a 14M line file)...")
    sys.stdout.flush()

    with open(diff_path, 'r', errors='replace') as f:
        for lineno, line in enumerate(f):
            if lineno % 1000000 == 0 and lineno > 0:
                print(f"  {lineno//1000000}M lines processed, {len(created)} files created...")
                sys.stdout.flush()

            line = line.rstrip('\n')

            if line.startswith('diff --git '):
                flush_file()
                m = re.match(r'diff --git a/(.+) b/(.+)', line)
                if m:
                    current_file = m.group(2)
                    current_status = 'modified'
                    current_lines = []
                    in_content = False
                else:
                    current_file = None
                continue

            if current_file is None:
                continue

            if line.startswith('new file mode'):
                current_status = 'new'
            elif line.startswith('deleted file mode'):
                current_status = 'deleted'
            elif line.startswith('--- ') or line.startswith('+++ '):
                pass
            elif line.startswith('@@ '):
                in_content = True
            elif in_content and current_status == 'new':
                if line.startswith('+'):
                    current_lines.append(line[1:])
                elif line.startswith('\\'):
                    pass  # "No newline at end of file"

    flush_file()  # Handle last file

    return created, skipped_exists, skipped_binary, skipped_modified_missing, failed

def main():
    print(f"Applying missing NEW files from diff to {REPO_ROOT}/{SCOPE}")
    print("=" * 70)

    created, skipped_exists, skipped_binary, skipped_modified_missing, failed = \
        apply_missing_files(DIFF_FILE, REPO_ROOT, SCOPE)

    print(f"\n{'='*70}")
    print(f"RESULTS:")
    print(f"  Created (new files applied):              {len(created)}")
    print(f"  Skipped (already exist on disk):          {len(skipped_exists)}")
    print(f"  Skipped (binary, can't restore):          {len(skipped_binary)}")
    print(f"  Skipped (modified-but-missing, need full  {len(skipped_modified_missing)}")
    print(f"           upstream sync to restore):       ")
    print(f"  Failed:                                   {len(failed)}")

    if created:
        print(f"\nCreated files ({len(created)}):")
        for f in sorted(created)[:50]:
            print(f"  + {f.replace('fbcode/comms/rcclx/', '')}")
        if len(created) > 50:
            print(f"  ... and {len(created)-50} more (see /tmp/created_files.txt)")
        with open('/tmp/created_files.txt', 'w') as out:
            for f in sorted(created):
                out.write(f + '\n')

    if failed:
        print(f"\nFailed ({len(failed)}):")
        for f, e in failed:
            print(f"  ! {f}: {e}")

    if skipped_binary:
        print(f"\nBinary files skipped ({len(skipped_binary)}) - need direct upstream sync:")
        for f in skipped_binary[:5]:
            print(f"  - {f.replace('fbcode/comms/rcclx/', '')}")
        if len(skipped_binary) > 5:
            print(f"  ... and {len(skipped_binary)-5} more")

    if skipped_modified_missing:
        print(f"\nModified-but-missing files ({len(skipped_modified_missing)}) - need full upstream sync:")
        for f in skipped_modified_missing[:10]:
            print(f"  - {f.replace('fbcode/comms/rcclx/', '')}")
        if len(skipped_modified_missing) > 10:
            print(f"  ... and {len(skipped_modified_missing)-10} more")
        with open('/tmp/modified_missing_files.txt', 'w') as out:
            for f in sorted(skipped_modified_missing):
                out.write(f + '\n')
        print(f"  Full list saved to /tmp/modified_missing_files.txt")

    return len(failed)

if __name__ == '__main__':
    sys.exit(main())
