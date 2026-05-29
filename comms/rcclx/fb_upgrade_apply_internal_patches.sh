#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

THIRD_PARTY_RCCL_ROOT="${SCRIPT_DIR}"
VERSION="${1:-develop}"

echo "Changing directory to ${THIRD_PARTY_RCCL_ROOT} to apply internal patches"
cd ${THIRD_PARTY_RCCL_ROOT}

if [[ -n $(hg status) ]]; then
    echo "ERROR: Uncommitted files in your repository when attempting to patch"
    hg status
    exit 1
fi

echo "Applying internal patches"
commit_msg="[rccl/${VERSION}] Apply internal patches"
rejects_file=$(mktemp)
failed_patches=""
applied_patches=""
applied_patch_count=0
internal_patches=$(find "patches/" -name '*.patch')
for internal_patch in ${internal_patches}; do
    if sl patch --no-commit $internal_patch \
            |& tee /dev/stderr \
            | awk '/saving rejects to file/ {print $NF}' \
            | tee -a "${rejects_file}"; then
        applied_patches+="${internal_patch} "
        ((++applied_patch_count))
        hg status
        hg commit -m "${commit_msg}"
    else
        failed_patches+="${internal_patch} "
    fi
done

if (( applied_patch_count > 1 )); then
    echo "Squashing applied patches"
    hg fold --from .~$((applied_patch_count-1)) -m "${commit_msg}"
fi

# If any failed patches are detected, then echo and fail
if [ -n "${failed_patches}" ]; then
    echo
    echo "* * * * * ERROR APPLYING INTERNAL PATCHES * * * * *"
    echo
    echo "One or more patches failed to apply"
    echo "Review the patches for the rejects (.rej) files and fix or remote the patch"
    echo "Discarfd any changes not yet committed and rerun the script to apply patches"
    echo "To reapply after discarding, use (version defaults to develop):"
    echo
    echo "${THIRD_PARTY_RCCL_ROOT}/fb_upgrade_apply_internal_patches.sh [version]"
    echo
    echo "Examples:"
    echo "  ${THIRD_PARTY_RCCL_ROOT}/fb_upgrade_apply_internal_patches.sh"
    echo "  ${THIRD_PARTY_RCCL_ROOT}/fb_upgrade_apply_internal_patches.sh develop"
    echo
    echo "Successfully applied patch files:"
    for applied_patch in $applied_patches; do
        echo "  ${applied_patch}"
    done
    echo
    echo "Failed patch files:"
    for failed_patch in $failed_patches; do
        echo "  ${failed_patch}"
    done
    echo
    echo "Reject files:"
    for reject_file in $(cat ${rejects_file}); do
        echo "  ${reject_file}"
    done
    echo
    if (( applied_patch_count > 0 )); then
        echo "Uncommitting successfully applied patches"
        hg uncommit
    fi
    exit 1
fi

echo "All patches successfully applied: ${internal_patches}"
