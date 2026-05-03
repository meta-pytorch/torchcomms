#!/bin/bash
# Audit script: for each file changed in rcclx feb drop (4acd7e79b385),
# check if OLD rcclx had customizations vs upstream rccl that are now lost.
#
# A file has "lost customizations" if:
#   diff(OLD_RCCLX, NEW_RCCL) is non-empty  (old had something different from upstream)
# AND
#   diff(NEW_RCCLX, NEW_RCCL) is empty      (new rcclx is now identical to upstream)

cd /data/users/srinathb/fbsource

OLD_RCCLX_REV="parents(4acd7e79b385)"
NEW_RCCLX_REV="4acd7e79b385"
NEW_RCCL_REV="cb3e41acd299"

RCCLX_SRC_PREFIX="fbcode/comms/rcclx"
RCCL_SRC_PREFIX="third-party/rccl"
SRC_SUBPATH="develop/projects/rccl/src"

echo "======================================================"
echo "RCCLX FEB DROP CUSTOMIZATION AUDIT"
echo "OLD rcclx: parent of 4acd7e79b385"
echo "NEW rcclx: 4acd7e79b385"
echo "NEW rccl:  cb3e41acd299"
echo "======================================================"
echo ""
echo "Legend:"
echo "  [LOST]    Old rcclx had customizations, new rcclx lost them (overwritten)"
echo "  [CUSTOM]  Old rcclx had customizations, new rcclx still has them (good)"
echo "  [NEW]     New rcclx has customizations not in old rcclx (added in merge)"
echo "  [CLEAN]   File was same as upstream in old rcclx (no customizations to lose)"
echo ""

LOST_FILES=()
CUSTOM_FILES=()
NEW_CUSTOM_FILES=()
CLEAN_FILES=()

FILES=(
  "collectives.cc"
  "enqueue.cc"
  "rccl_wrap.cc"
  "init.cc"
  "graph/connect.cc"
  "graph/tuning.cc"
  "graph/topo.cc"
  "graph/topo.h"
  "graph/paths.cc"
  "graph/xml.cc"
  "graph/xml.h"
  "include/comm.h"
  "include/collectives.h"
  "include/rccl_common.h"
  "include/device.h"
  "include/core.h"
  "include/nccl_common.h"
  "include/channel.h"
  "include/graph.h"
  "include/scheduler.h"
  "include/proxy.h"
  "include/transport.h"
  "include/comm.h"
  "device/all_gather.h"
  "device/all_reduce.h"
  "device/reduce_scatter.h"
  "device/broadcast.h"
  "device/reduce.h"
  "device/reduce_kernel.h"
  "device/common.h"
  "device/common.cu"
  "device/prims_simple.h"
  "device/prims_ll.h"
  "device/prims_ll128.h"
  "device/msccl_kernel_impl.h"
  "proxy.cc"
  "group.cc"
  "allocator.cc"
  "bootstrap.cc"
  "transport/p2p.cc"
  "transport/net.cc"
  "transport/net_ib.cc"
  "transport/generic.cc"
  "transport/coll_net.cc"
  "transport/nvls.cc"
  "misc/socket.cc"
  "misc/utils.cc"
  "misc/api_trace.cc"
  "misc/msccl/msccl_lifecycle.cc"
  "misc/msccl/msccl_parser.cc"
  "misc/rocm_smi_wrap.cc"
  "misc/npkit.cc"
  "misc/shmutils.cc"
)

for f in "${FILES[@]}"; do
  rcclx_old_file="$RCCLX_SRC_PREFIX/$SRC_SUBPATH/$f"
  rcclx_new_file="$RCCLX_SRC_PREFIX/$SRC_SUBPATH/$f"
  rccl_new_file="$RCCL_SRC_PREFIX/$SRC_SUBPATH/$f"

  old_vs_rccl=$(sl diff -r "$OLD_RCCLX_REV" -r "$NEW_RCCL_REV" -- "$rcclx_old_file" "$rccl_new_file" 2>/dev/null | grep -v "^diff\|^---\|^+++\|^@@\|^Binary" | wc -l)
  new_vs_rccl=$(diff <(sl cat -r "$NEW_RCCLX_REV" "$rcclx_new_file" 2>/dev/null) <(sl cat -r "$NEW_RCCL_REV" "$rccl_new_file" 2>/dev/null) 2>/dev/null | wc -l)

  # Simple approach: compare old rcclx vs new rccl, and new rcclx vs new rccl
  old_diff=$(diff <(sl cat -r "$OLD_RCCLX_REV" "$rcclx_old_file" 2>/dev/null) \
                  <(sl cat -r "$NEW_RCCL_REV" "$rccl_new_file" 2>/dev/null) 2>/dev/null | wc -l)
  new_diff=$(diff <(sl cat -r "$NEW_RCCLX_REV" "$rcclx_new_file" 2>/dev/null) \
                  <(sl cat -r "$NEW_RCCL_REV" "$rccl_new_file" 2>/dev/null) 2>/dev/null | wc -l)

  if [ "$old_diff" -gt 0 ] && [ "$new_diff" -eq 0 ]; then
    echo "[LOST]   $f  (old had $old_diff diff lines vs upstream, now identical to upstream)"
    LOST_FILES+=("$f|$old_diff")
  elif [ "$old_diff" -gt 0 ] && [ "$new_diff" -gt 0 ]; then
    echo "[CUSTOM] $f  (old: $old_diff diff lines, new: $new_diff diff lines vs upstream)"
    CUSTOM_FILES+=("$f|$old_diff|$new_diff")
  elif [ "$old_diff" -eq 0 ] && [ "$new_diff" -gt 0 ]; then
    echo "[NEW]    $f  (new rcclx added $new_diff diff lines vs upstream)"
    NEW_CUSTOM_FILES+=("$f|$new_diff")
  else
    echo "[CLEAN]  $f"
    CLEAN_FILES+=("$f")
  fi
done

echo ""
echo "======================================================"
echo "SUMMARY"
echo "======================================================"
echo "Files with LOST customizations: ${#LOST_FILES[@]}"
for f in "${LOST_FILES[@]}"; do echo "  - ${f%%|*}"; done
echo ""
echo "Files with SURVIVING customizations: ${#CUSTOM_FILES[@]}"
for f in "${CUSTOM_FILES[@]}"; do echo "  - ${f%%|*}"; done
echo ""
echo "Files with NEW customizations (added in merge): ${#NEW_CUSTOM_FILES[@]}"
echo ""
echo "Files with no customizations (clean upstream): ${#CLEAN_FILES[@]}"
