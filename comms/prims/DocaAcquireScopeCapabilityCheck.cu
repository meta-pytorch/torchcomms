// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Fails the build if the vendored DOCA headers lose the capability macro that
// P2pIbgdaTransportDevice.cuh feature-detects on.
//
// Patch 0002 adds both DOCA_GPUNETIO_VERBS_META_HAS_ACQUIRE_SCOPE and the
// acquire_scope template parameter. Re-applying it after a resync without the
// macro hunk leaves a green build, a passing test suite and correct results,
// with every fence silently back on SYS and the whole saving gone.
//
// This lives in its own target, which depends only on :doca_gpunetio_dl, so it
// always sees the vendored headers. The equivalent check cannot go in
// P2pIbgdaTransportDevice.cuh: that file is also compiled in builds where ncclx
// shadows these headers and the macro is legitimately absent, which is exactly
// what the feature-detect exists to tolerate.

#include <device/doca_gpunetio_dev_verbs_cq.cuh>

#ifndef DOCA_GPUNETIO_VERBS_META_HAS_ACQUIRE_SCOPE
#error \
    "The vendored DOCA headers no longer define DOCA_GPUNETIO_VERBS_META_HAS_ACQUIRE_SCOPE. prims silently falls back to the SYS-scope fence and the CQ-fence optimization is inert. This is what an incomplete re-apply of third-party/nvidia-doca/patches/0002-gpunetio-cq-fence-cta.patch looks like."
#endif

static_assert(
    DOCA_GPUNETIO_VERBS_META_HAS_ACQUIRE_SCOPE == 1,
    "DOCA_GPUNETIO_VERBS_META_HAS_ACQUIRE_SCOPE must be 1 where it is defined.");

// The macro and the parameter must travel together: instantiating with an
// explicit acquire_scope fails to compile if the parameter hunk was dropped.
template <enum doca_gpu_dev_verbs_sync_scope scope>
__device__ int doca_acquire_scope_capability_probe(
    struct doca_gpu_dev_verbs_cq* cq,
    uint64_t ticket) {
  return doca_gpu_dev_verbs_poll_one_cq_at<
      DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
      DOCA_GPUNETIO_VERBS_QP_SQ,
      scope>(cq, ticket);
}

template __device__ int
doca_acquire_scope_capability_probe<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_CTA>(
    struct doca_gpu_dev_verbs_cq*,
    uint64_t);
