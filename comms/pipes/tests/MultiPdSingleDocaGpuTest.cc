// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// MultiPdSingleDocaGpuTest — GATING test for the multi-rail IBGDA design.
//
// Verifies that ONE doca_gpu* instance (single doca_gpu_create() call) can be
// used to create QP groups bound to MULTIPLE different ibv_pd* (one per NIC).
// This is the load-bearing assumption of the multi-rail design — without it,
// we must fall back to per-rail doca_gpu* (one doca_gpu_create per NIC).
//
// The test covers:
//   1. Single doca_gpu_create(GPU_PCI_BUS_ID)
//   2. Open 2 different mlx5_X devices: ibv_context + ibv_pd per NIC
//   3. Allocate ONE GPU buffer via cuMemCreate (matches transport's sink path)
//   4. Export dmabuf fd ONCE via export_gpu_dmabuf_aligned
//   5. Register the dmabuf with BOTH PDs via lazy_ibv_reg_dmabuf_mr
//      → verify both succeed and produce DIFFERENT lkeys
//   6. Create 2 QP groups via doca_gpu_verbs_create_qp_group_hl
//      with same gpu_dev (the shared docaGpu_) and DIFFERENT ibpd
//      → verify both succeed and produce DIFFERENT QPNs
//
// If this test fails on any platform, the multi-rail design must be
// re-architected to use per-rail doca_gpu* (matching the wrapper-draft
// pattern), and the failure must be reported to NVIDIA.

#include <gtest/gtest.h>

#include <cuda.h>
#include <doca_gpunetio_host.h>
#include <glog/logging.h>
#include <unistd.h>

#include <vector>

#include "comms/pipes/CudaDriverLazy.h"
#include "comms/pipes/DocaHostUtils.h"
#include "comms/pipes/IbverbsLazy.h"
#include "comms/pipes/rdma/NicDiscovery.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "doca_verbs_net_wrapper.h"

namespace comms::pipes::tests {

namespace {

constexpr int kCudaDevice = 0;
constexpr size_t kBufferSize = 4096; // one page; enough for MR registration
constexpr uint32_t kQpDepth = 64;
constexpr int kAccessFlags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
    IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;

// Allocates a GPU buffer via cuMemCreate with gpuDirectRDMACapable=1.
// Mirrors MultipeerIbgdaTransport::allocateResources() pattern so the buffer
// is registrable as an IB MR on aarch64/SMMU platforms (GB200/GB300).
struct GpuBuffer {
  void* ptr{nullptr};
  size_t size{0};
  size_t allocSize{0};
  CUmemGenericAllocationHandle handle{0};

  ~GpuBuffer() {
    if (ptr) {
      auto devPtr = reinterpret_cast<CUdeviceptr>(ptr);
      pfn_cuMemUnmap(devPtr, allocSize);
      pfn_cuMemAddressFree(devPtr, allocSize);
      pfn_cuMemRelease(handle);
    }
  }
};

GpuBuffer allocateGpuBuffer(int cudaDevice, size_t size) {
  GpuBuffer buf;
  buf.size = size;

  CUdevice cuDevice;
  EXPECT_EQ(pfn_cuDeviceGet(&cuDevice, cudaDevice), CUDA_SUCCESS);

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = cuDevice;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;

  int rdmaFlag = 0;
  pfn_cuDeviceGetAttribute(
      &rdmaFlag,
      CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
      cuDevice);
  if (rdmaFlag) {
    prop.allocFlags.gpuDirectRDMACapable = 1;
  }

  size_t granularity = 0;
  EXPECT_EQ(
      pfn_cuMemGetAllocationGranularity(
          &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
      CUDA_SUCCESS);
  buf.allocSize = ((size + granularity - 1) / granularity) * granularity;

  EXPECT_EQ(
      pfn_cuMemCreate(&buf.handle, buf.allocSize, &prop, 0), CUDA_SUCCESS);

  CUdeviceptr devPtr = 0;
  EXPECT_EQ(
      pfn_cuMemAddressReserve(&devPtr, buf.allocSize, granularity, 0, 0),
      CUDA_SUCCESS);
  EXPECT_EQ(
      pfn_cuMemMap(devPtr, buf.allocSize, 0, buf.handle, 0), CUDA_SUCCESS);

  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = cuDevice;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  EXPECT_EQ(
      pfn_cuMemSetAccess(devPtr, buf.allocSize, &accessDesc, 1), CUDA_SUCCESS);

  buf.ptr = reinterpret_cast<void*>(devPtr);
  EXPECT_EQ(cudaMemset(buf.ptr, 0, size), cudaSuccess);
  return buf;
}

// Resources for one NIC (mirror of the future RailContext shape).
struct NicResources {
  std::string deviceName;
  ibv_context* ctx{nullptr};
  ibv_pd* pd{nullptr};
  ibv_mr* mr{nullptr};
  doca_gpu_verbs_qp_group_hl* qpGroup{nullptr};

  ~NicResources() {
    if (qpGroup) {
      doca_gpu_verbs_destroy_qp_group_hl(qpGroup);
    }
    if (mr) {
      doca_verbs_wrapper_ibv_dereg_mr(mr);
    }
    if (pd) {
      doca_verbs_wrapper_ibv_dealloc_pd(pd);
    }
    if (ctx) {
      doca_verbs_wrapper_ibv_close_device(ctx);
    }
  }
};

} // namespace

class MultiPdSingleDocaGpuTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int deviceCount = 0;
    ASSERT_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);
    if (deviceCount < 1) {
      GTEST_SKIP() << "No CUDA devices";
    }
    ASSERT_EQ(cudaSetDevice(kCudaDevice), cudaSuccess);
    ASSERT_EQ(cuda_driver_lazy_init(), 0)
        << "CUDA driver API required for cuMemCreate";
  }
};

TEST_F(MultiPdSingleDocaGpuTest, SharedDocaGpuAcrossTwoPds) {
  // Step 0: discover at least 2 NICs accessible from this GPU.
  std::vector<NicCandidate> candidates;
  try {
    GpuNicDiscovery discovery(kCudaDevice);
    candidates = discovery.getCandidates();
  } catch (const std::exception& e) {
    GTEST_SKIP() << "NIC discovery failed: " << e.what();
  }
  if (candidates.size() < 2) {
    GTEST_SKIP() << "Test requires >= 2 RDMA NICs accessible from GPU "
                 << kCudaDevice << "; got " << candidates.size();
  }

  LOG(INFO) << "MultiPdSingleDocaGpuTest: GPU " << kCudaDevice << " has "
            << candidates.size() << " NIC candidates; using "
            << candidates[0].name << " and " << candidates[1].name;

  // Step 1: doca_gpu_create() ONCE — the shared GPU context.
  std::string gpuPciBusId = GpuNicDiscovery::getCudaPciBusId(kCudaDevice);
  doca_gpu* docaGpu = nullptr;
  ASSERT_EQ(doca_gpu_create(gpuPciBusId.c_str(), &docaGpu), DOCA_SUCCESS)
      << "Failed to create shared DOCA GPU context for PCIe " << gpuPciBusId;
  ASSERT_NE(docaGpu, nullptr);
  LOG(INFO) << "Shared DOCA GPU context created: " << (void*)docaGpu << " ("
            << gpuPciBusId << ")";

  // Step 2: open 2 different NICs — one ibv_context + ibv_pd per NIC.
  std::vector<NicResources> nics(2);
  int numIbDevices = 0;
  ibv_device** deviceList = nullptr;
  ASSERT_EQ(
      doca_verbs_wrapper_ibv_get_device_list(&numIbDevices, &deviceList),
      DOCA_SUCCESS);
  ASSERT_GT(numIbDevices, 0);

  for (int n = 0; n < 2; n++) {
    nics[n].deviceName = candidates[n].name;
    int idx = -1;
    for (int i = 0; i < numIbDevices; i++) {
      const char* devName = nullptr;
      doca_verbs_wrapper_ibv_get_device_name(deviceList[i], &devName);
      if (devName && nics[n].deviceName == devName) {
        idx = i;
        break;
      }
    }
    ASSERT_GE(idx, 0) << "NIC " << nics[n].deviceName
                      << " not found in ibv list";

    ASSERT_EQ(
        doca_verbs_wrapper_ibv_open_device(deviceList[idx], &nics[n].ctx),
        DOCA_SUCCESS);
    ASSERT_NE(nics[n].ctx, nullptr);

    ASSERT_EQ(
        doca_verbs_wrapper_ibv_alloc_pd(nics[n].ctx, &nics[n].pd),
        DOCA_SUCCESS);
    ASSERT_NE(nics[n].pd, nullptr);

    LOG(INFO) << "NIC " << nics[n].deviceName << ": ctx=" << (void*)nics[n].ctx
              << " pd=" << (void*)nics[n].pd;
  }
  doca_verbs_wrapper_ibv_free_device_list(deviceList);

  // Step 3: allocate ONE GPU buffer (cuMemCreate, gpuDirectRDMACapable=1).
  auto gpuBuf = allocateGpuBuffer(kCudaDevice, kBufferSize);
  ASSERT_NE(gpuBuf.ptr, nullptr);
  LOG(INFO) << "GPU buffer: ptr=" << gpuBuf.ptr << " size=" << gpuBuf.size;

  // Step 4: export dmabuf ONCE (PD-independent).
  auto dmabuf = export_gpu_dmabuf_aligned(docaGpu, gpuBuf.ptr, gpuBuf.size);
  ASSERT_TRUE(dmabuf.has_value())
      << "DMABUF export failed; cannot validate shared docaGpu_ across PDs "
      << "via DMABUF path. (Test could fall back to plain ibv_reg_mr but "
      << "DMABUF is the production path on aarch64.)";
  LOG(INFO) << "DMABUF exported: fd=" << dmabuf->fd
            << " alignedBase=" << dmabuf->alignment.alignedBase
            << " alignedSize=" << dmabuf->alignment.alignedSize
            << " dmabufOffset=" << dmabuf->alignment.dmabufOffset;

  // Step 5: register the dmabuf with BOTH PDs — verify both succeed with
  // DIFFERENT lkeys (PDs are independent).
  for (int n = 0; n < 2; n++) {
    nics[n].mr = lazy_ibv_reg_dmabuf_mr(
        nics[n].pd,
        dmabuf->alignment.dmabufOffset,
        gpuBuf.size,
        reinterpret_cast<uint64_t>(gpuBuf.ptr),
        dmabuf->fd,
        kAccessFlags);
    ASSERT_NE(nics[n].mr, nullptr)
        << "MR registration failed on NIC " << nics[n].deviceName
        << " (PD=" << (void*)nics[n].pd << ")";
    LOG(INFO) << "NIC " << nics[n].deviceName << " MR: " << (void*)nics[n].mr
              << " lkey=0x" << std::hex << nics[n].mr->lkey << " rkey=0x"
              << nics[n].mr->rkey << std::dec;
  }
  // Close fd once both registrations are complete (matches v6 design: fd is
  // PD-independent; dereg_mr keeps the kernel object alive).
  close(dmabuf->fd);

  // Sanity: lkeys should be distinct (different PDs).
  EXPECT_NE(nics[0].mr->lkey, nics[1].mr->lkey)
      << "Both PDs returned identical lkeys — unexpected (PDs are independent)";

  // Step 6: THE LOAD-BEARING TEST.
  // Create 2 QP groups via doca_gpu_verbs_create_qp_group_hl with the SAME
  // shared docaGpu and DIFFERENT ibpds. If DOCA rejects this pattern, the
  // multi-rail design must use per-rail doca_gpu_create.
  for (int n = 0; n < 2; n++) {
    doca_gpu_verbs_qp_init_attr_hl initAttr{};
    initAttr.gpu_dev = docaGpu; // SHARED across both QPs
    initAttr.ibpd = nics[n].pd; // Per-rail PD
    initAttr.sq_nwqe = kQpDepth;
    initAttr.nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO;
    initAttr.mreg_type = DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT;

    doca_error_t err =
        doca_gpu_verbs_create_qp_group_hl(&initAttr, &nics[n].qpGroup);
    ASSERT_EQ(err, DOCA_SUCCESS)
        << "doca_gpu_verbs_create_qp_group_hl rejected shared gpu_dev with "
        << "PD " << n << " (NIC " << nics[n].deviceName << "). "
        << "Multi-rail design requires per-rail doca_gpu_create instead.";
    ASSERT_NE(nics[n].qpGroup, nullptr);

    LOG(INFO) << "NIC " << nics[n].deviceName << " QP group: main_qpn="
              << doca_verbs_qp_get_qpn(nics[n].qpGroup->qp_main.qp)
              << " companion_qpn="
              << doca_verbs_qp_get_qpn(nics[n].qpGroup->qp_companion.qp);
  }

  // Sanity: QPNs should differ (different PDs → different QPs).
  EXPECT_NE(
      doca_verbs_qp_get_qpn(nics[0].qpGroup->qp_main.qp),
      doca_verbs_qp_get_qpn(nics[1].qpGroup->qp_main.qp))
      << "Both PDs returned identical QPNs — unexpected";

  // Cleanup happens via NicResources / GpuBuffer destructors (RAII).
  doca_gpu_destroy(docaGpu);
  LOG(INFO)
      << "MultiPdSingleDocaGpuTest passed: shared docaGpu works across PDs.";
}

} // namespace comms::pipes::tests
