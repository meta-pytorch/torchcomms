// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/Ctran.h"
#include "comms/ctran/commstate/CommStateX.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsCuUtils.h"
#include "comms/testinfra/TestsDistUtils.h"

using namespace ncclx;

template <commDataType_t>
struct CommTypeTraits;

template <>
struct CommTypeTraits<commInt8> {
  using T = int8_t;
};

template <>
struct CommTypeTraits<commHalf> {
  using T = int16_t;
};

template <>
struct CommTypeTraits<commInt32> {
  using T = int32_t;
};

template <>
struct CommTypeTraits<commInt64> {
  using T = int64_t;
};

class CtranBaseTest {
 private:
  bool isBackendValid(
      const std::vector<CtranMapperBackend>& excludedBackends,
      CtranMapperBackend backend) {
    return std::find(
               excludedBackends.begin(), excludedBackends.end(), backend) ==
        excludedBackends.end();
  }
  size_t pageSize_{0};

 public:
  CtranBaseTest() {
    pageSize_ = getpagesize();
  }

  // Check no GPE internal memory leak after finished collective kernel
  void verifyGpeLeak(ICtran* ctran) {
    ASSERT_EQ(ctran->gpe->numInUseKernelElems(), 0);
    ASSERT_EQ(ctran->gpe->numInUseKernelFlags(), 0);
  }

  void resetBackendsUsed(ICtran* ctran) {
    ctran->mapper->iPutCount[CtranMapperBackend::NVL] = 0;
    ctran->mapper->iPutCount[CtranMapperBackend::IB] = 0;
  }

  // Wrapper of verifyBackendsUsed with empty excludedBackends
  void verifyBackendsUsed(
      ICtran* ctran,
      const CommStateX* statex,
      MemAllocType memType) {
    verifyBackendsUsed(ctran, statex, memType, {});
  }

  // Verify the traffic at each backend is expected based on the memType and
  // statex topo. Collective algorithm can exclude certain backends from being
  // used (e.g., Alltoall(v) doesn't use NVL backend).
  void verifyBackendsUsed(
      ICtran* ctran,
      const CommStateX* statex,
      MemAllocType memType,
      const std::vector<CtranMapperBackend>& excludedBackends) {
    const int nRanks = statex->nRanks();
    const int nLocalRanks = statex->nLocalRanks();

    switch (memType) {
      case kMemNcclMemAlloc:
      case kCuMemAllocDisjoint:
        // Expect usage from NVL backend unless excluded by particular
        // collective
        if (nLocalRanks > 1 &&
            isBackendValid(excludedBackends, CtranMapperBackend::NVL)) {
          if (NCCL_CTRAN_NVL_SENDRECV_COPY_ENGINE_ENABLE) {
            ASSERT_GT(ctran->mapper->iCopyCount, 0);
          } else {
            ASSERT_GT(ctran->mapper->iPutCount[CtranMapperBackend::NVL], 0);
          }
        } else {
          ASSERT_EQ(ctran->mapper->iPutCount[CtranMapperBackend::NVL], 0);
        }

        // Expect usage from IB backend unless excluded by particular collective
        if (nRanks > nLocalRanks &&
            isBackendValid(excludedBackends, CtranMapperBackend::IB)) {
          ASSERT_GT(ctran->mapper->iPutCount[CtranMapperBackend::IB], 0);
        }
        // Do not assume no IB usage, because IB backend may be used also for
        // local ranks if NVL backend is not available
        break;

      case kMemCudaMalloc:
        // memType is kMemCudaMalloc
        // Expect usage from IB backend as long as nRanks > 1, unless excluded
        // by particular collective
        if (nRanks > 1 &&
            isBackendValid(excludedBackends, CtranMapperBackend::IB)) {
          ASSERT_GT(ctran->mapper->iPutCount[CtranMapperBackend::IB], 0);
        }
        // Do not assume no IB usage, because IB backend may be used also for
        // local ranks if NVL backend is not available
        break;

      default:
        ASSERT_TRUE(false) << "Unsupported memType " << memType;
    }
  }

  void* prepareBuf(
      size_t bufSize,
      MemAllocType memType,
      std::vector<TestMemSegment>& segments);

  void releaseBuf(void* buf, size_t bufSize, MemAllocType memType);

  inline size_t pageAligned(size_t nBytes) {
    return ((nBytes + pageSize_ - 1) / pageSize_) * pageSize_;
  }
};

class CtranDistBaseTest : public NcclxBaseTest, public CtranBaseTest {
 public:
  CtranDistBaseTest() : NcclxBaseTest(true){};

  // Global commWorld shared by all tests running by the process.
  // Destorying in TearDownTestSuite() to ensure release commWorld only after
  // all tests.
  static ncclComm_t commWorld;
  static std::unique_ptr<c10d::TCPStore> tcpStoreServer;
  static void TearDownTestSuite();

 protected:
  cudaStream_t stream = 0;

  void SetUp() override;
  void TearDown() override;
};
