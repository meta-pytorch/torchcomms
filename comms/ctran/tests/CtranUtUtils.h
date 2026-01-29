// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/Ctran.h"
#include "comms/ctran/commstate/CommStateX.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsCuUtils.h"
#include "comms/testinfra/TestsDistUtils.h"

using ncclx::CommStateX;

// TODO: migrate to the test fixture in CtranXPlatUtUtils.h and deprecate this
// file

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
  std::unordered_set<void*> devArgs_;

 protected:
  // allocate an argument from device memory without assigning any value
  void allocDevArg(const size_t nbytes, void*& ptr);

  // allocate an argument from device memory and assign the value specified by
  // argVec
  template <typename T>
  void allocDevArg(const std::vector<T>& argVec, T*& ptr) {
    void* ptr_ = nullptr;
    size_t nbytes = sizeof(T) * argVec.size();
    allocDevArg(nbytes, ptr_);
    CUDACHECK_ASSERT(cudaMemcpy(
        ptr_, argVec.data(), argVec.size() * sizeof(T), cudaMemcpyDefault));
    ptr = reinterpret_cast<T*>(ptr_);
  }

  // release all device arguments allocated by allocDevArg
  void releaseDevArgs();
  // release a single device argument allocated by allocDevArg with the pointer
  void releaseDevArg(void* ptr);

  // asynchrounously assign value to a device argument specified by argVec
  template <typename T>
  void assignDevArg(T* buf, const std::vector<T>& vals, cudaStream_t stream) {
    CUDACHECK_ASSERT(cudaMemcpyAsync(
        buf, vals.data(), vals.size() * sizeof(T), cudaMemcpyDefault, stream));
  }

  // asynchronously assign value to a device argument specified by single value
  template <typename T>
  void assignDevArg(T* buf, const int count, const T val, cudaStream_t stream) {
    std::vector<T> expVals(count, val);
    return assignDevArg(buf, expVals, stream);
  }

  // check the value of a device argument specified by argVec; if any error is
  // detected, returns maxNumErrs errors at most.
  template <typename T>
  void checkDevArg(
      const T* buf,
      const std::vector<T> expVals,
      std::vector<std::string>& errs,
      const int maxNumErrs = 10) {
    const auto count = expVals.size();
    std::vector<T> obsVals(count, -1);
    CUDACHECK_ASSERT(
        cudaMemcpy(obsVals.data(), buf, count * sizeof(T), cudaMemcpyDefault));
    errs.reserve(maxNumErrs);
    for (auto i = 0; i < count; ++i) {
      if (obsVals[i] != expVals[i] && errs.size() < maxNumErrs) {
        errs.push_back(
            fmt::format(
                "observed[{}] = {}, expected = {}", i, obsVals[i], expVals[i]));
      }
    }
  }

  // wrapper of checkDevArg with single value
  template <typename T>
  void checkDevArg(
      const T* buf,
      const int count,
      const T expVal,
      std::vector<std::string>& errs,
      const int maxNumErrs = 10) {
    std::vector<T> obsVals(count, expVal);
    return checkDevArg(buf, obsVals, errs, maxNumErrs);
  }

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
      std::vector<TestMemSegment>& segments,
      size_t numSegments = 2);

  void releaseBuf(
      void* buf,
      size_t bufSize,
      MemAllocType memType,
      size_t numSegments = 2);

  inline size_t pageAligned(size_t nBytes) {
    return ((nBytes + pageSize_ - 1) / pageSize_) * pageSize_;
  }
};

class CtranDistBaseTest : public NcclxBaseTest, public CtranBaseTest {
 public:
  CtranDistBaseTest() : NcclxBaseTest(true) {};

  // Global commWorld shared by all tests running by the process.
  // Destorying in TearDownTestSuite() to ensure release commWorld only after
  // all tests.
  static ncclComm_t commWorld;
  static std::unique_ptr<c10d::TCPStore> tcpStoreServer;
  static void TearDownTestSuite();

  // Below provide convenient functions to communicate among testing ranks; use
  // bootstrap to avoid interference with GPU communication. Not for
  // communication with performance.

  // - AllGather data from all ranks. buf is a pointer to continuous memory with
  // nRanks * len bytes, and each rank sets its own data in the buffer at
  // postion len * rank.
  inline void allGather(void* buf, const size_t len) {
    auto resFuture = ctranComm_->bootstrap_->allGather(
        buf, len, ctranComm_->statex_->rank(), ctranComm_->statex_->nRanks());
    ASSERT_EQ(
        static_cast<commResult_t>(std::move(resFuture).get()), commSuccess);
  }

  // - Barrier to ensure all ranks have arrived
  inline void barrier() {
    auto resFuture = ctranComm_->bootstrap_->barrier(
        ctranComm_->statex_->rank(), ctranComm_->statex_->nRanks());
    ASSERT_EQ(
        static_cast<commResult_t>(std::move(resFuture).get()), commSuccess);
  }

 protected:
  cudaStream_t stream = 0;
  CtranComm* ctranComm_{nullptr};
  void SetUp() override;
  void TearDown() override;
};
