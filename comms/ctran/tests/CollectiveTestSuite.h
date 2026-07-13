// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <folly/init/Init.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/tests/CollectiveTestSuiteKernels.cuh"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/ctran/tests/CtranUtUtils.h"
#include "comms/ctran/utils/Alloc.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace ctran::testsuite {

// Device-buffer RAII shared by the templated harness (user buffers, via
// makeDeviceBuffer) and the compare/checksum helpers below (aligned scratch).
// Free functions/type so both a class member and these non-members use one
// buffer type. `commCuMemAlloc` backing (copy path); no nccl APIs.
inline void* allocateDeviceBuffer(size_t bytes) {
  void* ptr = nullptr;
  COMMCHECK_TEST(
      ctran::utils::commCuMemAlloc(
          &ptr,
          nullptr,
          ctran::utils::getCuMemAllocHandleType(),
          bytes,
          nullptr,
          "CollectiveTestSuite"));
  return ptr;
}

inline void releaseDeviceBuffer(void* ptr) {
  COMMCHECK_TEST(ctran::utils::commCuMemFree(ptr));
}

struct DeviceBufferDeleter {
  void operator()(void* ptr) const {
    if (ptr != nullptr) {
      releaseDeviceBuffer(ptr);
    }
  }
};
using DeviceBuffer = std::unique_ptr<void, DeviceBufferDeleter>;

// Host-side `commDataType_t` dispatch helpers shared by more than one Traits
// struct (and, for the checksum, by the harness determinism check). They wrap
// the algo-agnostic kernels in CollectiveTestSuiteKernels.cuh; they use gtest,
// so they are defined here in the host-only header, not the nvcc-compiled
// `.cuh`. Per-collective expected builders are folded into each Traits'
// buildExpected().

// Return whether `ptr` can be safely accessed as `datatype` elements.
inline bool isAlignedForDatatype(const void* ptr, commDataType_t datatype) {
  const auto alignment = commTypeSize(datatype);
  return alignment != 0 && reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

// max |actual - expected| over `count` elements, for floating-point dtypes.
inline double computeMaxDeltaForType(
    const void* actual,
    const void* expected,
    commDataType_t datatype,
    size_t count,
    cudaStream_t stream) {
  if (!isAlignedForDatatype(actual, datatype)) {
    // Unaligned inputs cannot use typed CUDA loads. Stage through an aligned
    // scratch buffer, then compare. The recursive call syncs the stream before
    // returning, so the RAII free on scope exit is ordered after the read.
    const size_t bytes = count * commTypeSize(datatype);
    DeviceBuffer alignedActual(allocateDeviceBuffer(bytes));
    CUDACHECK_TEST(cudaMemcpyAsync(
        alignedActual.get(), actual, bytes, cudaMemcpyDeviceToDevice, stream));
    return computeMaxDeltaForType(
        alignedActual.get(), expected, datatype, count, stream);
  }
  switch (datatype) {
    case commFloat32:
      return computeMaxDelta(
          static_cast<const float*>(actual),
          static_cast<const float*>(expected),
          count,
          stream);
    case commFloat16:
      return computeMaxDelta(
          static_cast<const __half*>(actual),
          static_cast<const __half*>(expected),
          count,
          stream);
    default:
      ADD_FAILURE() << "computeMaxDeltaForType: unsupported datatype "
                    << datatype;
      return DBL_MAX;
  }
}

// deterministic raw-bits digest over `count` elements.
inline uint64_t computeRawBitsChecksumForType(
    const void* data,
    commDataType_t datatype,
    size_t count,
    cudaStream_t stream) {
  if (!isAlignedForDatatype(data, datatype)) {
    // Unaligned inputs cannot use typed CUDA loads. Stage through an aligned
    // scratch buffer, then digest. The recursive call syncs the stream before
    // returning, so the RAII free on scope exit is ordered after the read.
    const size_t bytes = count * commTypeSize(datatype);
    DeviceBuffer alignedData(allocateDeviceBuffer(bytes));
    CUDACHECK_TEST(cudaMemcpyAsync(
        alignedData.get(), data, bytes, cudaMemcpyDeviceToDevice, stream));
    return computeRawBitsChecksumForType(
        alignedData.get(), datatype, count, stream);
  }
  switch (datatype) {
    case commFloat32:
      return computeRawBitsChecksum(
          static_cast<const float*>(data), count, stream);
    case commFloat16:
      return computeRawBitsChecksum(
          static_cast<const __half*>(data), count, stream);
    default:
      ADD_FAILURE() << "unsupported datatype " << datatype;
      return 0;
  }
}

// per-dtype absolute-error tolerance, scaled by nranks (floating-point only).
inline double thresholdForDatatype(commDataType_t datatype, int nranks) {
  const int rankScale = std::max(1, nranks - 1);
  switch (datatype) {
    case commFloat32:
      return 1e-5 * rankScale;
    case commFloat16:
      // FP16 accumulation error grows with rank count, but keep the bound
      // tight enough to catch small bit-corruption regressions.
      return 1e-3 * rankScale;
    default:
      return 0.0;
  }
}

} // namespace ctran::testsuite

// Per-collective buffer layout for one correctness iteration. `plan()` on each
// Traits struct derives every field from (count, nranks, rank, inPlace,
// offsetBytes, typeBytes); the shared harness below consumes it verbatim.
struct BufferPlan {
  size_t sendAllocBytes{0};
  size_t recvAllocBytes{0}; // 0 when in-place (recv aliases send).
  size_t sendUserOffsetBytes{0};
  size_t recvUserOffsetBytes{0};
  bool recvAliasesSend{false};
  bool sendAliasesRecv{false};
  size_t sendInitElems{0};
  size_t resultElems{0};
};

/**
 * Traits-templated distributed CTRAN collective correctness fixture. One shared
 * harness (buffer alloc, guard fill, determinism loop, guard-byte verify,
 * GPE-leak verify) parameterized over a collective's layout + verification
 * semantics via `Traits`. Leaf fixtures override the algo dispatch hooks and
 * invoke REGISTER_COLLECTIVE_TESTS to emit their gtest suites.
 */
template <class Traits>
class CollectiveTestSuite : public ctran::CtranDistTestFixture,
                            public CtranBaseTest {
 public:
  using Algo = typename Traits::Algo;
  using DeviceBuffer = ctran::testsuite::DeviceBuffer;

  CollectiveTestSuite() = default;

  /** Identical-input repeats used to check bitwise deterministic output. */
  static constexpr int kDeterminismRepeats = 4;

  void SetUp() override {
    ctran::CtranDistTestFixture::SetUp(envOverrides());
    ctranComm = makeCtranComm();
  }

  void TearDown() override {
    ctran::CtranDistTestFixture::TearDown();
    ncclCvarInit();
  }

  void runCorrectnessTest(
      Algo algo,
      commDataType_t datatype,
      size_t count,
      bool inPlace,
      size_t bufferOffsetBytes = 0) {
    if (!algo_supports(datatype, commSum)) {
      GTEST_SKIP() << "algo " << Traits::algoName(algo)
                   << " does not support datatype "
                   << commDataTypeToString(datatype);
    }
    if (!Traits::support(ctranComm.get(), algo)) {
      GTEST_SKIP() << "support returns false for algo "
                   << Traits::algoName(algo);
    }
    if (count == 0) {
      auto res =
          run(algo,
              nullptr,
              nullptr,
              0,
              datatype,
              commSum,
              ctranComm.get(),
              testStream,
              std::nullopt);
      EXPECT_EQ(res, commSuccess);
      return;
    }

    constexpr uint8_t kGuardPattern = 0xA5;
    const size_t typeBytes = commTypeSize(datatype);
    const BufferPlan p = Traits::plan(
        count, numRanks, globalRank, inPlace, bufferOffsetBytes, typeBytes);

    // Send-alloc buffer. When the recv buffer is the primary output (AG), the
    // send buffer may instead alias into recv (in-place) - handled below.
    DeviceBuffer sendAlloc = nullptr;
    if (p.sendAllocBytes > 0) {
      sendAlloc = makeDeviceBuffer(p.sendAllocBytes);
      CUDACHECK_TEST(
          cudaMemset(sendAlloc.get(), kGuardPattern, p.sendAllocBytes));
    }
    DeviceBuffer recvAlloc = nullptr;
    if (p.recvAllocBytes > 0) {
      recvAlloc = makeDeviceBuffer(p.recvAllocBytes);
      CUDACHECK_TEST(
          cudaMemset(recvAlloc.get(), kGuardPattern, p.recvAllocBytes));
    }

    // Resolve the primary (non-aliasing) allocation for each of send/recv, then
    // derive user pointers. recvAliasesSend (RS/AR in-place) and
    // sendAliasesRecv (AG in-place) fold one buffer onto the other. Offset only
    // a non-null base: a zero-byte allocation returns nullptr, and
    // `nullptr + offset` would be UB for a nonzero offset (a future traits with
    // sendAllocBytes/recvAllocBytes == 0), so keep the null through.
    auto userPtr = [](void* base, std::size_t offset) -> void* {
      return base == nullptr ? nullptr : static_cast<char*>(base) + offset;
    };
    void* sendPtr = nullptr;
    void* recvPtr = nullptr;
    if (p.sendAliasesRecv) {
      // AG in-place: recv is primary; send aliases into recv.
      recvPtr = userPtr(recvAlloc.get(), p.recvUserOffsetBytes);
      sendPtr = userPtr(recvAlloc.get(), p.sendUserOffsetBytes);
    } else if (p.recvAliasesSend) {
      // RS/AR in-place: send is primary; recv aliases into send.
      sendPtr = userPtr(sendAlloc.get(), p.sendUserOffsetBytes);
      recvPtr = userPtr(sendAlloc.get(), p.recvUserOffsetBytes);
    } else {
      // Out-of-place: independent send + recv allocations.
      sendPtr = userPtr(sendAlloc.get(), p.sendUserOffsetBytes);
      recvPtr = userPtr(recvAlloc.get(), p.recvUserOffsetBytes);
    }

    DeviceBuffer expectedBuf = makeDeviceBuffer(p.resultElems * typeBytes);
    Traits::buildExpected(
        expectedBuf.get(), datatype, count, numRanks, globalRank, testStream);

    // Unaligned send pointers cannot be filled with typed CUDA stores; stage
    // the init through an aligned scratch buffer. Aligned pointers (and empty
    // send payloads, which init as a no-op) skip this.
    DeviceBuffer sendStaging =
        (p.sendInitElems == 0 ||
         ctran::testsuite::isAlignedForDatatype(sendPtr, datatype))
        ? nullptr
        : makeDeviceBuffer(p.sendInitElems * typeBytes);

    std::array<uint64_t, kDeterminismRepeats> checksums{};
    for (int repeat = 0; repeat < kDeterminismRepeats; ++repeat) {
      launchInitData(
          sendPtr,
          datatype,
          p.sendInitElems,
          globalRank,
          /*rep=*/0,
          testStream,
          sendStaging.get());
      CUDACHECK_TEST(cudaStreamSynchronize(testStream));

      commResult_t res =
          run(algo,
              sendPtr,
              recvPtr,
              count,
              datatype,
              commSum,
              ctranComm.get(),
              testStream,
              std::nullopt);
      EXPECT_EQ(res, commSuccess);
      CUDACHECK_TEST(cudaStreamSynchronize(testStream));

      Traits::checkResult(
          recvPtr,
          expectedBuf.get(),
          datatype,
          p.resultElems,
          numRanks,
          testStream,
          count,
          globalRank,
          inPlace,
          repeat);

      checksums[repeat] = ctran::testsuite::computeRawBitsChecksumForType(
          recvPtr, datatype, p.resultElems, testStream);
    }

    CUDACHECK_TEST(cudaStreamSynchronize(testStream));
    // Verify the guard region before AND after the user payload on each real
    // allocation. The leading region is [0, userOffset); the trailing region is
    // [userOffset + payloadBytes, allocBytes) - non-empty only if a plan()
    // over-allocates a tail past the payload (none do today, but this keeps
    // coverage complete if one ever does).
    if (sendAlloc != nullptr) {
      expectDeviceBytePattern(
          sendAlloc.get(), p.sendUserOffsetBytes, kGuardPattern);
      const size_t sendUsed =
          p.sendUserOffsetBytes + p.sendInitElems * typeBytes;
      if (p.sendAllocBytes > sendUsed) {
        expectDeviceBytePattern(
            static_cast<char*>(sendAlloc.get()) + sendUsed,
            p.sendAllocBytes - sendUsed,
            kGuardPattern);
      }
    }
    if (recvAlloc != nullptr) {
      expectDeviceBytePattern(
          recvAlloc.get(), p.recvUserOffsetBytes, kGuardPattern);
      const size_t recvUsed = p.recvUserOffsetBytes + p.resultElems * typeBytes;
      if (p.recvAllocBytes > recvUsed) {
        expectDeviceBytePattern(
            static_cast<char*>(recvAlloc.get()) + recvUsed,
            p.recvAllocBytes - recvUsed,
            kGuardPattern);
      }
    }
    for (int repeat = 1; repeat < kDeterminismRepeats; ++repeat) {
      ASSERT_EQ(checksums[0], checksums[repeat])
          << "non-deterministic output for count=" << count
          << " rank=" << globalRank
          << " datatype=" << commDataTypeToString(datatype)
          << " inPlace=" << inPlace << " repeat=" << repeat;
    }
    verifyGpeLeak(ctranComm->ctran_.get());
  }

  void runCorrectnessSweep(
      Algo algo,
      commDataType_t datatype,
      const std::vector<size_t>& counts,
      bool inPlace,
      size_t bufferOffsetBytes = 0) {
    for (size_t count : counts) {
      runCorrectnessTest(algo, datatype, count, inPlace, bufferOffsetBytes);
    }
  }

 protected:
  /** Dispatch one collective call for the algorithm under test. AG leaves
   * ignore the op arg. */
  virtual commResult_t run(
      Algo algo,
      const void* send,
      void* recv,
      size_t count,
      commDataType_t datatype,
      commRedOp_t op,
      CtranComm* comm,
      cudaStream_t stream,
      std::optional<std::chrono::milliseconds> timeout) = 0;

  virtual bool algo_supports(commDataType_t /*dt*/, commRedOp_t /*op*/) const {
    return true;
  }

  virtual ctran::CtranEnvs envOverrides() const {
    ctran::CtranEnvs envs{
        {"NCCL_CTRAN_USE_PIPES", "1"},
        // Keep setup cost bounded across the parameterized sweep.
        {"NCCL_CTRAN_MAX_NBLOCKS", "8"},
    };
#ifdef CTRAN_TEST_SOCKET_ONLY_BACKEND
    envs.emplace_back("NCCL_CTRAN_BACKENDS", "socket, nvl");
#endif
    return envs;
  }

  // User buffer, `commCuMemAlloc`-backed (copy path). Wraps the free
  // ctran::testsuite::allocateDeviceBuffer in the shared DeviceBuffer RAII.
  static DeviceBuffer makeDeviceBuffer(size_t bytes) {
    return DeviceBuffer(ctran::testsuite::allocateDeviceBuffer(bytes));
  }

  // Initialize an aligned `datatype` buffer directly through typed CUDA stores.
  static void launchInitDataAligned(
      void* buf,
      commDataType_t datatype,
      size_t count,
      int rank,
      int rep,
      cudaStream_t stream) {
    ASSERT_TRUE(ctran::testsuite::isAlignedForDatatype(buf, datatype));
    switch (datatype) {
      case commFloat32:
        launchInitDataKernel(
            static_cast<float*>(buf), count, rank, rep, stream);
        return;
      case commFloat16:
        launchInitDataKernel(
            static_cast<__half*>(buf), count, rank, rep, stream);
        return;
      default:
        ADD_FAILURE() << "unsupported datatype " << datatype;
        return;
    }
  }

  // Initialize a possibly unaligned buffer with deterministic test data.
  //
  // Unaligned offset tests cannot use typed CUDA stores for setup. Stage
  // through an aligned scratch buffer so only the collective call receives the
  // unaligned payload pointer.
  static void launchInitData(
      void* buf,
      commDataType_t datatype,
      size_t count,
      int rank,
      int rep,
      cudaStream_t stream,
      void* alignedScratch = nullptr) {
    if (count == 0) {
      return;
    }
    if (ctran::testsuite::isAlignedForDatatype(buf, datatype)) {
      launchInitDataAligned(buf, datatype, count, rank, rep, stream);
      return;
    }
    ASSERT_NE(alignedScratch, nullptr);
    const size_t bytes = count * commTypeSize(datatype);
    launchInitDataAligned(alignedScratch, datatype, count, rank, rep, stream);
    CUDACHECK_TEST(cudaMemcpyAsync(
        buf, alignedScratch, bytes, cudaMemcpyDeviceToDevice, stream));
  }

  // ASSERT that `bytes` bytes at `ptr` all equal `expected` (guard-byte check).
  static void
  expectDeviceBytePattern(const void* ptr, size_t bytes, uint8_t expected) {
    if (bytes == 0) {
      return;
    }
    std::vector<uint8_t> host(bytes);
    CUDACHECK_TEST(cudaMemcpy(host.data(), ptr, bytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < bytes; ++i) {
      ASSERT_EQ(host[i], expected) << "guard byte mismatch at offset " << i;
    }
  }

  cudaStream_t testStream{0};
  std::unique_ptr<CtranComm> ctranComm{nullptr};
};

// -----------------------------------------------------------------------------
// Shared sweep builders + helpers used by REGISTER_COLLECTIVE_TESTS. These are
// generic across every collective; a leaf only supplies its Traits, algo value,
// and dtype list. The consolidated size sweep folds in the former mixed-order
// sequence, the full correctness sweep, the 100B offset payload, and (at
// offset 0) a 512 MiB large payload.
// -----------------------------------------------------------------------------
inline void collectiveAppendUniqueCount(
    std::vector<size_t>& counts,
    size_t count) {
  if (std::find(counts.begin(), counts.end(), count) == counts.end()) {
    counts.push_back(count);
  }
}

inline std::vector<size_t> makeCollectiveElementCounts(
    commDataType_t datatype,
    bool includeZeroCount,
    bool includeLargePayload) {
  std::vector<size_t> counts;
  const size_t elemBytes = commTypeSize(datatype);
  for (size_t count : {
           // Preserve the previous mixed-order sequence first.
           size_t{1},
           size_t{1048576 + 7},
           size_t{7},
           size_t{65536},
           size_t{16},
           size_t{1048576 - 7},
           // Complete the previous correctness sweep.
           size_t{0},
           size_t{64},
           size_t{256},
           size_t{1024},
           size_t{4096},
           size_t{16384},
           size_t{262144},
           size_t{1048576},
           size_t{4194304},
           size_t{4194304 - 13},
           size_t{4194304 + 13},
       }) {
    if (count == 0 && !includeZeroCount) {
      continue;
    }
    collectiveAppendUniqueCount(counts, count);
  }
  // Preserve the previous 100-byte offset-test payload.
  collectiveAppendUniqueCount(counts, 100 / elemBytes);
  if (includeLargePayload) {
    collectiveAppendUniqueCount(counts, (512 * 1024 * 1024) / elemBytes);
  }
  return counts;
}

// Byte offsets that exercise unaligned user-buffer starts: 0, one element, and
// 15 bytes when that is a whole number of elements for this dtype.
inline std::vector<size_t> collectiveBufferOffsetBytes(
    commDataType_t datatype) {
  constexpr size_t kByteOffsetCoverageBytes = 15;
  const size_t elemBytes = commTypeSize(datatype);
  std::vector<size_t> offsets;
  auto appendUnique = [&](size_t offset) {
    if (std::find(offsets.begin(), offsets.end(), offset) == offsets.end()) {
      offsets.push_back(offset);
    }
  };
  appendUnique(0);
  appendUnique(elemBytes);
  if (kByteOffsetCoverageBytes % elemBytes == 0) {
    appendUnique(kByteOffsetCoverageBytes);
  }
  return offsets;
}

// NCCL_CTRAN_MAX_NBLOCKS caps swept for block-cap correctness coverage. The
// block cap is a shared cnvlmm knob, so every collective gets this coverage.
inline const std::vector<int>& collectiveMaxBlockCaps() {
  static const std::vector<int> caps{1, 2, 3, 4, 5, 6, 7, 8};
  return caps;
}

// Set (or replace) an env override in place.
inline void collectiveSetEnvOverride(
    ctran::CtranEnvs& envs,
    const std::string& key,
    const std::string& value) {
  for (auto& env : envs) {
    if (env.first == key) {
      env.second = value;
      return;
    }
  }
  envs.emplace_back(key, value);
}

inline std::string collectiveDataTypeName(commDataType_t datatype) {
  switch (datatype) {
    case commFloat32:
      return "FP32";
    case commFloat16:
      return "FP16";
    default:
      return "UnsupportedType";
  }
}

// dtype lists reused by leaf fixtures.
// cnvlmmDataTypes() is consumed by the RS/AG/AR cnvlmm leaf fixtures added
// later in this stack (first use in D110902873); defined here so the shared
// list lives with the harness rather than being duplicated per leaf.
inline const std::vector<commDataType_t>& cnvlmmDataTypes() {
  static const std::vector<commDataType_t> datatypes{commFloat32, commFloat16};
  return datatypes;
}

inline const std::vector<commDataType_t>& allReduceTreeDataTypes() {
  static const std::vector<commDataType_t> datatypes{commFloat32, commFloat16};
  return datatypes;
}

// One (algo, datatype, inPlace, block-cap) point for the block-cap sweep.
template <class Algo>
struct CollectiveBlockCapParam {
  Algo algo;
  commDataType_t datatype;
  bool inPlace;
  int maxBlockCap;
};

template <class Algo>
std::vector<CollectiveBlockCapParam<Algo>> makeCollectiveBlockCapParams(
    Algo algo,
    const std::vector<commDataType_t>& datatypes) {
  std::vector<CollectiveBlockCapParam<Algo>> params;
  for (commDataType_t datatype : datatypes) {
    for (int maxBlockCap : collectiveMaxBlockCaps()) {
      params.push_back(
          CollectiveBlockCapParam<Algo>{
              algo, datatype, /*inPlace=*/false, maxBlockCap});
      params.push_back(
          CollectiveBlockCapParam<Algo>{
              algo, datatype, /*inPlace=*/true, maxBlockCap});
    }
  }
  return params;
}

template <class Traits>
std::string collectiveSweepTestName(
    const testing::TestParamInfo<
        std::tuple<typename Traits::Algo, commDataType_t, bool>>& info) {
  auto [algo, datatype, inPlace] = info.param;
  return Traits::algoName(algo) + "_" + collectiveDataTypeName(datatype) +
      "_Sweep_" + (inPlace ? "InPlace" : "OutOfPlace");
}

template <class Traits>
std::string collectiveBlockCapTestName(
    const testing::TestParamInfo<
        CollectiveBlockCapParam<typename Traits::Algo>>& info) {
  const auto& param = info.param;
  return Traits::algoName(param.algo) + "_" +
      collectiveDataTypeName(param.datatype) + "_Sweep_" +
      (param.inPlace ? "InPlace" : "OutOfPlace") + "_MaxBlocks" +
      std::to_string(param.maxBlockCap);
}

// Emit the consolidated correctness sweep + block-cap sweep suites bound to a
// concrete `Fixture` (must derive CollectiveTestSuite<Traits>). `AlgoEnum` is
// the single algo value under test; `DataTypesFn` supplies the dtype list. Each
// correctness case sweeps every buffer offset x element count internally; each
// block-cap case pins NCCL_CTRAN_MAX_NBLOCKS and sweeps element counts.
#define REGISTER_COLLECTIVE_TESTS(Fixture, TraitsType, AlgoEnum, DataTypesFn) \
  using Fixture##SweepParam =                                                 \
      std::tuple<TraitsType::Algo, commDataType_t, bool>;                     \
  using Fixture##BlockCapParam = CollectiveBlockCapParam<TraitsType::Algo>;   \
  class Fixture##CorrectnessTest                                              \
      : public Fixture,                                                       \
        public ::testing::WithParamInterface<Fixture##SweepParam> {};         \
  class Fixture##BlockCapCorrectnessTest                                      \
      : public Fixture,                                                       \
        public ::testing::WithParamInterface<Fixture##BlockCapParam> {        \
   protected:                                                                 \
    ctran::CtranEnvs envOverrides() const override {                          \
      auto envs = Fixture::envOverrides();                                    \
      collectiveSetEnvOverride(                                               \
          envs,                                                               \
          "NCCL_CTRAN_MAX_NBLOCKS",                                           \
          std::to_string(this->GetParam().maxBlockCap));                      \
      return envs;                                                            \
    }                                                                         \
  };                                                                          \
  TEST_P(Fixture##CorrectnessTest, Correctness) {                             \
    auto [algo, datatype, inPlace] = GetParam();                              \
    for (size_t offsetBytes : collectiveBufferOffsetBytes(datatype)) {        \
      runCorrectnessSweep(                                                    \
          algo,                                                               \
          datatype,                                                           \
          makeCollectiveElementCounts(                                        \
              datatype,                                                       \
              /*includeZeroCount=*/offsetBytes == 0,                          \
              /*includeLargePayload=*/offsetBytes == 0),                      \
          inPlace,                                                            \
          offsetBytes);                                                       \
    }                                                                         \
  }                                                                           \
  TEST_P(Fixture##BlockCapCorrectnessTest, Correctness) {                     \
    const auto& param = GetParam();                                           \
    runCorrectnessSweep(                                                      \
        param.algo,                                                           \
        param.datatype,                                                       \
        makeCollectiveElementCounts(                                          \
            param.datatype,                                                   \
            /*includeZeroCount=*/true,                                        \
            /*includeLargePayload=*/true),                                    \
        param.inPlace);                                                       \
  }                                                                           \
  INSTANTIATE_TEST_SUITE_P(                                                   \
      Fixture,                                                                \
      Fixture##CorrectnessTest,                                               \
      ::testing::Combine(                                                     \
          ::testing::Values(AlgoEnum),                                        \
          ::testing::ValuesIn(DataTypesFn()),                                 \
          ::testing::Bool()),                                                 \
      collectiveSweepTestName<TraitsType>);                                   \
  INSTANTIATE_TEST_SUITE_P(                                                   \
      Fixture,                                                                \
      Fixture##BlockCapCorrectnessTest,                                       \
      ::testing::ValuesIn(                                                    \
          makeCollectiveBlockCapParams(AlgoEnum, DataTypesFn())),             \
      collectiveBlockCapTestName<TraitsType>);

// Standard distributed test main() shared by every collective leaf. folly::Init
// is required for now so the cnvlmm/prims singletons initialize; it will be
// dropped when the folly dependency is removed.
#define COLLECTIVE_TEST_MAIN()                                            \
  int main(int argc, char* argv[]) {                                      \
    ::testing::InitGoogleTest(&argc, argv);                               \
    ::testing::AddGlobalTestEnvironment(new ctran::CtranDistEnvironment); \
    folly::Init init(&argc, &argv);                                       \
    return RUN_ALL_TESTS();                                               \
  }
