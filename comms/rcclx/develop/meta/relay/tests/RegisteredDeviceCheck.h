// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#pragma once

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

namespace rcclx::relay::test {

struct DeviceCheckStatus {
  unsigned long long count;
  unsigned long long inverseCount;
  unsigned long long first;
  unsigned long long inverseFirst;
  unsigned long long sequence;
  unsigned long long inverseSequence;
};

static __global__ void initDeviceCheck(
    DeviceCheckStatus* status,
    size_t count) {
  status->count = 0;
  status->first = count;
}

template <typename Word, bool Reference>
static __global__ void compareDeviceWords(
    const Word* actual,
    const Word* expected,
    Word constant,
    size_t count,
    DeviceCheckStatus* status) {
  unsigned long long different = 0;
  unsigned long long first = count;
  for (size_t i = size_t{blockIdx.x} * blockDim.x + threadIdx.x; i < count;
       i += size_t{gridDim.x} * blockDim.x) {
    if (actual[i] != (Reference ? expected[i] : constant)) {
      ++different;
      first = first < i ? first : i;
    }
  }
  if (different != 0) {
    atomicAdd(&status->count, different);
    atomicMin(&status->first, first);
  }
}

static __global__ void finishDeviceCheck(
    DeviceCheckStatus* status,
    unsigned long long sequence) {
  status->inverseCount = ~status->count;
  status->inverseFirst = ~status->first;
  status->sequence = sequence;
  status->inverseSequence = ~sequence;
}

// Only the encoded status crosses D2H. Actual/reference words stay on device.
class RegisteredDeviceCheck {
 public:
  struct Result {
    size_t count;
    size_t first;
  };

  explicit RegisteredDeviceCheck(size_t capacity) : capacity_(capacity) {}
  RegisteredDeviceCheck(const RegisteredDeviceCheck&) = delete;
  RegisteredDeviceCheck& operator=(const RegisteredDeviceCheck&) = delete;
  ~RegisteredDeviceCheck() {
    Shutdown();
  }

  static bool Valid(
      const DeviceCheckStatus& status,
      size_t count,
      unsigned long long sequence) {
    return status.count <= count && status.first <= count &&
        (status.count == 0 ? status.first == count : status.first < count) &&
        status.inverseCount == ~status.count &&
        status.inverseFirst == ~status.first && status.sequence == sequence &&
        status.inverseSequence == ~status.sequence;
  }

  Result CompareHost(
      const void* actual,
      const std::vector<uint16_t>& expected,
      hipStream_t stream) {
    Range<uint16_t>(actual, expected.size());
    if (reference_ == nullptr) {
      Check(hipMalloc(&reference_, capacity_), "hipMalloc reference");
    }
    Check(
        hipMemcpyAsync(
            reference_,
            expected.data(),
            expected.size() * sizeof(uint16_t),
            hipMemcpyHostToDevice,
            stream),
        "hipMemcpyAsync reference");
    return CompareDevice(actual, reference_, expected.size(), stream);
  }

  Result CompareDevice(
      const void* actual,
      const void* expected,
      size_t count,
      hipStream_t stream) {
    if (expected == nullptr) {
      Fail("null reference", hipErrorInvalidValue);
    }
    return Compare<uint16_t, true>(actual, expected, 0, count, stream);
  }

  template <typename Word>
  Result CompareConstant(
      const void* actual,
      Word expected,
      size_t count,
      hipStream_t stream) {
    return Compare<Word, false>(actual, nullptr, expected, count, stream);
  }

  const void* data() const {
    return storage_;
  }
  static constexpr size_t capacityBytes() {
    return 4096;
  }

  void Shutdown() {
    if (storage_ == nullptr) {
      return;
    }
    CheckDevice();
    Check(hipHostUnregister(storage_), "hipHostUnregister");
    Check(hipFree(status_), "hipFree status");
    if (reference_ != nullptr) {
      Check(hipFree(reference_), "hipFree reference");
      reference_ = nullptr;
    }
    std::free(storage_);
    storage_ = nullptr;
    status_ = nullptr;
  }

 private:
  template <typename Word>
  void Range(const void* actual, size_t count) {
    static_assert(
        std::is_same_v<Word, uint16_t> || std::is_same_v<Word, uint32_t>);
    if (actual == nullptr || count == 0 || count > capacity_ / sizeof(Word)) {
      Fail("invalid comparison range", hipErrorInvalidValue);
    }
    if (storage_ == nullptr) {
      Check(hipGetDevice(&device_), "hipGetDevice");
      if (posix_memalign(&storage_, 4096, 4096) != 0) {
        Fail("posix_memalign", hipErrorOutOfMemory);
      }
      Check(hipHostRegister(storage_, 4096, 0), "hipHostRegister");
      Check(
          hipMalloc(reinterpret_cast<void**>(&status_), sizeof(*status_)),
          "hipMalloc status");
    }
    CheckDevice();
  }

  template <typename Word, bool Reference>
  Result Compare(
      const void* actual,
      const void* expected,
      Word constant,
      size_t count,
      hipStream_t stream) {
    Range<Word>(actual, count);
    if (++sequence_ == 0) {
      Fail("status sequence exhausted", hipErrorInvalidValue);
    }
    initDeviceCheck<<<1, 1, 0, stream>>>(status_, count);
    Check(hipGetLastError(), "GPU status initialization");
    const auto blocks =
        static_cast<unsigned int>(std::min(size_t{1024}, (count + 255) / 256));
    compareDeviceWords<Word, Reference><<<blocks, 256, 0, stream>>>(
        static_cast<const Word*>(actual),
        static_cast<const Word*>(expected),
        constant,
        count,
        status_);
    Check(hipGetLastError(), "GPU bitwise comparison");
    finishDeviceCheck<<<1, 1, 0, stream>>>(status_, sequence_);
    Check(hipGetLastError(), "GPU comparison launch");
    std::memset(storage_, 0xCD, sizeof(DeviceCheckStatus));
    Check(hipStreamSynchronize(stream), "GPU comparison completion");
    Check(
        hipMemcpy(storage_, status_, sizeof(*status_), hipMemcpyDeviceToHost),
        "hipMemcpy status");
    DeviceCheckStatus snapshot;
    std::memcpy(&snapshot, storage_, sizeof(snapshot));
    if (!Valid(snapshot, count, sequence_)) {
      Fail("invalid, incomplete or stale GPU status", hipErrorInvalidValue);
    }
    return {
        static_cast<size_t>(snapshot.count),
        static_cast<size_t>(snapshot.first)};
  }

  void CheckDevice() const {
    int device = -1;
    Check(hipGetDevice(&device), "hipGetDevice");
    if (device != device_) {
      Fail("GPU verifier device changed", hipErrorInvalidDevice);
    }
  }
  [[noreturn]] static void Fail(const char* operation, hipError_t status) {
    std::fprintf(
        stderr,
        "GPU verifier transport: %s: %s\n",
        operation,
        hipGetErrorString(status));
    std::fflush(stderr);
    std::_Exit(EXIT_FAILURE);
  }
  static void Check(hipError_t status, const char* operation) {
    if (status != hipSuccess) {
      Fail(operation, status);
    }
  }

  size_t capacity_;
  void* storage_{nullptr};
  void* reference_{nullptr};
  DeviceCheckStatus* status_{nullptr};
  int device_{-1};
  unsigned long long sequence_{0};
};

} // namespace rcclx::relay::test
