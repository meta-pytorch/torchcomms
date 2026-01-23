// Copyright (c) Meta Platforms, Inc. and affiliates.

// @lint-ignore-every CLANGTIDY facebook-modularize-issue-check

#pragma once

#include <cassert>
#include <cstdint>

namespace comms::pipes {

// Device-side bounds check helper
#ifdef __CUDA_ARCH__
#define DEVICE_SPAN_CHECK_LT(val, bound)                          \
  do {                                                            \
    if (!((val) < (bound))) {                                     \
      printf(                                                     \
          "DeviceSpan: %u < %u failed at %s:%d block=(%u,%u,%u) " \
          "thread=(%u,%u,%u)\n",                                  \
          (unsigned)(val),                                        \
          (unsigned)(bound),                                      \
          __FILE__,                                               \
          __LINE__,                                               \
          blockIdx.x,                                             \
          blockIdx.y,                                             \
          blockIdx.z,                                             \
          threadIdx.x,                                            \
          threadIdx.y,                                            \
          threadIdx.z);                                           \
      __trap();                                                   \
    }                                                             \
  } while (0)

#define DEVICE_SPAN_CHECK_LE(val, bound)                           \
  do {                                                             \
    if (!((val) <= (bound))) {                                     \
      printf(                                                      \
          "DeviceSpan: %u <= %u failed at %s:%d block=(%u,%u,%u) " \
          "thread=(%u,%u,%u)\n",                                   \
          (unsigned)(val),                                         \
          (unsigned)(bound),                                       \
          __FILE__,                                                \
          __LINE__,                                                \
          blockIdx.x,                                              \
          blockIdx.y,                                              \
          blockIdx.z,                                              \
          threadIdx.x,                                             \
          threadIdx.y,                                             \
          threadIdx.z);                                            \
      __trap();                                                    \
    }                                                              \
  } while (0)

#define DEVICE_SPAN_CHECK_GT(val, bound)                          \
  do {                                                            \
    if (!((val) > (bound))) {                                     \
      printf(                                                     \
          "DeviceSpan: %u > %u failed at %s:%d block=(%u,%u,%u) " \
          "thread=(%u,%u,%u)\n",                                  \
          (unsigned)(val),                                        \
          (unsigned)(bound),                                      \
          __FILE__,                                               \
          __LINE__,                                               \
          blockIdx.x,                                             \
          blockIdx.y,                                             \
          blockIdx.z,                                             \
          threadIdx.x,                                            \
          threadIdx.y,                                            \
          threadIdx.z);                                           \
      __trap();                                                   \
    }                                                             \
  } while (0)
#else
#define DEVICE_SPAN_CHECK_LT(val, bound) ((void)0)
#define DEVICE_SPAN_CHECK_LE(val, bound) ((void)0)
#define DEVICE_SPAN_CHECK_GT(val, bound) ((void)0)
#endif

/**
 * DeviceSpan - A lightweight, non-owning view of contiguous device memory
 *
 * This is a simple span implementation for passing device memory arrays to
 * kernels. Element access is restricted to device code only - the host can
 * only access metadata (data pointer, size, empty).
 *
 * HOST OPERATIONS (construction and metadata only):
 * =================================================
 *
 * 1. From raw device pointer and size:
 *    uint32_t* device_ptr;
 *    cudaMalloc(&device_ptr, 100 * sizeof(uint32_t));
 *    DeviceSpan<uint32_t> span(device_ptr, 100);
 *
 * 2. From DeviceBuffer (common pattern):
 *    DeviceBuffer buffer(numElements * sizeof(uint32_t));
 *    auto span = DeviceSpan<uint32_t>(
 *        static_cast<uint32_t*>(buffer.get()), numElements);
 *
 * 3. Using factory function:
 *    auto span = make_device_span(device_ptr, size);
 *
 * 4. Host can only access:
 *    - span.data()  -> pointer to device memory
 *    - span.size()  -> number of elements
 *    - span.empty() -> whether span is empty
 *
 * DEVICE OPERATIONS (element access):
 * ====================================
 *
 *   __global__ void myKernel(DeviceSpan<const uint32_t> weights) {
 *     // Range-based for loop (device only)
 *     for (uint32_t val : weights) {
 *       // process val
 *     }
 *
 *     // Index-based access (device only)
 *     for (uint32_t i = 0; i < weights.size(); i++) {
 *       uint32_t val = weights[i];
 *     }
 *
 *     // Subspan operations (device only)
 *     auto first_half = weights.first(weights.size() / 2);
 *     auto last_half = weights.last(weights.size() / 2);
 *     auto middle = weights.subspan(10, 20);
 *   }
 *
 * @tparam T The element type (can be const-qualified for read-only spans)
 */
template <typename T>
class DeviceSpan {
 public:
  using element_type = T;
  using value_type = typename std::remove_cv<T>::type;
  using size_type = uint32_t;
  using pointer = T*;
  using const_pointer = T const* const;
  using reference = T&;
  using const_reference = const T&;

  __host__ __device__ constexpr DeviceSpan() noexcept
      : data_(nullptr), size_(0) {}

  __host__ __device__ constexpr DeviceSpan(
      pointer data,
      size_type size) noexcept
      : data_(data), size_(size) {}

  // Allow implicit conversion from non-const to const span
  template <
      typename U,
      typename = typename std::enable_if<
          std::is_same<const U, T>::value && !std::is_same<U, T>::value>::type>
  __host__ __device__ constexpr DeviceSpan(const DeviceSpan<U>& other) noexcept
      : data_(other.data()), size_(other.size()) {}

  // Host and device: metadata access
  __host__ __device__ constexpr pointer data() const noexcept {
    return data_;
  }

  __host__ __device__ constexpr size_type size() const noexcept {
    return size_;
  }

  __host__ __device__ constexpr bool empty() const noexcept {
    return size_ == 0;
  }

  // Device only: element access (with bounds checking)
  __device__ __forceinline__ constexpr reference operator[](
      size_type idx) const {
    DEVICE_SPAN_CHECK_LT(idx, size_);
    return data_[idx];
  }

  __device__ __forceinline__ constexpr reference front() const {
    DEVICE_SPAN_CHECK_GT(size_, 0);
    return data_[0];
  }

  __device__ __forceinline__ constexpr reference back() const {
    DEVICE_SPAN_CHECK_GT(size_, 0);
    return data_[size_ - 1];
  }

  // Device only: iterator support for range-based for loops
  __device__ __forceinline__ constexpr pointer begin() const noexcept {
    return data_;
  }

  __device__ __forceinline__ constexpr pointer end() const noexcept {
    return data_ + size_;
  }

  // Device only: subspan operations (with bounds checking)
  __device__ __forceinline__ constexpr DeviceSpan<T> subspan(
      size_type offset,
      size_type count) const {
    DEVICE_SPAN_CHECK_LE(offset, size_);
    DEVICE_SPAN_CHECK_LE(count, size_ - offset);
    return DeviceSpan<T>(data_ + offset, count);
  }

  __device__ __forceinline__ constexpr DeviceSpan<T> subspan(
      size_type offset) const {
    DEVICE_SPAN_CHECK_LE(offset, size_);
    return DeviceSpan<T>(data_ + offset, size_ - offset);
  }

  __device__ __forceinline__ constexpr DeviceSpan<T> first(
      size_type count) const {
    DEVICE_SPAN_CHECK_LE(count, size_);
    return DeviceSpan<T>(data_, count);
  }

  __device__ __forceinline__ constexpr DeviceSpan<T> last(
      size_type count) const {
    DEVICE_SPAN_CHECK_LE(count, size_);
    return DeviceSpan<T>(data_ + size_ - count, count);
  }

 private:
  pointer data_;
  size_type size_;
};

/**
 * PERFORMANCE NOTE: Lambda Capture and Aliasing
 * =============================================
 *
 * When using DeviceSpan inside a lambda that writes to its elements, extract
 * the raw pointer BEFORE the lambda to avoid aliasing issues:
 *
 *   // SLOW - compiler reloads data_ on each access:
 *   DeviceSpan<T> span = ...;
 *   lambda([&] { span[i].modify(); });
 *
 *   // FAST - pointer is a local variable, no aliasing concerns:
 *   T* ptr = span.data();
 *   lambda([&] { ptr[i].modify(); });
 *
 * The compiler cannot prove that writes to elements don't affect the span's
 * data_ member, causing repeated loads. Extracting to a local pointer variable
 * solves this because local variables are provably not aliased.
 */

// Convenience factory function
template <typename T>
__host__ __device__ constexpr DeviceSpan<T> make_device_span(
    T* data,
    uint32_t size) noexcept {
  return DeviceSpan<T>(data, size);
}

} // namespace comms::pipes
