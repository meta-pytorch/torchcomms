// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

//
// Provides the `EPException` host-side exception type plus three macros used
// across the dispatch / combine kernels:
//   - `EP_STATIC_ASSERT(cond, reason)` — compile-time assert (alias of
//     `static_assert`)
//   - `EP_HOST_ASSERT(cond)` — runtime check on host code; throws EPException
//   - `EP_DEVICE_ASSERT(cond)` — device-side check; printf + trap on failure
//   - `CUDA_CHECK(cmd)` — wrap a CUDA call and throw on non-success

namespace comms::prims::moe_ep::kernels {

class EPException : public std::exception {
 public:
  EPException(
      const char* name,
      const char* file,
      int line,
      const std::string& error) {
    message_ = std::string("Failed: ") + name + " error " + file + ":" +
        std::to_string(line) + " '" + error + "'";
  }

  const char* what() const noexcept override {
    return message_.c_str();
  }

 private:
  std::string message_;
};

} // namespace comms::prims::moe_ep::kernels

#ifndef EP_STATIC_ASSERT
#define EP_STATIC_ASSERT(cond, reason) static_assert(cond, reason)
#endif

#ifndef CUDA_CHECK
#define CUDA_CHECK(cmd)                                       \
  do {                                                        \
    cudaError_t e = (cmd);                                    \
    if (e != cudaSuccess) {                                   \
      throw ::comms::prims::moe_ep::kernels::EPException(     \
          "CUDA", __FILE__, __LINE__, cudaGetErrorString(e)); \
    }                                                         \
  } while (0)
#endif

#ifndef EP_HOST_ASSERT
#define EP_HOST_ASSERT(cond)                              \
  do {                                                    \
    if (!(cond)) {                                        \
      throw ::comms::prims::moe_ep::kernels::EPException( \
          "Assertion", __FILE__, __LINE__, #cond);        \
    }                                                     \
  } while (0)
#endif

#ifndef EP_DEVICE_ASSERT
// Device-side: only emit `printf`+`__trap()` when actually compiling the
// device pass. The host pass uses `abort()` so the macro stays well-formed
// in cross-platform contexts.
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#ifdef __HIP_PLATFORM_AMD__
#define EP_DEVICE_ASSERT(cond)                        \
  do {                                                \
    if (!(cond)) {                                    \
      printf(                                         \
          "Assertion failed: %s:%d, condition: %s\n", \
          __FILE__,                                   \
          __LINE__,                                   \
          #cond);                                     \
      abort();                                        \
    }                                                 \
  } while (0)
#else
#define EP_DEVICE_ASSERT(cond)                        \
  do {                                                \
    if (!(cond)) {                                    \
      printf(                                         \
          "Assertion failed: %s:%d, condition: %s\n", \
          __FILE__,                                   \
          __LINE__,                                   \
          #cond);                                     \
      asm("trap;");                                   \
    }                                                 \
  } while (0)
#endif
#else
// Host pass: no-op so the macro can appear unconditionally inside `__device__`
// + `__host__` annotated helpers without breaking the host compile.
#define EP_DEVICE_ASSERT(cond) ((void)0)
#endif
#endif
