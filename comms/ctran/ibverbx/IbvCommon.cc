// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/ibverbx/IbvCommon.h"
#include <cerrno>
#include <cstring>
#include <iostream>
#include <string>

namespace ibverbx {
namespace {

// Two incompatible strerror_r variants exist, and which one a platform declares
// is not portably detectable with the preprocessor. Overload on the function
// pointer so the compiler picks, as folly::errnoStr does; `if constexpr`
// cannot, because outside a template both of its branches are type-checked even
// though one is discarded.
[[maybe_unused]] std::string invokeStrerrorR(
    int (*strerrorR)(int, char*, size_t),
    int errNum,
    char* buf,
    size_t bufLen) {
  // XSI: only a 0 return guarantees buf was filled. Linux reports failure as
  // -1, OSX/FreeBSD as EINVAL, so treat any non-zero as failure and synthesize
  // the message rather than returning whatever buf happens to hold.
  if (strerrorR(errNum, buf, bufLen) != 0) {
    return "Unknown error " + std::to_string(errNum) +
        " (strerror_r failed with error " + std::to_string(errno) + ")";
  }
  return buf;
}

[[maybe_unused]] std::string invokeStrerrorR(
    char* (*strerrorR)(int, char*, size_t),
    int errNum,
    char* buf,
    size_t bufLen) {
  // GNU: returns the message, which may be a static string rather than buf.
  return strerrorR(errNum, buf, bufLen);
}

} // namespace

std::string errnoStr(const int errNum) {
  // strerror_r may set errno, so callers can keep writing errnoStr(errno).
  const int savedErrno = errno;
  char buf[256];
  buf[0] = '\0';
  std::string result = invokeStrerrorR(::strerror_r, errNum, buf, sizeof(buf));
  errno = savedErrno;
  return result;
}

Error::Error() : errNum(errno), errStr(errnoStr(errno)) {}
Error::Error(int errNum) : errNum(errNum), errStr(errnoStr(errNum)) {}
Error::Error(int errNum, std::string errStr)
    : errNum(errNum), errStr(std::move(errStr)) {}

std::ostream& operator<<(std::ostream& out, Error const& err) {
  out << err.errStr << " (errno=" << err.errNum << ")";
  return out;
}

} // namespace ibverbx
