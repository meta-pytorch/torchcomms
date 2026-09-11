// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/ibverbx/IbvCommon.h"
#include <cstring>
#include <iostream>
#include <type_traits>

namespace ibverbx {

// Thread-safe strerror. Two incompatible strerror_r variants exist: the GNU one
// returns char* and may hand back a static string rather than filling the
// caller's buffer, while the XSI one returns int and always fills the buffer.
// Dispatch on the return type so this compiles and is correct under either.
std::string errnoStr(const int errNum) {
  char buf[256];
  buf[0] = '\0';
  if constexpr (std::is_same_v<
                    decltype(::strerror_r(errNum, buf, sizeof(buf))),
                    char*>) {
    return ::strerror_r(errNum, buf, sizeof(buf));
  } else {
    // Non-zero means the message was truncated or the errno was unknown; buf
    // still holds the best available text, so report it either way.
    (void)::strerror_r(errNum, buf, sizeof(buf));
    return buf;
  }
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
