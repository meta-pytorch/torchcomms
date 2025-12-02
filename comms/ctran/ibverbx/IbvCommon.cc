// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/ibverbx/IbvCommon.h"
#include <folly/String.h>
#include <iostream>

namespace ibverbx {

Error::Error() : errNum(errno), errStr(folly::errnoStr(errno)) {}
Error::Error(int errNum) : errNum(errNum), errStr(folly::errnoStr(errNum)) {}
Error::Error(int errNum, std::string errStr)
    : errNum(errNum), errStr(std::move(errStr)) {}

std::ostream& operator<<(std::ostream& out, Error const& err) {
  out << err.errStr << " (errno=" << err.errNum << ")";
  return out;
}

} // namespace ibverbx
