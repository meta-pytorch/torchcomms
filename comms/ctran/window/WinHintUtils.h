// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef WIN_HINTS_H_
#define WIN_HINTS_H_

#include <string>
#include <unordered_map>

#include "comms/utils/commSpecs.h"

namespace meta::comms::hints {

typedef std::unordered_map<std::string, std::string> kvType;

class WinHintUtils {
 public:
  static void init(kvType& kv);
  static commResult_t
  set(const std::string& key, const std::string& val, kvType& kv);

  static const std::vector<std::string>& keys();
};

} // namespace meta::comms::hints

#endif /* WIN_HINTS_H_ */
