// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <pybind11/pybind11.h>

extern void ncclxSetIteration(int64_t);
extern int64_t ncclxGetIteration();

namespace ncclx {
// FIXME: we avoid using std::string to bypass pybind11 string format version
// mismatch issue for now. Better to fix it in the future.
extern bool setGlobalHint(const char* key, const char* val);
extern bool resetGlobalHint(const char* key);
} // namespace ncclx

__attribute__((noinline, visibility("default"))) bool ncclxSetHint(
    const char* name,
    const char* val) {
  return ncclx::setGlobalHint(name, val);
}

__attribute__((noinline, visibility("default"))) bool ncclxResetHint(
    const char* name) {
  return ncclx::resetGlobalHint(name);
}

PYBIND11_MODULE(ncclx_trainer_context, m) {
  m.doc() = "step counter python interface example";
  m.def("setIteration", &ncclxSetIteration);
  m.def("getIteration", &ncclxGetIteration);
  m.def("setHint", &ncclxSetHint);
  m.def("resetHint", &ncclxResetHint);
}
