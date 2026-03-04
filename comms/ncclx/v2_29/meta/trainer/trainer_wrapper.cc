// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <pybind11/pybind11.h>

#include "ncclx/trainer/trainer.h" // @manual

void setTrainingIterationWrapper(int64_t iter) {
  setTrainingIteration(iter);
}

int64_t getTrainingIterationWrapper(void) {
  return getTrainingIteration();
}

PYBIND11_MODULE(trainer_iteration_wrapper, m) {
  m.doc() = "step counter python interface example";
  m.def("setTrainingIteration", &setTrainingIterationWrapper);
  m.def("getTrainingIteration", &getTrainingIterationWrapper);
}
