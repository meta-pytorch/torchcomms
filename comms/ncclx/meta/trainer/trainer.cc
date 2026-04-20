// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "trainer.h" // @manual

int64_t iteration = 0;

void setTrainingIteration(int64_t i) {
  iteration = i;
}

int64_t getTrainingIteration(void) {
  return iteration;
}
