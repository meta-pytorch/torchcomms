// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef FB_TRAINER_H_H
#define FB_TRAINER_H_H

#include <cstdint>

extern int64_t iteration;

void setTrainingIteration(int64_t iteration);
int64_t getTrainingIteration(void);

#define ncclFbGetTrainerIteration() getTrainingIteration()

#endif
