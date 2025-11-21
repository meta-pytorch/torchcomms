// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

constexpr int kMaxNumRoles = 10;
struct WorkerGroupInfo {
  int start;
  int end;
  int numGroups;
  int numWorkers;
  int groupId;
  int workerId;
};
