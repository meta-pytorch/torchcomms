# Copyright (c) Meta Platforms, Inc. and affiliates.

uniflow_add_test(
  uniflow_topology_integration_test TopologyTest.cpp
  "cuda;nvlink;rdma;multi_gpu")
