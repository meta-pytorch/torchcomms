# Copyright (c) Meta Platforms, Inc. and affiliates.

uniflow_add_test(
  uniflow_topology_integration_test TopologyTest.cpp
  "cuda;nvlink;rdma;multi_gpu")
target_compile_definitions(uniflow_topology_integration_test PRIVATE
  UNIFLOW_TOPOLOGY_TEST_CUDA=1)
