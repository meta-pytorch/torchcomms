# Copyright (c) Meta Platforms, Inc. and affiliates.

uniflow_add_test(uniflow_topology_test TopologyTest.cpp unit)

if(UNIFLOW_ENABLE_RDMA)
  uniflow_add_test(
    uniflow_topology_rdma_test TopologyRdmaTest.cpp "unit;rdma")
endif()
