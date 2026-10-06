# Copyright (c) Meta Platforms, Inc. and affiliates.

uniflow_add_unit_test(uniflow_topology_test TopologyTest.cpp)

if(UNIFLOW_ENABLE_RDMA)
  uniflow_add_unit_test(
    uniflow_topology_rdma_test TopologyRdmaTest.cpp)
endif()
