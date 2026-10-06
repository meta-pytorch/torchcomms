# Copyright (c) Meta Platforms, Inc. and affiliates.

set(_UNIFLOW_RDMA_TEST_LABELS rdma multi_gpu)
if(UNIFLOW_GPU_PLATFORM STREQUAL CUDA)
  list(APPEND _UNIFLOW_RDMA_TEST_LABELS cuda)
endif()

uniflow_add_gpu_test(
  uniflow_rdma_single_host_test SingleHostTest.cpp
  "${_UNIFLOW_RDMA_TEST_LABELS}")
set_tests_properties(uniflow_rdma_single_host_test PROPERTIES
  ENVIRONMENT "UNIFLOW_RDMA_TEST_NICS=${UNIFLOW_RDMA_TEST_NICS}")
