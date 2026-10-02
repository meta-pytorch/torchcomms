# Copyright (c) Meta Platforms, Inc. and affiliates.

uniflow_add_test(
  uniflow_rdma_single_host_test SingleHostTest.cpp
  "rdma;cuda;multi_gpu")
set_tests_properties(uniflow_rdma_single_host_test PROPERTIES
  ENVIRONMENT "UNIFLOW_RDMA_TEST_NICS=${UNIFLOW_RDMA_TEST_NICS}")
