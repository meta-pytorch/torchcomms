# Copyright (c) Meta Platforms, Inc. and affiliates.

uniflow_add_test(
  uniflow_rdma_single_host_test SingleHostTest.cpp
  "rdma;cuda;multi_gpu")
