# Copyright (c) Meta Platforms, Inc. and affiliates.

uniflow_add_test(
  uniflow_multi_transport_single_host_test MultiTransportSingleHostTest.cpp
  "cuda;nvlink;rdma;multi_gpu")
