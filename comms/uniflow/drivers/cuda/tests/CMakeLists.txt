# Copyright (c) Meta Platforms, Inc. and affiliates.

uniflow_add_unit_test(
  uniflow_cuda_device_adapter_test CudaDeviceAdapterTest.cpp)
uniflow_add_test(uniflow_cuda_api_test CudaApiTest.cpp cuda)
uniflow_add_test(
  uniflow_cuda_driver_api_test CudaDriverApiTest.cpp cuda)
