# Copyright (c) Meta Platforms, Inc. and affiliates.

if(UNIFLOW_ENABLE_TCP)
  add_subdirectory(../transport/tcp/tests/unit tcp_unit_tests)
  add_subdirectory(../transport/tcp/tests/integration tcp_integration_tests)
endif()

if(UNIFLOW_GPU_PLATFORM STREQUAL HIP)
  add_subdirectory(../transport/p2p/tests p2p_tests)
  uniflow_add_unit_test(uniflow_extension_multi_transport_test
    "${UNIFLOW_SOURCE_DIR}/tests/unit/MultiTransportTest.cpp")
  uniflow_add_gpu_test(uniflow_extension_cuda_api_test
    "${UNIFLOW_SOURCE_DIR}/drivers/cuda/tests/CudaApiTest.cpp" hip)
  uniflow_add_gpu_test(uniflow_extension_cuda_device_adapter_test
    "${UNIFLOW_SOURCE_DIR}/drivers/cuda/tests/CudaDeviceAdapterTest.cpp" unit)
  uniflow_add_gpu_test(uniflow_extension_multi_transport_single_host_test
    "${UNIFLOW_SOURCE_DIR}/tests/integration/MultiTransportSingleHostTest.cpp"
    "hip;rdma")
  if(UNIFLOW_ENABLE_RDMA)
    uniflow_add_gpu_test(uniflow_extension_rdma_registration_handle_test
      "${UNIFLOW_SOURCE_DIR}/transport/rdma/tests/unit/RdmaRegistrationHandleTest.cpp" unit)
    uniflow_add_gpu_test(uniflow_extension_rdma_resources_test
      "${UNIFLOW_SOURCE_DIR}/transport/rdma/tests/unit/RdmaResourcesTest.cpp" unit)
    uniflow_add_gpu_test(uniflow_extension_rdma_slab_pool_test
      "${UNIFLOW_SOURCE_DIR}/transport/rdma/tests/unit/RdmaSlabPoolTest.cpp" unit)
    uniflow_add_gpu_test(uniflow_extension_rdma_transport_test
      "${UNIFLOW_SOURCE_DIR}/transport/rdma/tests/unit/RdmaTransportTest.cpp" unit)
    uniflow_add_gpu_test(uniflow_extension_rdma_single_host_test
      "${UNIFLOW_SOURCE_DIR}/transport/rdma/tests/integration/SingleHostTest.cpp"
      "hip;rdma;multi_gpu")
    set_tests_properties(uniflow_extension_rdma_single_host_test PROPERTIES
      ENVIRONMENT "UNIFLOW_RDMA_TEST_NICS=${UNIFLOW_RDMA_TEST_NICS}")
  endif()
endif()
