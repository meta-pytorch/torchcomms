# Copyright (c) Meta Platforms, Inc. and affiliates.
# Extension: torchcomms._comms_nccl
file(GLOB TORCHCOMMS_NCCL_SOURCES "comms/torchcomms/nccl/*.cpp")
file(GLOB TORCHCOMMS_CUDA_API_SOURCE "comms/torchcomms/device/cuda/CudaApi.cpp")
find_package(CUDA)

if(USE_SYSTEM_LIBS)
    # Use system installed NCCL
    set(NCCL_INCLUDE "${CONDA_INCLUDE}")
    set(NCCL_SHARED_LIB "${CONDA_LIB}/libnccl.so")
elseif(DEFINED TORCHCOMMS_NCCL_INCLUDE AND DEFINED TORCHCOMMS_NCCL_LIBRARY)
    set(NCCL_INCLUDE "${TORCHCOMMS_NCCL_INCLUDE}")
    set(NCCL_SHARED_LIB "${TORCHCOMMS_NCCL_LIBRARY}")
elseif(EXISTS "${TORCH_INSTALL_PREFIX}/../build/nccl/include/nccl.h")
    # Use NCCL from PyTorch source build
    set(NCCL_INCLUDE "${TORCH_INSTALL_PREFIX}/../build/nccl/include")
    set(NCCL_STATIC_LIB
        "${TORCH_INSTALL_PREFIX}/../build/nccl/lib/libnccl_static.a"
    )
else()
    # Use NCCL packaged with PyTorch (pip install)
    set(NCCL_INCLUDE "${TORCH_INSTALL_PREFIX}/../nvidia/nccl/include")
    set(NCCL_SHARED_LIB
        "${TORCH_INSTALL_PREFIX}/../nvidia/nccl/lib/libnccl.so.2"
    )
endif()

add_library(torchcomms_comms_nccl MODULE
    ${TORCHCOMMS_NCCL_SOURCES}
    ${TORCHCOMMS_CUDA_API_SOURCE}
)
set_target_properties(torchcomms_comms_nccl PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_comms_nccl"
    SUFFIX ".${Python3_SOABI}.so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/comms/torchcomms"
)
target_include_directories(torchcomms_comms_nccl PRIVATE
    ${ROOT}
    ${NCCL_INCLUDE}
    ${CONDA_INCLUDE}
    ${Python3_INCLUDE_DIRS}
)
target_compile_features(torchcomms_comms_nccl PRIVATE cxx_std_20)

# Check whether the NCCL device API header is available (NCCL 2.28+). When
# present, define TORCHCOMMS_HAS_NCCL_DEVICE_API so TorchCommNCCL can
# compile the registerWithSymmMem path that publishes its ncclComm into
# pytorch's c10d::symmetric_memory::NCCLDevCommManager.
if(EXISTS "${NCCL_INCLUDE}/nccl_device.h")
    message(STATUS
        "NCCL device API header found, building TorchCommNCCL with "
        "symm_mem registration support")
    target_compile_definitions(torchcomms_comms_nccl
        PRIVATE TORCHCOMMS_HAS_NCCL_DEVICE_API=1)
else()
    message(STATUS
        "NCCL device API header NOT found (requires NCCL 2.28+), "
        "building TorchCommNCCL without symm_mem registration support")
endif()
target_link_directories(torchcomms_comms_nccl PRIVATE ${CONDA_LIB})
target_link_libraries(torchcomms_comms_nccl PRIVATE
    glog::glog_headers
    fmt::fmt
    torchcomms
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIB}
)
if(USE_SYSTEM_LIBS)
    target_link_libraries(torchcomms_comms_nccl PRIVATE
        ${NCCL_SHARED_LIB}
    )
elseif(DEFINED TORCHCOMMS_NCCL_INCLUDE AND DEFINED TORCHCOMMS_NCCL_LIBRARY)
    target_link_libraries(torchcomms_comms_nccl PRIVATE
        ${NCCL_SHARED_LIB}
    )
else()
    target_link_libraries(torchcomms_comms_nccl PRIVATE
        ${NCCL_STATIC_LIB}
    )
endif()

if(CUDA_FOUND)
    target_link_libraries(torchcomms_comms_nccl PRIVATE CUDA::cudart)
endif()

install(TARGETS torchcomms_comms_nccl
    LIBRARY DESTINATION .
)
