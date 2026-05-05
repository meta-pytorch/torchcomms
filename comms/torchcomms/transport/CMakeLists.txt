# Copyright (c) Meta Platforms, Inc. and affiliates.
# Extension: torchcomms._transport (uniflow backend)

find_package(CUDA)
if(NOT CUDA_FOUND)
    message(
        FATAL_ERROR
        "Transport build requires CUDA. \
Install the CUDA toolkit or rerun with USE_TRANSPORT=OFF.")
endif()

if(NOT DEFINED ENV{NVCC_GENCODE})
    set(ENV{NVCC_GENCODE} "-gencode=arch=compute_90,code=sm_90")
    message(STATUS "ENV NVCC_GENCODE: $ENV{NVCC_GENCODE}")
endif()
set(NVCC_GENCODE $ENV{NVCC_GENCODE})
message(STATUS "NVCC_GENCODE: ${NVCC_GENCODE}")

if(NOT DEFINED ENV{TORCH_CUDA_ARCH_LIST})
    set(ENV{TORCH_CUDA_ARCH_LIST} "9.0a")
endif()
set(TORCH_CUDA_ARCH_LIST $ENV{TORCH_CUDA_ARCH_LIST})
message(STATUS "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}")
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:${NVCC_GENCODE}>)

# Build uniflow as a sub-project.
add_subdirectory(comms/uniflow)

add_library(torchcomms_transport MODULE
    comms/torchcomms/transport/UniflowTransportPy.cpp
)
set_target_properties(torchcomms_transport PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_transport"
    SUFFIX ".${Python3_SOABI}.so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/comms/torchcomms"
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH ON
)
target_include_directories(torchcomms_transport PRIVATE
    ${ROOT}
    ${CONDA_INCLUDE}
    ${Python3_INCLUDE_DIRS}
)
target_compile_features(torchcomms_transport PRIVATE cxx_std_20)
target_link_libraries(torchcomms_transport PRIVATE
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIB}
    uniflow
)

if(CUDA_FOUND)
    target_link_libraries(torchcomms_transport PRIVATE CUDA::cudart)
endif()

install(TARGETS torchcomms_transport
    LIBRARY DESTINATION .
)

# libuniflow.so is already in the output directory via LIBRARY_OUTPUT_DIRECTORY.
# Copy it to the install destination so it's included in the wheel.
install(FILES $<TARGET_FILE:uniflow> DESTINATION .)
