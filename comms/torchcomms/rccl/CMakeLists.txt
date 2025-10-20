# Copyright (c) Meta Platforms, Inc. and affiliates.
# Extension: torchcomms._comms_rccl
file(GLOB TORCHCOMMS_RCCL_SOURCES "comms/torchcomms/rccl/*.cpp")

set(CONDA_INCLUDE "${CONDA_PREFIX}/include")
set(ROCM_HOME $ENV{ROCM_HOME})
if(NOT ROCM_HOME)
    set(ROCM_HOME "/opt/rocm")
endif()
set(ROCM_INCLUDE "${ROCM_HOME}/include")
set(RCCL_INCLUDE $ENV{RCCL_INCLUDE})
message(STATUS "RCCL_INCLUDE is set to ${RCCL_INCLUDE}")
set(RCCL_SHARED_LIB ${ROCM_HOME}/lib/librccl.so)
set(FMT_INCLUDE "${ROOT}/third-party/fmt/include")

add_library(torchcomms_comms_rccl MODULE ${TORCHCOMMS_RCCL_SOURCES})
set_target_properties(torchcomms_comms_rccl PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_comms_rccl"
    SUFFIX ".${Python3_SOABI}.so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/comms/torchcomms"
)

target_include_directories(torchcomms_comms_rccl PRIVATE
    ${ROOT}
    ${RCCL_INCLUDE}
    ${FMT_INCLUDE}
    ${CONDA_INCLUDE}
    ${Python3_INCLUDE_DIRS}
    ${ROCM_INCLUDE}
    ${ROCM_INCLUDE}/hip
)
target_compile_features(torchcomms_comms_rccl PRIVATE cxx_std_20)
target_link_directories(torchcomms_comms_rccl PRIVATE ${CONDA_LIB})
target_link_libraries(torchcomms_comms_rccl PRIVATE
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIB}
    torchcomms
)
target_link_libraries(torchcomms_comms_rccl PRIVATE
    ${RCCL_SHARED_LIB}
    "-lfmt"
)

set(CMAKE_CXX_COMPILE_OBJECT "${CMAKE_CXX_COMPILE_OBJECT} -std=c++20")

install(TARGETS torchcomms_comms_rccl
        LIBRARY DESTINATION .
)
