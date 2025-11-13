# Copyright (c) Meta Platforms, Inc. and affiliates.
# Extension: torchcomms._comms_rcclx
file(GLOB TORCHCOMMS_RCCLX_SOURCES "comms/torchcomms/rcclx/*.cpp")

set(CONDA_INCLUDE "${CONDA_PREFIX}/include")
set(ROCM_HOME $ENV{ROCM_HOME})
if(NOT ROCM_HOME)
    set(ROCM_HOME "/opt/rocm")
endif()
set(ROCM_INCLUDE "${ROCM_HOME}/include")
set(RCCLX_INCLUDE $ENV{RCCLX_INCLUDE})
message(STATUS "RCCLX_INCLUDE is set to ${RCCLX_INCLUDE}")
set(RCCLX_SHARED_LIB $ENV{RCCLX_LIB}/librccl.so)
message(STATUS "RCCLX_SHARED_LIB is set to ${RCCLX_SHARED_LIB}")

set(FMT_INCLUDE "${ROOT}/third-party/fmt/include")

add_library(torchcomms_comms_rcclx MODULE ${TORCHCOMMS_RCCLX_SOURCES})
set_target_properties(torchcomms_comms_rcclx PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_comms_rcclx"
    SUFFIX ".${Python3_SOABI}.so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/comms/torchcomms"
)

target_include_directories(torchcomms_comms_rcclx PRIVATE
    ${ROOT}
    ${RCCLX_INCLUDE}
    ${FMT_INCLUDE}
    ${CONDA_INCLUDE}
    ${Python3_INCLUDE_DIRS}
    ${ROCM_INCLUDE}
    ${ROCM_INCLUDE}/hip
)
target_compile_features(torchcomms_comms_rcclx PRIVATE cxx_std_20)
target_link_directories(torchcomms_comms_rcclx PRIVATE ${CONDA_LIB})
target_link_libraries(torchcomms_comms_rcclx PRIVATE
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIB}
    torchcomms
)
target_link_libraries(torchcomms_comms_rcclx PRIVATE
    ${RCCLX_SHARED_LIB}
    "-lfmt"
)

set(CMAKE_CXX_COMPILE_OBJECT "${CMAKE_CXX_COMPILE_OBJECT} -std=c++20")

install(TARGETS torchcomms_comms_rcclx
        LIBRARY DESTINATION .
)
