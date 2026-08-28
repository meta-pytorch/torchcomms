# Copyright (c) Meta Platforms, Inc. and affiliates.
# Extension: torchcomms._comms_rcclx
file(GLOB TORCHCOMMS_RCCLX_SOURCES "comms/torchcomms/rcclx/*.cpp")

# RcclxApiShardedRelay.cpp (real) and RcclxApiShardedRelayStub.cpp (ncclInternalError
# stubs) define the SAME DefaultRcclxApi::shardedRelay* symbols with no #ifdef
# guards, so exactly one must be compiled or the link fails with duplicate symbols.
# In buck this is a select() on the rccl constraint (see comms/torchcomms/rcclx/BUCK);
# here (OSS/wheel CMake path) it keys off USE_RCCLX_DEV, forwarded by setup.py.
# Default OFF -> stub (safe against rcclx-stable, which lacks the relay symbols);
# ON -> real impl, requires linking an rcclx-dev librccl that exports
# ncclShardedRelayMultiGroup*.
option(USE_RCCLX_DEV "Link the real rcclx-dev sharded-relay impl instead of the stub" OFF)
if(USE_RCCLX_DEV)
    list(FILTER TORCHCOMMS_RCCLX_SOURCES EXCLUDE REGEX "RcclxApiShardedRelayStub\\.cpp$")
    message(STATUS "rcclx: using real sharded-relay impl (RcclxApiShardedRelay.cpp)")
else()
    list(FILTER TORCHCOMMS_RCCLX_SOURCES EXCLUDE REGEX "RcclxApiShardedRelay\\.cpp$")
    message(STATUS "rcclx: using sharded-relay stub (RcclxApiShardedRelayStub.cpp)")
endif()

set(CONDA_INCLUDE "${CONDA_PREFIX}/include")
# User can override via ROCM_HOME environment variable; otherwise use
# ROCM_PATH set by LoadHIP.cmake, included via find_package(Torch).
if(DEFINED ENV{ROCM_HOME})
    set(ROCM_HOME $ENV{ROCM_HOME})
elseif(ROCM_PATH)
    set(ROCM_HOME ${ROCM_PATH})
else()
    message(FATAL_ERROR "ROCM_PATH is not set. Ensure PyTorch is installed and find_package(Torch) is called before this point.")
endif()
set(ROCM_INCLUDE "${ROCM_HOME}/include")
set(RCCLX_INCLUDE $ENV{RCCLX_INCLUDE})
message(STATUS "RCCLX_INCLUDE is set to ${RCCLX_INCLUDE}")
set(RCCLX_SHARED_LIB $ENV{RCCLX_LIB}/librccl.so)
message(STATUS "RCCLX_SHARED_LIB is set to ${RCCLX_SHARED_LIB}")

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
    ${CONDA_INCLUDE}
    ${Python3_INCLUDE_DIRS}
    ${ROCM_INCLUDE}
    ${ROCM_INCLUDE}/hip
)
torchcomms_use_torch_pybind11(torchcomms_comms_rcclx)
target_compile_features(torchcomms_comms_rcclx PRIVATE cxx_std_20)
target_link_directories(torchcomms_comms_rcclx PRIVATE ${CONDA_LIB})
target_link_libraries(torchcomms_comms_rcclx PRIVATE
    torchcomms
    glog::glog
    fmt::fmt
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIB}
    ${RCCLX_SHARED_LIB}
)

set(CMAKE_CXX_COMPILE_OBJECT "${CMAKE_CXX_COMPILE_OBJECT} -std=c++20")

install(TARGETS torchcomms_comms_rcclx
        LIBRARY DESTINATION .
)
