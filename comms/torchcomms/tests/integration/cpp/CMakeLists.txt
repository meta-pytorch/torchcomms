# Copyright (c) Meta Platforms, Inc. and affiliates.
# Integration tests for torchcomms — built against conda environment.
#
# Builds ncclx_iterated_test binary. All sources are compiled directly into
# the executable because torchcomms_comms_ncclx is a MODULE target (Python
# extension loaded via dlopen) and cmake cannot link executables against
# MODULE libraries.
#
# ORDERING DEPENDENCY: This add_subdirectory() must appear after the ncclx
# include() in the top-level CMakeLists.txt, which sets NCCLX_INCLUDE,
# NCCLX_SHARED_LIB, FOLLY_LDFLAGS, TORCHCOMMS_HAS_NCCL_DEVICE_API,
# FMT_INCLUDE, and CONDA_LIB in parent scope.
#
# If linker errors for meta::comms::logger::* symbols appear, add:
#   ${ROOT}/comms/utils/logger/LogUtils.cc
#   ${ROOT}/comms/utils/logger/LoggingFormat.cc
# to NCCLX_BACKEND_SOURCES below.

# Verify NCCL device API headers exist before doing anything expensive.
# Gracefully skip when headers are missing (e.g., feedstock CI with standard NCCL).
if(NOT EXISTS "${NCCLX_INCLUDE}/nccl_device/core.h")
    message(WARNING
        "NCCL Device API headers not found at ${NCCLX_INCLUDE}/nccl_device/core.h. "
        "Skipping integration tests (requires NCCLX 2.28+ with device API headers).")
    return()
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# Test sources (relative to this directory)
set(INTEGRATION_TEST_DIR ${ROOT}/comms/torchcomms/tests/integration/cpp)
set(NCCLX_ITERATED_TEST_SOURCES
    ${INTEGRATION_TEST_DIR}/DeviceApiIteratedTest.cpp
    ${INTEGRATION_TEST_DIR}/DeviceApiIteratedTestMain.cpp
    ${INTEGRATION_TEST_DIR}/DeviceApiIteratedTestKernels.cu
    ${INTEGRATION_TEST_DIR}/IteratedTestHelpers.cpp
    ${INTEGRATION_TEST_DIR}/TorchCommTestHelpers.cpp
)

# Backend sources compiled directly — torchcomms_comms_ncclx is MODULE, cannot link
set(NCCLX_BACKEND_SOURCES
    ${ROOT}/comms/torchcomms/device/ncclx/NCCLDeviceBackend.cpp
    ${ROOT}/comms/torchcomms/device/cuda/CudaApi.cpp
    ${ROOT}/comms/torchcomms/device/cuda/AtomicAddKernel.cu
    ${ROOT}/comms/torchcomms/device/cuda/DeviceCounter.cpp
    ${ROOT}/comms/utils/CudaRAII.cc
)

add_executable(ncclx_iterated_test
    ${NCCLX_ITERATED_TEST_SOURCES}
    ${NCCLX_BACKEND_SOURCES}
)

target_include_directories(ncclx_iterated_test PRIVATE
    ${ROOT}
    ${NCCLX_INCLUDE}
    ${CONDA_INCLUDE}
    ${FMT_INCLUDE}
)

target_compile_features(ncclx_iterated_test PRIVATE cxx_std_20)

if(TORCHCOMMS_HAS_NCCL_DEVICE_API)
    target_compile_definitions(ncclx_iterated_test
        PRIVATE TORCHCOMMS_HAS_NCCL_DEVICE_API=1)
endif()

target_link_directories(ncclx_iterated_test PRIVATE ${CONDA_LIB})
target_link_libraries(ncclx_iterated_test PRIVATE
    torchcomms
    ${TORCH_LIBRARIES}
    ${NCCLX_SHARED_LIB}
    ${FOLLY_LDFLAGS}
    GTest::gtest
)
if(USE_SYSTEM_LIBS)
    target_link_libraries(ncclx_iterated_test PRIVATE
        "-lglog"
        "-lfmt"
        "-lboost_program_options"
        "-lboost_filesystem"
    )
else()
    target_link_libraries(ncclx_iterated_test PRIVATE
        "-l:libglog.a"
        "-l:libfmt.a"
    )
endif()
target_link_libraries(ncclx_iterated_test PRIVATE CUDA::cudart)

target_link_options(ncclx_iterated_test PRIVATE
    "LINKER:--allow-shlib-undefined"
)

# Install to libexec/torchcomms-tests/ relative to CMAKE_INSTALL_PREFIX.
# Note: CMAKE_INSTALL_PREFIX is set by setup.py to extdir.parent (pip's build
# temp), not $PREFIX. The feedstock build.sh handles copying the binary from
# the build tree to $PREFIX/libexec/torchcomms-tests/ post-install.
install(TARGETS ncclx_iterated_test
    RUNTIME DESTINATION libexec/torchcomms-tests
)
