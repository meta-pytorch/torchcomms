# Copyright (c) Meta Platforms, Inc. and affiliates.

# Backend test dirs are pulled in via include() of the backend CMakeLists before
# the shared tests/unit/cpp subdirectory that normally fetches GoogleTest, so
# fetch it here if it has not been made available yet. FetchContent dedups by
# name, so this is a no-op when the shared dir already fetched it.
if(NOT TARGET GTest::gmock)
    include(FetchContent)
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

enable_testing()

# The bootstrap drives NCCL/CUDA only through injected NcclApi/CudaApi
# interfaces (both mocked here), so the test runs on CPU without a GPU. It
# compiles TorchCommNCCLBootstrap.cpp directly (the nccl backend ships as a
# Python MODULE, not a linkable library) and links libtorchcomms for the shared
# store/utility symbols. NCCL_INCLUDE is inherited from the parent nccl
# CMakeLists (headers only; no libnccl symbols are referenced).
add_executable(TorchCommNCCLBootstrapTest
    TorchCommNCCLBootstrapTest.cpp
    ${ROOT}/comms/torchcomms/nccl/tests/unit/cpp/mocks/CudaMock.cpp
    ${ROOT}/comms/torchcomms/nccl/TorchCommNCCLBootstrap.cpp
)
target_include_directories(TorchCommNCCLBootstrapTest PRIVATE
    ${ROOT}
    ${NCCL_INCLUDE}
    ${CONDA_INCLUDE}
)
target_compile_features(TorchCommNCCLBootstrapTest PRIVATE cxx_std_20)
target_link_libraries(TorchCommNCCLBootstrapTest PRIVATE
    torchcomms
    ${TORCH_LIBRARIES}
    GTest::gtest_main
    GTest::gmock
)
target_link_options(TorchCommNCCLBootstrapTest PRIVATE
    "LINKER:--allow-shlib-undefined"
)

include(GoogleTest)
add_test(NAME TorchCommNCCLBootstrapTest COMMAND TorchCommNCCLBootstrapTest)
