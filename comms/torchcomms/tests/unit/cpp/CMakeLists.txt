# Copyright (c) Meta Platforms, Inc. and affiliates.

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# Test utilities (e.g., CudaGraphTestUtils) — only needed for CUDA backends
if(USE_NCCL OR USE_NCCLX)
  add_subdirectory(
    ${ROOT}/comms/utils/test_utils
    ${CMAKE_CURRENT_BINARY_DIR}/test_utils)
endif()

# Build the fake test backend as a shared library for dynamic loading tests
add_library(fake_test_backend SHARED
    FakeTorchCommBackend.cpp
    ${ROOT}/comms/torchcomms/fake/TorchCommFake.cpp
)
target_include_directories(fake_test_backend PRIVATE
    ${ROOT}
    ${CONDA_INCLUDE}
)
target_compile_features(fake_test_backend PRIVATE cxx_std_20)
target_link_libraries(fake_test_backend PRIVATE
    torchcomms
    ${TORCH_LIBRARIES}
)
set_target_properties(fake_test_backend PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# TorchCommOptionsTest - basic options/utils tests
add_executable(TorchCommOptionsTest
    TorchCommOptionsTest.cpp
)
target_include_directories(TorchCommOptionsTest PRIVATE
    ${ROOT}
    ${CONDA_INCLUDE}
)
target_compile_features(TorchCommOptionsTest PRIVATE cxx_std_20)
target_link_libraries(TorchCommOptionsTest PRIVATE
    torchcomms
    ${TORCH_LIBRARIES}
    GTest::gtest_main
    GTest::gmock
    atomic
)
target_link_options(TorchCommOptionsTest PRIVATE
    "LINKER:--allow-shlib-undefined"
)

# TorchCommFactoryTest - factory and backend tests
add_executable(TorchCommFactoryTest
    TorchCommFactoryTest.cpp
)
target_include_directories(TorchCommFactoryTest PRIVATE
    ${ROOT}
    ${CONDA_INCLUDE}
)
target_compile_features(TorchCommFactoryTest PRIVATE cxx_std_20)
target_link_libraries(TorchCommFactoryTest PRIVATE
    torchcomms
    ${TORCH_LIBRARIES}
    GTest::gtest_main
    GTest::gmock
)
target_link_options(TorchCommFactoryTest PRIVATE
    "LINKER:--allow-shlib-undefined"
)

# Add dependency so fake backend is built before factory test runs
add_dependencies(TorchCommFactoryTest fake_test_backend)

# TorchCommOptionsTest's BackendWrapper tag tests load the fake backend
# dynamically via new_comm("fake_test"), so it also needs the fake backend
# built and its path exported through FAKE_TEST_BACKEND_LIB_PATH.
add_dependencies(TorchCommOptionsTest fake_test_backend)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(TorchCommOptionsTest
    PROPERTIES
    ENVIRONMENT "FAKE_TEST_BACKEND_LIB_PATH=$<TARGET_FILE:fake_test_backend>"
)

# For TorchCommFactoryTest, we need to set the environment variable
# pointing to the fake backend library
add_test(NAME TorchCommFactoryTest COMMAND TorchCommFactoryTest)
set_tests_properties(TorchCommFactoryTest PROPERTIES
    ENVIRONMENT "FAKE_TEST_BACKEND_LIB_PATH=$<TARGET_FILE:fake_test_backend>"
)
