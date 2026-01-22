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

# Build the dummy test backend as a shared library for dynamic loading tests
add_library(dummy_test_backend SHARED
    DummyTorchCommBackend.cpp
)
target_include_directories(dummy_test_backend PRIVATE
    ${ROOT}
    ${CONDA_INCLUDE}
)
target_compile_features(dummy_test_backend PRIVATE cxx_std_20)
target_link_libraries(dummy_test_backend PRIVATE
    torchcomms
    ${TORCH_LIBRARIES}
)
set_target_properties(dummy_test_backend PROPERTIES
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

# Add dependency so dummy backend is built before factory test runs
add_dependencies(TorchCommFactoryTest dummy_test_backend)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(TorchCommOptionsTest)

# For TorchCommFactoryTest, we need to set the environment variable
# pointing to the dummy backend library
add_test(NAME TorchCommFactoryTest COMMAND TorchCommFactoryTest)
set_tests_properties(TorchCommFactoryTest PROPERTIES
    ENVIRONMENT "DUMMY_TEST_BACKEND_LIB_PATH=$<TARGET_FILE:dummy_test_backend>"
)
