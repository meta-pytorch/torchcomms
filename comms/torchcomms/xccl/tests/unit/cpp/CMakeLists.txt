enable_testing()

add_library(TorchCommXCCLTestBase STATIC
    TorchCommXCCLTestBase.cpp
    ${ROOT}/comms/torchcomms/xccl/TorchCommXCCL.cpp
    ${ROOT}/comms/torchcomms/xccl/TorchWorkXCCL.cpp
    ${ROOT}/comms/torchcomms/xccl/TorchWorkXCCLQueue.cpp
    ${ROOT}/comms/torchcomms/xccl/TorchCommXCCLUtils.cpp
    ${ROOT}/comms/torchcomms/xccl/TorchCommXCCLBootstrap.cpp
    ${ROOT}/comms/torchcomms/xccl/XcclApi.cpp
    ${ROOT}/comms/torchcomms/xccl/../device/xpu/XpuApi.cpp
)
target_include_directories(TorchCommXCCLTestBase PRIVATE
    ${ROOT}
    ${CONDA_INCLUDE}
    ${XCCL_INCLUDE}
)
target_compile_features(TorchCommXCCLTestBase PRIVATE cxx_std_20)
target_link_directories(TorchCommXCCLTestBase PUBLIC ${CONDA_LIB})
target_link_libraries(TorchCommXCCLTestBase PUBLIC
    torchcomms
    ${TORCH_LIBRARIES}
    GTest::gmock
)
if(USE_SYSTEM_LIBS)
    target_link_libraries(TorchCommXCCLTestBase PUBLIC
        "-lglog"
        "-lgflags"
        "-lfmt"
    )
else()
    target_link_libraries(TorchCommXCCLTestBase PUBLIC
        "-l:libglog.a"
        "-l:libgflags.a"
        "-l:libfmt.a"
    )
endif()

add_executable(TorchCommXCCLTest TorchCommXCCLTest.cpp)
target_include_directories(TorchCommXCCLTest PRIVATE ${ROOT} ${CONDA_INCLUDE} ${XCCL_INCLUDE})
target_compile_features(TorchCommXCCLTest PRIVATE cxx_std_20)
target_link_libraries(TorchCommXCCLTest PRIVATE torchcomms TorchCommXCCLTestBase ${TORCH_LIBRARIES} GTest::gtest_main GTest::gmock)

add_executable(TorchCommXCCLBootstrapTest TorchCommXCCLBootstrapTest.cpp)
target_include_directories(TorchCommXCCLBootstrapTest PRIVATE ${ROOT} ${CONDA_INCLUDE} ${XCCL_INCLUDE})
target_compile_features(TorchCommXCCLBootstrapTest PRIVATE cxx_std_20)
target_link_libraries(TorchCommXCCLBootstrapTest PRIVATE torchcomms TorchCommXCCLTestBase ${TORCH_LIBRARIES} GTest::gtest_main GTest::gmock)

add_executable(TorchWorkXCCLQueueTest TorchWorkXCCLQueueTest.cpp)
target_include_directories(TorchWorkXCCLQueueTest PRIVATE ${ROOT} ${CONDA_INCLUDE} ${XCCL_INCLUDE})
target_compile_features(TorchWorkXCCLQueueTest PRIVATE cxx_std_20)
target_link_libraries(TorchWorkXCCLQueueTest PRIVATE torchcomms TorchCommXCCLTestBase ${TORCH_LIBRARIES} GTest::gtest_main GTest::gmock)

add_executable(HintParsingTest HintParsingTest.cpp)
target_include_directories(HintParsingTest PRIVATE ${ROOT} ${CONDA_INCLUDE} ${XCCL_INCLUDE})
target_compile_features(HintParsingTest PRIVATE cxx_std_20)
target_link_libraries(HintParsingTest PRIVATE torchcomms TorchCommXCCLTestBase ${TORCH_LIBRARIES} GTest::gtest_main GTest::gmock)

include(GoogleTest)
add_test(NAME TorchCommXCCLTest COMMAND TorchCommXCCLTest)
add_test(NAME TorchCommXCCLBootstrapTest COMMAND TorchCommXCCLBootstrapTest)
add_test(NAME TorchWorkXCCLQueueTest COMMAND TorchWorkXCCLQueueTest)
add_test(NAME HintParsingTest COMMAND HintParsingTest)
