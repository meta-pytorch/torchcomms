# Extension: torchcomms._comms_hccl

# Check if ASCEND_TOOLKIT_HOME is set
if(NOT DEFINED ENV{ASCEND_TOOLKIT_HOME})
    message(
        WARNING
        "HCCL environment not found (ASCEND_TOOLKIT_HOME not set), Skipping HCCL backend compilation."
    )
    return()
endif()

# Set HCCL paths
set(HCCL_INCLUDE "$ENV{ASCEND_TOOLKIT_HOME}/include")
set(HCCL_LIB_PATH "$ENV{ASCEND_TOOLKIT_HOME}/aarch64-linux/lib64")
set(HCCL_SHARED_LIB "${HCCL_LIB_PATH}/libhccl.so")
set(ACL_SHARED_LIB "${HCCL_LIB_PATH}/libascendcl.so")
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import os, torch_npu; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_VARIABLE TORCH_NPU_PKG_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
set(TORCH_NPU_INCLUDE "${TORCH_NPU_PKG_DIR}/include")
set(TORCH_NPU_LIB_DIR "${TORCH_NPU_PKG_DIR}/lib")
set(TORCH_NPU_SHARED_LIB "${TORCH_NPU_LIB_DIR}/libtorch_npu.so")

# Validate HCCL installation
if(NOT EXISTS "${HCCL_INCLUDE}" OR NOT EXISTS "${HCCL_SHARED_LIB}")
    message(WARNING "Invalid HCCL path. Skipping HCCL backend compilation.")
    message(STATUS "HCCL include path   : ${HCCL_INCLUDE}")
    message(STATUS "HCCL library        : ${HCCL_SHARED_LIB}")
    return()
endif()

message(STATUS "HCCL include path   : ${HCCL_INCLUDE}")
message(STATUS "HCCL library        : ${HCCL_SHARED_LIB}")
message(STATUS "ACL library         : ${ACL_SHARED_LIB}")
message(STATUS "Torch NPU include   : ${TORCH_NPU_INCLUDE}")
message(STATUS "Torch NPU library   : ${TORCH_NPU_SHARED_LIB}")

file(GLOB TORCHCOMMS_HCCL_SOURCES "comms/torchcomms/hccl/*.cpp")
file(GLOB TORCHCOMMS_NPU_API_SOURCE "comms/torchcomms/device/npu/NpuApi.cpp")

include(FindPackageHandleStandardArgs)

add_library(torchcomms_comms_hccl MODULE
    ${TORCHCOMMS_HCCL_SOURCES}
    ${TORCHCOMMS_NPU_API_SOURCE}
)
set_target_properties(torchcomms_comms_hccl PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_comms_hccl"
    SUFFIX ".${Python3_SOABI}.so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/comms/torchcomms"
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)
target_include_directories(torchcomms_comms_hccl PRIVATE
    ${ROOT}
    ${HCCL_INCLUDE}
    ${CONDA_INCLUDE}
    ${Python3_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE}
)
target_compile_features(torchcomms_comms_hccl PRIVATE cxx_std_20)
target_link_directories(torchcomms_comms_hccl PRIVATE
    ${CONDA_LIB}
    ${TORCH_NPU_LIB_DIR}
)
target_link_libraries(torchcomms_comms_hccl PRIVATE
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIB}
    torchcomms
)

target_link_libraries(torchcomms_comms_hccl PRIVATE
    ${HCCL_SHARED_LIB}
    ${ACL_SHARED_LIB}
    ${TORCH_NPU_SHARED_LIB}
)

set_property(TARGET torchcomms_comms_hccl APPEND PROPERTY BUILD_RPATH
    "${TORCH_NPU_LIB_DIR}"
)
set_property(TARGET torchcomms_comms_hccl APPEND PROPERTY INSTALL_RPATH
    "${TORCH_NPU_LIB_DIR}"
)

install(TARGETS torchcomms_comms_hccl
    LIBRARY DESTINATION .
)
