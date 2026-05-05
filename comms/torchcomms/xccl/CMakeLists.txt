# Extension: torchcomms._comms_xccl

# Check if CCL_ROOT is set
if(NOT DEFINED ENV{CCL_ROOT})
    message(
        FATAL_ERROR
        "oneCCL environment not found (CCL_ROOT not set). Source the oneCCL vars.sh before building with USE_XCCL=ON."
    )
endif()

# Set XCCL paths
set(XCCL_INCLUDE "$ENV{CCL_ROOT}/include")
set(XCCL_SHARED_LIB "$ENV{CCL_ROOT}/lib/libccl.so.2")

# Validate oneCCL installation
if(NOT EXISTS "${XCCL_INCLUDE}" OR NOT EXISTS "${XCCL_SHARED_LIB}")
    message(FATAL_ERROR "Invalid oneCCL path: ${XCCL_INCLUDE} or ${XCCL_SHARED_LIB} does not exist.")
endif()

message(STATUS "XCCL include path    : ${XCCL_INCLUDE}")
message(STATUS "XCCL library         : ${XCCL_SHARED_LIB}")

file(GLOB TORCHCOMMS_XCCL_SOURCES "comms/torchcomms/xccl/*.cpp")
file(GLOB TORCHCOMMS_XPU_API_SOURCE "comms/torchcomms/device/xpu/XpuApi.cpp")

include(FindPackageHandleStandardArgs)

add_library(torchcomms_comms_xccl MODULE
    ${TORCHCOMMS_XCCL_SOURCES}
    ${TORCHCOMMS_XPU_API_SOURCE}
)
set_target_properties(torchcomms_comms_xccl PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_comms_xccl"
    SUFFIX ".${Python3_SOABI}.so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/comms/torchcomms"
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)
target_include_directories(torchcomms_comms_xccl PRIVATE
    ${ROOT}
    ${XCCL_INCLUDE}
    ${CONDA_INCLUDE}
    ${Python3_INCLUDE_DIRS}
)
target_compile_features(torchcomms_comms_xccl PRIVATE cxx_std_20)
target_link_directories(torchcomms_comms_xccl PRIVATE ${CONDA_LIB})
target_link_libraries(torchcomms_comms_xccl PRIVATE
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIB}
    torchcomms
)

target_link_libraries(torchcomms_comms_xccl PRIVATE
    ${XCCL_SHARED_LIB}
)

SET(XCCL_COMPILE_OPTIONS -Wall -Wextra)

# Enable additional warnings and treat them as errors if XCCL_WERROR is set
set(XCCL_WERROR $ENV{XCCL_WERROR})
if(XCCL_WERROR)
    list(APPEND XCCL_COMPILE_OPTIONS -Wpedantic -Werror)
endif()

message(STATUS "XCCL compile options : ${XCCL_COMPILE_OPTIONS}")

target_compile_options(torchcomms_comms_xccl PRIVATE ${XCCL_COMPILE_OPTIONS})

install(TARGETS torchcomms_comms_xccl
    LIBRARY DESTINATION .
)
