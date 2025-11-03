# Extension: torchcomms._comms_xccl
file(GLOB TORCHCOMMS_XCCL_SOURCES "comms/torchcomms/xccl/*.cpp")
file(GLOB TORCHCOMMS_XPU_API_SOURCE "comms/torchcomms/device/XpuApi.cpp")

set(XCCL_INCLUDE "$ENV{CCL_ROOT}/include")
set(XCCL_SHARED_LIB "$ENV{CCL_ROOT}/lib/libccl.so.2")

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

install(TARGETS torchcomms_comms_xccl
    LIBRARY DESTINATION .
)
