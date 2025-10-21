# Copyright (c) Meta Platforms, Inc. and affiliates.
# Extension: torchcomms._comms_gloo

include(FetchContent)
FetchContent_Declare(
  gloo
  GIT_REPOSITORY https://github.com/pytorch/gloo.git
  GIT_TAG        main
)
# fetch but don't build
FetchContent_Populate(gloo)

file(GLOB TORCHCOMMS_GLOO_SOURCES "comms/torchcomms/gloo/*.cpp")

add_library(torchcomms_comms_gloo MODULE ${TORCHCOMMS_GLOO_SOURCES})
set_target_properties(torchcomms_comms_gloo PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_comms_gloo"
    SUFFIX ".${Python3_SOABI}.so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/comms/torchcomms"
)
target_include_directories(torchcomms_comms_gloo PRIVATE
    ${ROOT}
    ${FMT_INCLUDE}
    ${CONDA_INCLUDE}
    ${Python3_INCLUDE_DIRS}
    ${gloo_SOURCE_DIR}
)
target_compile_features(torchcomms_comms_gloo PRIVATE cxx_std_20)
target_link_directories(torchcomms_comms_gloo PRIVATE ${CONDA_LIB})
target_link_libraries(torchcomms_comms_gloo PRIVATE
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIB}
    torchcomms
)
if(USE_SYSTEM_LIBS)
    target_link_libraries(torchcomms_comms_gloo PRIVATE
        "-lfmt"
    )
else()
    target_link_libraries(torchcomms_comms_gloo PRIVATE
        "-l:libfmt.a"
    )
endif()

install(TARGETS torchcomms_comms_gloo
    LIBRARY DESTINATION .
)
