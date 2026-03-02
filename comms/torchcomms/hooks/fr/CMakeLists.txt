# Copyright (c) Meta Platforms, Inc. and affiliates.
# Extension: torchcomms.hooks.fr._fr
file(GLOB TORCHCOMMS_FR_SOURCES "comms/torchcomms/hooks/fr/FlightRecorder.cpp")
file(GLOB TORCHCOMMS_FR_PYTHON_SOURCES "comms/torchcomms/hooks/fr/FlightRecorderPy.cpp")

add_library(torchcomms_fr MODULE
    ${TORCHCOMMS_FR_SOURCES}
    ${TORCHCOMMS_FR_PYTHON_SOURCES}
)
set_target_properties(torchcomms_fr PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_fr"
    SUFFIX ".${Python3_SOABI}.so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/comms/torchcomms/hooks/fr"
)
target_include_directories(torchcomms_fr PRIVATE
    ${ROOT}
    ${CONDA_INCLUDE}
    ${Python3_INCLUDE_DIRS}
)
target_compile_features(torchcomms_fr PRIVATE cxx_std_20)
target_link_directories(torchcomms_fr PRIVATE ${CONDA_LIB})
target_link_libraries(torchcomms_fr PRIVATE
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIB}
    torchcomms
    nlohmann_json::nlohmann_json
)

install(TARGETS torchcomms_fr
    LIBRARY DESTINATION hooks/fr
)
