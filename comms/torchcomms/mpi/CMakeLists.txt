# Copyright (c) Meta Platforms, Inc. and affiliates.
# Extension: torchcomms._comms_mpi

# MPI is expected to be installed system-wide (e.g. OpenMPI).
# Set MPI_HOME to point to the MPI installation prefix.
if(DEFINED ENV{MPI_HOME})
    set(MPI_HOME "$ENV{MPI_HOME}")
    set(MPI_C_COMPILER "${MPI_HOME}/bin/mpicc")
    set(MPI_CXX_COMPILER "${MPI_HOME}/bin/mpicxx")
endif()
find_package(MPI REQUIRED)

file(GLOB TORCHCOMMS_MPI_SOURCES "comms/torchcomms/mpi/*.cpp")

set(FMT_INCLUDE "${ROOT}/third-party/fmt/include")

add_library(torchcomms_comms_mpi MODULE ${TORCHCOMMS_MPI_SOURCES})
set_target_properties(torchcomms_comms_mpi PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_comms_mpi"
    SUFFIX ".${Python3_SOABI}.so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/comms/torchcomms"
    INSTALL_RPATH "${CONDA_LIB}"
    BUILD_RPATH "${CONDA_LIB}"
)
target_include_directories(torchcomms_comms_mpi PRIVATE
    ${ROOT}
    ${FMT_INCLUDE}
    ${CONDA_INCLUDE}
    ${Python3_INCLUDE_DIRS}
    ${MPI_CXX_INCLUDE_DIRS}
)
target_compile_features(torchcomms_comms_mpi PRIVATE cxx_std_20)
target_link_directories(torchcomms_comms_mpi PRIVATE ${CONDA_LIB})
target_link_libraries(torchcomms_comms_mpi PRIVATE
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIB}
    torchcomms
    MPI::MPI_CXX
)
if(USE_SYSTEM_LIBS)
    target_link_libraries(torchcomms_comms_mpi PRIVATE "-lfmt")
else()
    target_link_libraries(torchcomms_comms_mpi PRIVATE "-l:libfmt.a")
endif()

install(TARGETS torchcomms_comms_mpi
    LIBRARY DESTINATION .
)
