cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(timemory-CXX-Tuple-Example LANGUAGES C CXX)

set(EXE_NAME ex_cxx_tuple)
set(COMPONENTS compile-options threading analysis-tools OPTIONAL_COMPONENTS xml cxx)

option(USE_MPI "Enable MPI" ON)
if(USE_MPI)
    list(APPEND COMPONENTS mpi)
endif()

option(USE_PAPI "Enable PAPI" OFF)
if(USE_PAPI)
    list(APPEND COMPONENTS papi)
endif()

set(timemory_FIND_COMPONENTS_INTERFACE timemory-cxx-tuple-example)
find_package(timemory REQUIRED COMPONENTS ${COMPONENTS})

add_executable(${EXE_NAME} ${EXE_NAME}.cpp)
target_link_libraries(${EXE_NAME} timemory-cxx-tuple-example)
install(
    TARGETS ${EXE_NAME}
    DESTINATION bin
    OPTIONAL)
