file(GLOB_RECURSE mpi_header_files ${CMAKE_CURRENT_SOURCE_DIR}/types/mpi/*.hpp)
file(GLOB_RECURSE mpi_source_files ${CMAKE_CURRENT_SOURCE_DIR}/types/mpi/*.cpp)
file(GLOB_RECURSE header_files ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB_RECURSE source_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

if(TIMEMORY_USE_MPI_INIT)
    timemory_build_intermediate_library(
        NAME mpi_init
        TARGET common
        CATEGORY GLOBAL
        FOLDER ${CMAKE_INSTALL_LIBDIR}
        HEADERS ${mpi_header_files}
        SOURCES ${mpi_source_files}
        DEPENDS timemory::timemory-mpi timemory::timemory-core
        PROPERTY_DEPENDS GLOBAL)
endif()

list(REMOVE_ITEM header_files ${mpi_header_files})
list(REMOVE_ITEM source_files ${mpi_source_files})

if(WIN32 OR APPLE)
    file(GLOB_RECURSE capability_files ${CMAKE_CURRENT_SOURCE_DIR}/capability.*)
    list(REMOVE_ITEM header_files ${capability_files})
    list(REMOVE_ITEM source_files ${capability_files})
endif()

timemory_build_intermediate_library(
    USE_INTERFACE INSTALL_SOURCES
    NAME backends
    TARGET common
    CATEGORY GLOBAL
    FOLDER ${CMAKE_INSTALL_LIBDIR}
    HEADERS ${header_files}
    SOURCES ${source_files}
    DEPENDS timemory::timemory-mpi timemory::timemory-core timemory::timemory-gotcha
    PROPERTY_DEPENDS GLOBAL)
