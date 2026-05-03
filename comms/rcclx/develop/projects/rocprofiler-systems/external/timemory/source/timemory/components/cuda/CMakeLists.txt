if(TIMEMORY_USE_CUDA)
    set(NAME cuda)
    set(DEPS timemory::timemory-cuda timemory::timemory-cudart)

    file(GLOB_RECURSE header_files ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
    file(GLOB_RECURSE source_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
         ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

    timemory_build_intermediate_library(
        USE_INTERFACE
        NAME ${NAME}
        TARGET ${NAME}-component
        CATEGORY COMPONENT
        FOLDER ${CMAKE_INSTALL_LIBDIR}
        HEADERS ${header_files}
        SOURCES ${source_files}
        DEPENDS ${DEPS}
        PROPERTY_DEPENDS GLOBAL)
endif()
