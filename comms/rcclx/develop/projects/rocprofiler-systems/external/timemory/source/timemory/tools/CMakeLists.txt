set(NAME tools)

file(GLOB_RECURSE header_files ${CMAKE_CURRENT_SOURCE_DIR}/*.h
     ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB_RECURSE source_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

timemory_install_header_files(${header_files})

if(source_files)
    timemory_build_intermediate_library(
        USE_INTERFACE
        NAME ${tools}
        TARGET ${tools}
        CATEGORY TOOLS
        FOLDER ${CMAKE_INSTALL_LIBDIR}
        HEADERS ${header_files}
        SOURCES ${source_files}
        PROPERTY_DEPENDS TOOLS)
endif()
