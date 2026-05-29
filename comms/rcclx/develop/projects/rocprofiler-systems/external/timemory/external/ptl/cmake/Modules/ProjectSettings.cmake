################################################################################
#
# Project settings
#
################################################################################

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
    set(CMAKE_BUILD_TYPE Release)
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" _CONFIG)

if(WIN32)
    set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ STL standard")
else(WIN32)
    set(CMAKE_CXX_STANDARD 11 CACHE STRING "C++ STL standard")
endif(WIN32)

ptl_add_feature(CMAKE_C_FLAGS_${_CONFIG} "C compiler build-specific flags")
ptl_add_feature(CMAKE_CXX_FLAGS_${_CONFIG} "C++ compiler build-specific flags")

################################################################################
#
#   installation directories
#
################################################################################

include(GNUInstallDirs)

# cmake installation folder
set(CMAKE_INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME})

# create the full path version and generic path versions
foreach(_TYPE DATAROOT CONFIG INCLUDE LIB BIN MAN DOC)
    # set the absolute versions
    if(NOT IS_ABSOLUTE "${CMAKE_INSTALL_${_TYPE}DIR}")
        set(CMAKE_INSTALL_FULL_${_TYPE}DIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_${_TYPE}DIR})
    else()
        set(CMAKE_INSTALL_FULL_${_TYPE}DIR ${CMAKE_INSTALL_${_TYPE}DIR})
    endif()

    # generic "PROJECT_INSTALL_" variables (used by documentation)"
    set(PROJECT_INSTALL_${_TYPE}DIR ${CMAKE_INSTALL_${TYPE}DIR})
    set(PROJECT_INSTALL_FULL_${_TYPE}DIR ${CMAKE_INSTALL_FULL_${TYPE}DIR})

    if(NOT DEFINED PTL_INSTALL_${_TYPE}DIR)
        set(PTL_INSTALL_${_TYPE}DIR "${CMAKE_INSTALL_${_TYPE}DIR}")
    endif()
    set(PTL_INSTALL_${_TYPE}DIR "${CMAKE_INSTALL_${_TYPE}DIR}" CACHE STRING
        "${_TYPE} Installation Path")
endforeach()

