if(NOT timemory_MAIN_PROJECT
   OR NOT TIMEMORY_BUILD_AVAIL
   OR TIMEMORY_BUILD_EXCLUDE_FROM_ALL)
    set(_EXCLUDE EXCLUDE_FROM_ALL)
endif()

set(_USE_TIMING 0)
if(TIMEMORY_BUILD_TESTING
   OR TIMEMORY_BUILD_MINIMAL_TESTING
   OR TIMEMORY_CI)
    set(_USE_TIMING 1)
endif()

add_executable(timemory-avail ${_EXCLUDE} ${CMAKE_CURRENT_LIST_DIR}/timemory-avail.cpp
                              ${CMAKE_CURRENT_LIST_DIR}/timemory-avail.hpp)

target_include_directories(timemory-avail PRIVATE ${CMAKE_CURRENT_LIST_DIR})

target_link_libraries(
    timemory-avail
    PRIVATE timemory::timemory-compile-options
            $<IF:${_USE_TIMING},timemory::timemory-compile-timing,>
            timemory::timemory-extensions
            timemory::timemory-headers
            timemory::timemory-python)

set_target_properties(timemory-avail PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)

install(
    TARGETS timemory-avail
    DESTINATION bin
    COMPONENT tools
    OPTIONAL)
