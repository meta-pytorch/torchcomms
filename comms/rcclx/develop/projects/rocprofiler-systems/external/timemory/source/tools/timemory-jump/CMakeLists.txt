if(WIN32)
    return()
endif()

if(DEFINED TIMEMORY_BUILD_JUMP AND NOT TIMEMORY_BUILD_JUMP)
    return()
endif()

if(NOT BUILD_SHARED_LIBS)
    # if shared libraries are not build, dlopen will not be possible
    return()
endif()

if(NOT timemory_MAIN_PROJECT OR TIMEMORY_BUILD_EXCLUDE_FROM_ALL)
    set(_EXCLUDE EXCLUDE_FROM_ALL)
endif()

foreach(LIB_TYPE shared static)

    set(LIB_TARGET timemory-jump-${LIB_TYPE})
    string(TOUPPER "${LIB_TYPE}" UC_LIB_TYPE)

    timemory_build_library(
        PIC NO_CACHE_LIST ${_EXCLUDE}
        TYPE ${UC_LIB_TYPE}
        TARGET_NAME timemory-jump-${LIB_TYPE}
        OUTPUT_NAME timemory
        LANGUAGE CXX
        LINKER_LANGUAGE ${_LINKER_LANGUAGE}
        OUTPUT_DIR ${PROJECT_BINARY_DIR}/jump
        SOURCES ${CMAKE_CURRENT_LIST_DIR}/library.cpp)

    target_compile_definitions(timemory-jump-${LIB_TYPE} INTERFACE TIMEMORY_USE_EXTERN)

    target_link_libraries(
        timemory-jump-${LIB_TYPE} PRIVATE timemory::timemory-headers
                                          timemory::timemory-compile-options)

    if(TARGET timemory::timemory-environment-${LIB_TYPE})
        target_link_libraries(timemory-jump-${LIB_TYPE}
                              PRIVATE timemory::timemory-environment-${LIB_TYPE})
    endif()

    install(
        TARGETS timemory-jump-${LIB_TYPE}
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/timemory/jump
        EXPORT timemory-library-depends
        OPTIONAL)

    # build tree
    add_custom_command(
        TARGET timemory-jump-${LIB_TYPE}
        POST_BUILD
        BYPRODUCTS
            ${PROJECT_BINARY_DIR}/libtimemory-jump${CMAKE_${UC_LIB_TYPE}_LIBRARY_SUFFIX}
        COMMAND
            ${CMAKE_COMMAND} -E create_symlink
            jump/libtimemory${CMAKE_${UC_LIB_TYPE}_LIBRARY_SUFFIX}
            libtimemory-jump${CMAKE_${UC_LIB_TYPE}_LIBRARY_SUFFIX}
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        COMMENT
            "Creating library symlink libtimemory-jump${CMAKE_${UC_LIB_TYPE}_LIBRARY_SUFFIX}"
        )

    # install tree
    install(
        CODE "
if(EXISTS \"${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/timemory/jump/libtimemory${CMAKE_${UC_LIB_TYPE}_LIBRARY_SUFFIX}\")
    EXECUTE_PROCESS(
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            timemory/jump/libtimemory${CMAKE_${UC_LIB_TYPE}_LIBRARY_SUFFIX}
            libtimemory-jump${CMAKE_${UC_LIB_TYPE}_LIBRARY_SUFFIX}
        WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
endif()
    "
        OPTIONAL)

endforeach()
