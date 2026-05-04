#
#
#

if(TIMEMORY_USE_PYTHON AND TIMEMORY_BUILD_PYTHON_HATCHET)
    timemory_checkout_git_submodule(
        RECURSIVE
        RELATIVE_PATH external/hatchet
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        REPO_URL https://github.com/jrmadsen/hatchet.git
        REPO_BRANCH timemory)

    if(NOT CYTHON_EXECUTABLE)
        find_package(Cython QUIET)
    endif()

    if(CYTHON_EXECUTABLE OR Cython_FOUND)
        timemory_message(STATUS "Adding external/hatchet...")
        add_subdirectory(hatchet)
    else()
        message(
            WARNING "Internal Hatchet could not built because Cython could not be found")
        set(TIMEMORY_BUILD_PYTHON_HATCHET
            OFF
            PARENT_SCOPE)
    endif()

    if(TARGET libpytimemory AND TARGET hatchet)
        add_dependencies(libpytimemory hatchet)
        if(CMAKE_INSTALL_PYTHONDIR)
            set(HATCHET_PYDIR "${CMAKE_CURRENT_LIST_DIR}/hatchet/hatchet")

            file(GLOB_RECURSE hatchet_files "${HATCHET_PYDIR}/*.py"
                 "${HATCHET_PYDIR}/*.js")

            file(GLOB_RECURSE hatchet_test_files "${HATCHET_PYDIR}/tests/*")

            if("${HATCHET_PYDIR}/tests/timemory_test.py" IN_LIST hatchet_test_files)
                list(REMOVE_ITEM hatchet_test_files
                     "${HATCHET_PYDIR}/tests/timemory_test.py")
            endif()

            set(HATCHET_CYTHONDIR "${HATCHET_PYDIR}/cython_modules/libs")

            set(hatchet_install_files ${hatchet_files} ${hatchet_test_files})
            list(REMOVE_DUPLICATES hatchet_install_files)
            foreach(_FILE ${hatchet_install_files})
                get_filename_component(_SUBDIR "${_FILE}" PATH)
                string(REPLACE "${CMAKE_CURRENT_LIST_DIR}/hatchet/" "" _SUBDIR
                               "${_SUBDIR}")
                install(
                    FILES ${_FILE}
                    DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/timemory/${_SUBDIR}
                    OPTIONAL)
            endforeach()

            if(NOT WIN32)
                execute_process(
                    COMMAND
                        ${CMAKE_COMMAND} -E create_symlink
                        ${PROJECT_BINARY_DIR}/timemory/hatchet
                        ${PROJECT_BINARY_DIR}/hatchet
                    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
            endif()
            # cython build modules
            install(
                FILES "${HATCHET_CYTHONDIR}/graphframe_modules${PYTHON_MODULE_EXTENSION}"
                      "${HATCHET_CYTHONDIR}/reader_modules${PYTHON_MODULE_EXTENSION}"
                DESTINATION
                    ${CMAKE_INSTALL_PYTHONDIR}/timemory/hatchet/cython_modules/libs
                OPTIONAL)
        endif()
    endif()
endif()

if(NOT TARGET hatchet)
    set(TIMEMORY_BUILD_PYTHON_HATCHET
        OFF
        PARENT_SCOPE)
endif()
