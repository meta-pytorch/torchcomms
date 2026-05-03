add_library(num SHARED num.c)
add_library(num2 SHARED num2.c)
add_executable(test_dlopen test_dlopen.c)
set_target_properties(test_dlopen
        PROPERTIES COMPILE_FLAGS "-DLIB_NAME_RAW=\"\"${CMAKE_CURRENT_BINARY_DIR}/libnum.so\"\" -DLIB2_NAME_RAW=\"\"${CMAKE_CURRENT_BINARY_DIR}/libnum2.so\"\""
)
target_link_libraries(test_dlopen gotcha dl)
add_dependencies(test_dlopen num num2)
gotcha_add_test(dlopen_test test_dlopen)
environment_add(dlopen_test TEST "GOTCHA_DEBUG=3 LIBNUM_DIR=${CMAKE_CURRENT_BINARY_DIR}")
set_tests_properties(dlopen_test PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
