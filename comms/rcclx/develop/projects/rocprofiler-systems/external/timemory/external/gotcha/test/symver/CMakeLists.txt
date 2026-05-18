set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/symver.map")
add_library(retX SHARED retX_old.c retX_new.c)

add_executable(symver_test test_symver.c)
target_link_libraries(symver_test retX gotcha)
gotcha_add_test(test_symbol_versions symver_test)
