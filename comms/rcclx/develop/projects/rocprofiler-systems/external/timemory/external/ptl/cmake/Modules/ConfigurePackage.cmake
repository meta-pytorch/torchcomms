
################################################################################
#
#        PTL Package installation
#
################################################################################

include(CMakePackageConfigHelpers)

set(INCLUDE_INSTALL_DIR     ${PTL_INSTALL_INCLUDEDIR})
set(LIB_INSTALL_DIR         ${PTL_INSTALL_LIBDIR})

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/Templates/${PROJECT_NAME}Config.cmake.in
    ${PROJECT_BINARY_DIR}/installation/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION ${PTL_INSTALL_CONFIGDIR}
    INSTALL_PREFIX ${PTL_INSTALL_PREFIX}
    PATH_VARS
        INCLUDE_INSTALL_DIR
        LIB_INSTALL_DIR)

write_basic_package_version_file(
    ${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES ${PROJECT_BINARY_DIR}/installation/${PROJECT_NAME}Config.cmake
    ${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION ${PTL_INSTALL_CONFIGDIR})

install(FILES ${PROJECT_SOURCE_DIR}/cmake/Modules/FindTBB.cmake
    DESTINATION ${PTL_INSTALL_CONFIGDIR}/Modules)


set(BUILD_TREE ON)
configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/Templates/${PROJECT_NAME}Build.cmake.in
    ${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION ${PROJECT_BINARY_DIR}
    INSTALL_PREFIX ${PROJECT_BINARY_DIR}
    PATH_VARS
        INCLUDE_INSTALL_DIR
        LIB_INSTALL_DIR)
