# Explicit source list, never a glob. See README.md before adding to it.

add_library(observatory SHARED
    colltrace/LifecycleFeedRegistry.cc
)

target_compile_features(observatory PRIVATE cxx_std_20)
# PRIVATE: the target is exported, and an exported target may not carry a raw
# source path in its interface. Consumers get ${ROOT} from their own build.
target_include_directories(observatory PRIVATE ${ROOT})

# No PREFIX override: the default gives libobservatory.so, matching how the
# backend shared libraries name their artifacts.
set_target_properties(observatory PROPERTIES
    OUTPUT_NAME observatory
    SOVERSION 1
    POSITION_INDEPENDENT_CODE ON
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH ON
    # Default visibility, and deliberately no version script or -Bsymbolic:
    # consumers must resolve these symbols here rather than locally.
    CXX_VISIBILITY_PRESET default
    LINK_FLAGS "-Wl,--no-undefined"
)
