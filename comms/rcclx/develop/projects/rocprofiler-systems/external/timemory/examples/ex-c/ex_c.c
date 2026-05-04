
#include "timemory/library.h"
#include "timemory/timemory.h"

int
main(int argc, char** argv)
{
    // configure settings
    int overwrite       = 0;
    int update_settings = 1;
    // default to flat-profile
    timemory_set_environ("TIMEMORY_FLAT_PROFILE", "ON", overwrite, update_settings);
    // force timing units
    overwrite = 1;
    timemory_set_environ("TIMEMORY_TIMING_UNITS", "msec", 1, update_settings);

    // initialize with cmd-line
    timemory_init_library(argc, argv);

    // check if inited, init with name
    if(!timemory_library_is_initialized())
        timemory_named_init_library("ex-c");

    // define the default set of components
    timemory_set_default("wall_clock, cpu_clock");

    // create a region "main"
    timemory_push_region("main");
    timemory_pop_region("main");

    // pause and resume collection globally
    timemory_pause();
    timemory_push_region("hidden");
    timemory_pop_region("hidden");
    timemory_resume();

    // Add/remove component(s) to the current set of components
    timemory_add_components("peak_rss");
    timemory_remove_components("peak_rss");

    // get an identifier for a region and end it
    uint64_t idx = timemory_get_begin_record("indexed");
    timemory_end_record(idx);

    // assign an existing identifier for a region
    timemory_begin_record("indexed/2", &idx);
    timemory_end_record(idx);

    // create region collecting a specific set of data
    timemory_begin_record_enum("enum", &idx, TIMEMORY_PEAK_RSS, TIMEMORY_COMPONENTS_END);
    timemory_end_record(idx);

    timemory_begin_record_types("types", &idx, "peak_rss");
    timemory_end_record(idx);

    // replace current set of components and then restore previous set
    timemory_push_components("page_rss");
    timemory_pop_components();

    timemory_push_components_enum(2, TIMEMORY_WALL_CLOCK, TIMEMORY_CPU_CLOCK);
    timemory_pop_components();

    // Output the results
    timemory_finalize_library();
    return 0;
}
