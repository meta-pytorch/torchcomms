
program fortran_example
    use timemory
    use iso_c_binding, only : C_INT64_T
    implicit none
    integer(C_INT64_T) :: idx

    ! initialize with explicit name
    call timemory_init_library("ex-fortran")

    ! initialize with name extracted from get_command_argument(0, ...)
    ! call timemory_init_library("")

    ! define the default set of components
    call timemory_set_default("wall_clock, cpu_clock")

    ! Start region "main"
    call timemory_push_region("main")

    ! Add peak_rss to the current set of components
    call timemory_add_components("peak_rss")

    ! Nested region "inner" nested under "main"
    call timemory_push_region("inner")

    ! End the "inner" region
    call timemory_pop_region("inner")

    ! remove peak_rss
    call timemory_remove_components("peak_rss")

    ! begin a region and get an identifier
    idx = timemory_get_begin_record("indexed")

    ! replace current set of components
    call timemory_push_components("page_rss")

    ! Nested region "inner" with only page_rss components
    call timemory_push_region("inner (pushed)")

    ! Stop "inner" region with only page_rss components
    call timemory_pop_region("inner (pushed)")

    ! restore previous set of components
    call timemory_pop_components()

    ! end the "indexed" region
    call timemory_end_record(idx)

    ! End "main"
    call timemory_pop_region("main")

    ! Output the results
    call timemory_finalize_library()

end program fortran_example
