#include "fpapi_test.h"

      program johnmay2
      implicit integer (p)

      integer*8 values(10)
      integer es, event
      integer retval
      character*PAPI_MAX_STR_LEN name

      Integer last_char, n
      External last_char
      integer tests_quiet, get_quiet
      external get_quiet

      tests_quiet = get_quiet()
      es = PAPI_NULL

      retval = PAPI_VER_CURRENT
      call PAPIf_library_init(retval)
      if ( retval.NE.PAPI_VER_CURRENT) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPI_library_init', retval)
      end if

      call PAPIf_query_event(PAPI_FP_INS, retval)
      if (retval.EQ.PAPI_OK) then
        event = PAPI_FP_INS
      else
        call PAPIf_query_event(PAPI_TOT_INS, retval)
        if ( retval.EQ.PAPI_OK) then
          event = PAPI_TOT_INS
        else
          call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_query_event', retval)
        end if
      end if

      call PAPIf_create_eventset(es, retval)
      if ( retval.NE.PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_create_eventset', 
     *retval)
      end if

      call PAPIf_add_event( es, event, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_add_event', retval)
      end if

      call PAPIf_start(es, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_start', retval)
      end if

      call PAPIf_cleanup_eventset(es, retval)
      if (retval .NE. PAPI_EISRUN) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_cleanup_eventset', 
     *retval)
      end if

      call PAPIf_destroy_eventset(es, retval)
      if (retval .NE. PAPI_EISRUN) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_destroy_eventset', 
     *retval)
      end if

      call fdo_flops(NUM_FLOPS)

      call PAPIf_stop(es, values(1), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_stop', retval)
      end if

      call PAPIf_destroy_eventset(es, retval)
      if (retval .NE. PAPI_EINVAL) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_destroy_eventset', 
     *retval)
      end if

      call PAPIf_cleanup_eventset(es, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_cleanup_eventset', 
     *retval)
      end if

      call PAPIf_destroy_eventset(es, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_destroy_eventset', 
     *retval)
      end if

      if (es .NE. PAPI_NULL) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_destroy_eventset', 
     *retval)
      end if

      if (tests_quiet .EQ. 0) then
        call PAPIf_event_code_to_name (event, name, retval)
        if ( retval.NE.PAPI_OK) then
          call ftest_fail(__FILE__, __LINE__, 
     * 'PAPIf_event_code_to_name', retval)
        end if
        n=last_char(name)
        print *, "Test case John May 2: cleanup / ",
     * "destroy eventset."
        print *, "--------------------------------",
     * "-----------------"
        print *, "Test run    : 1"
        print *, name(1:n), " : ", values(1)
        print *, "----------------------------------",
     * "---------------"
 
        print *, "Verification:"
        print *, "These error messages:"
        print *, "PAPI Error Code -10: PAPI_EISRUN: ", 
     * "EventSet is currently counting"
        print *, "PAPI Error Code -10: PAPI_EISRUN: ",
     * "EventSet is currently counting"
        print *, "PAPI Error Code -1: PAPI_EINVAL: ",
     * "Invalid argument"
      end if
      
      call ftests_pass(__FILE__)
      end
