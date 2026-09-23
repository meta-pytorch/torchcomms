#include "fpapi_test.h"

      program accum
      implicit integer (p)

      integer es1, number, i
      integer*8 values(10)
      integer events(2)
      character*PAPI_MAX_STR_LEN name
      integer retval
      integer tests_quiet, get_quiet
      external get_quiet
      integer last_char, n
      external last_char

      tests_quiet = get_quiet()
      es1 = PAPI_NULL

      retval = PAPI_VER_CURRENT
      call PAPIf_library_init(retval)
      if ( retval.NE.PAPI_VER_CURRENT) then
        call ftest_fail(__FILE__, __LINE__,
     .     'PAPI_library_init', retval)
      end if

      call PAPIf_create_eventset(es1, retval)
      if ( retval.NE.PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     .     'PAPIf_create_eventset', 
     *retval)
      end if

      number=2

      call PAPIf_query_event(PAPI_FP_INS, retval)
      if (retval .NE. PAPI_OK) then
        events(1) = PAPI_TOT_INS 
      else
        events(1) = PAPI_FP_INS 
      end if

      events(2) = PAPI_TOT_CYC
      call PAPIf_add_events( es1, events, number, retval )
      if ( retval.LT.PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     .     'PAPIf_add_events', retval)
      end if

      do i=1,10
        values(i)=0
      end do

      call PAPIf_start(es1, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .     'PAPIf_start', retval)
      end if

      call fdo_flops(NUM_FLOPS)

      call PAPIf_accum(es1, values(7), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .     'PAPIf_accum', retval)
      end if

      values(1)=values(7)
      values(2)=values(8)
      call PAPIf_stop(es1, values(3), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .     'PAPIf_stop', retval)
      end if

      call PAPIf_start(es1, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .     'PAPIf_start', retval)
      end if

      call fdo_flops(NUM_FLOPS)
      call PAPIf_accum(es1, values(7), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .     'PAPIf_accum', retval)
      end if

      values(5)=values(7)
      values(6)=values(8)

      call fdo_flops(NUM_FLOPS)

      call PAPIf_accum(es1, values(7), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .     'PAPIf_accum', retval)
      end if

      call fdo_flops(NUM_FLOPS)      

      call PAPIf_stop(es1, values(9), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .     'PAPIf_stop', retval)
      end if

      call PAPIf_remove_events( es1, events, number, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .     'PAPIf_remove_events', retval)
      end if

      if (tests_quiet .EQ. 0) then
        call PAPIf_event_code_to_name (events(1), name, retval)
        if ( retval.NE.PAPI_OK) then
            call ftest_fail(__FILE__, __LINE__, 
     *        'PAPIf_event_code_to_name', retval)
        end if
        n=last_char(name)
        print *, "Test case accum: Test of PAPI_add_events, ",
     *  "PAPI_remove_events, PAPI_accum"
        print *, "------------------------------------------",
     *    "------------------------"
        write (*,100) "Test type", 1, 2, 3, 4, 5
        write (*,100) name(1:n), values(1), values(3),
     *    values(5), values(7), values(9)
        write (*,100) "PAPI_TOT_CYC", values(2), values(4),
     *    values(6), values(8), values(10)
        print *, "------------------------------------------",
     *    "------------------------"
 100    format(a15, ":", i10, i10, i10, i10, i10)
        print *
        print *, "Verification:"
        print *, "Column 2 approximately equals to 0;"
        print *, "Column 3 approximately equals 2 * Column 1;"
        print *, "Column 4 approximately equals 3 * Column 1;"
        print *, "Column 5 approximately equals Column 1."
      end if
 
      call ftests_pass(__FILE__)
      end
