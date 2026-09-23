#include "fpapi_test.h"

      program first
      IMPLICIT integer (p)
      
      integer event1
      INTEGER retval
      INTEGER*8 values(10)
      INTEGER*8 max, min
      INTEGER EventSet
      integer domain, granularity
      character*(PAPI_MAX_STR_LEN) domainstr, grnstr
      character*(PAPI_MAX_STR_LEN) name

      Integer last_char, n
      External last_char
      integer tests_quiet, get_quiet
      external get_quiet

      tests_quiet = get_quiet()
      EventSet = PAPI_NULL

      retval = PAPI_VER_CURRENT
      call PAPIf_library_init(retval)
      if ( retval.NE.PAPI_VER_CURRENT) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPI_library_init', retval)
      end if

      call PAPIf_query_event(PAPI_FP_INS, retval)
       if (retval .NE. PAPI_OK) then
        event1 = PAPI_TOT_INS 
       else
        event1 = PAPI_FP_INS 
       end if

      call PAPIf_create_eventset(EventSet, retval)
      if ( retval.NE.PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_create_eventset', 
     *retval)
      end if

      call PAPIf_add_event( EventSet, event1, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_add_event', 
     *retval)
      end if
 
      call PAPIf_add_event( EventSet, PAPI_TOT_CYC, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_add_event', 
     *retval)
      end if
      
      call PAPIf_start(EventSet, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_start', retval)
      end if

      call fdo_flops(NUM_FLOPS)

      call PAPIf_read(EventSet, values(1), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_read', retval)
      end if

      call PAPIf_reset(EventSet, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_reset', retval)
      end if

      call fdo_flops(NUM_FLOPS)

      call PAPIf_read(EventSet, values(3), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_read', retval)
      end if

      call fdo_flops(NUM_FLOPS)

      call PAPIf_read(EventSet, values(5), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_read', retval)
      end if

      call fdo_flops(NUM_FLOPS)

      call PAPIf_stop(EventSet, values(7), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_stop', retval)
      end if

      call PAPIf_read(EventSet, values(9), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_read', retval)
      end if

      if (tests_quiet .EQ. 0) then
      print *, 'TEST CASE 1: Non-overlapping start, stop, read.'
      print *, '--------------------------------------------------'//
     * '--------------------------------'
      end if

      call PAPIf_get_domain(EventSet, domain, PAPI_DEFDOM, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_get_domain', retval)
      end if

      call stringify_domain(domain, domainstr)
      if (tests_quiet .EQ. 0) then
      write (*,900) 'Default domain is:', domain, domainstr
 900  format(a20, i3, ' ', a70)
      end if
 
      call PAPIf_get_granularity(eventset, granularity, PAPI_DEFGRN, 
     *retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_get_granularity', 
     *retval)
      end if

      call stringify_granularity(granularity, grnstr)
      if (tests_quiet .EQ. 0) then
        call PAPIf_event_code_to_name (event1, name, retval)
        if ( retval.NE.PAPI_OK) then
            call ftest_fail(__FILE__, __LINE__, 
     * 'PAPIf_event_code_to_name', retval)
        end if
        n=last_char(name)
        write (*,800) 'Default granularity is:', granularity, grnstr
 800    format(a25, i3, ' ', a20)

        print *, 'Using', NUM_FLOPS, ' iterations of c += b*c'
        print *, '-----------------------------------------------'//
     *   '-----------------------------------'
 
        write (*,100) 'Test type', 1, 2, 3, 4, 5
        write (*,100) name(1:n), values(1), values(3), 
     *          values(5), values(7), values(9)

        write (*,100) 'PAPI_TOT_CYC', values(2), values(4), 
     *          values(6), values(8), values(10)

 100    format(a13, ':  ', i11, i11, i11, i11, i11)
        print *, '-----------------------------------------------'//
     *   '-----------------------------------'


        print *, 'Verification:'
        print *, 'Column 1 approximately equals column 2'
        print *, 'Column 3 approximately equals 2 * column 2'
        print *, 'Column 4 approximately equals 3 * column 2'
        print *, 'Column 4 exactly equals column 5'
      end if

      min = INT(REAL(values(3))*0.8)
      max = INT(REAL(values(3))*1.2)

      if ((values(1).gt.max) .OR. (values(1).lt.min) .OR. 
     *(values(5).gt.(max*2)) .OR. (values(5).lt.(min*2)) .OR.
     *(values(7).gt.(max*3)) .OR. (values(7).lt.(min*3)) .OR.
     *(values(7).NE.values(9))) then
        call ftest_fail(__FILE__, __LINE__,
     . name, 1)
      end if

      min = INT(REAL(values(4))*0.65)
      max = INT(REAL(values(4))*1.35)
      if ((values(2).gt.max) .OR. (values(2).lt.min) .OR. 
     *(values(6).gt.(max*2)) .OR. (values(6).lt.(min*2)) .OR.
     *(values(8).gt.(max*3)) .OR. (values(8).lt.(min*3)) .OR.
     *(values(8).NE.values(10))) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPI_TOT_CYC', 1)
      end if

      call ftests_pass(__FILE__)

      end
